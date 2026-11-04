use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, LazyLock, Mutex, PoisonError},
};

use litellm_config::{ConfiguredRedis, SecretSnapshot, Value};
use litellm_core_utils::settings::ProcessEnvironment;
use litellm_redis_client::{RedisClient, RedisEvents};
use litellm_redis_connect::{HostServices, SharedEnvironment};
use litellm_redis_types::Resolution;
use pyo3::{
    prelude::*,
    sync::MutexExt,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};

use crate::redis::events::{ServiceLoggingEvents, state_label};

/// The `redis_kwargs` dict `RedisCache` hands redis-py, split the way the resolver takes it. Plain
/// values are read like `litellm/_redis.py` reads them, and a `None` stays an explicit null that
/// hides its `REDIS_*` value. A key whose value is a Python object, such as `retry`, is kept by name
/// only, so the resolver can name the setting it cannot honor.
pub(crate) struct RedisSpec(ConfiguredRedis);

impl<'a, 'py> FromPyObject<'a, 'py> for RedisSpec {
    type Error = PyErr;

    fn extract(kwargs: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = kwargs.py();
        let plain = PyDict::new(py);
        let mut host_objects = BTreeSet::new();
        for (name, value) in kwargs.cast::<PyDict>()?.iter() {
            match value.is_none() || is_plain(&value) {
                true => plain.set_item(&name, value)?,
                false => {
                    host_objects.insert(name.extract::<String>()?);
                }
            }
        }
        let fields: BTreeMap<String, Value> =
            litellm_host_python::from_py_argument(plain.as_any())?;
        Ok(Self(
            ConfiguredRedis::redis_cache_kwargs(fields).with_host_objects(host_objects),
        ))
    }
}

fn is_plain(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyList>()
        || value.is_instance_of::<PyTuple>()
        || value.is_instance_of::<PyDict>()
}

struct Shared {
    resolved: Resolution,
    client: RedisClient,
}

static CLIENTS: LazyLock<Mutex<Vec<Shared>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// The one client for a resolved Redis config, so every family in this process shares its pool
/// and its breaker. Like redis-py, which never connects at construction, a config the resolver or
/// the connector cannot honor leaves the family on memory with a warning rather than failing the
/// `Router`.
fn shared_client(
    py: Python<'_>,
    spec: RedisSpec,
    namespace: Option<String>,
) -> PyResult<Option<RedisClient>> {
    let environment: SharedEnvironment = Arc::new(ProcessEnvironment);
    let secrets = match secret_snapshot(py, spec.0.secret_names())? {
        Ok(secrets) => secrets,
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let logger = crate::logger::capture(py);
    let resolved = match logger.scope(|| spec.0.resolve(&secrets, &*environment)) {
        Ok(resolution) => {
            logger.scope(|| {
                for ignored in &resolution.ignored {
                    litellm_tracing::warn!(
                        "Redis setting {} is not supported by the Rust client and is ignored: {}",
                        ignored.setting,
                        ignored.consequence
                    );
                }
            });
            Resolution {
                namespace,
                ..resolution
            }
        }
        Err(error) => return Ok(memory_only(py, &error)),
    };
    let runtime = litellm_host_python::runtime_handle()?;
    let events: Arc<dyn RedisEvents> = Arc::new(ServiceLoggingEvents::new(py, runtime.clone())?);
    let host = HostServices {
        events: Some(events),
        http: crate::http::host_client(py, litellm_http::ClientVariant::Provider)?,
        environment,
    };
    let mut clients = CLIENTS
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(shared) = clients.iter().find(|shared| shared.resolved == resolved) {
        return Ok(Some(shared.client.clone()));
    }
    let connected = {
        let _runtime = runtime.enter();
        litellm_redis_connect::connect(resolved.clone(), host)
    };
    let client = match connected {
        Ok(client) => client,
        Err(error) => {
            drop(clients);
            return Ok(memory_only(py, &error));
        }
    };
    clients.push(Shared {
        resolved,
        client: client.clone(),
    });
    Ok(Some(client))
}

/// The Rust client for one `RedisCache`, built once and handed to every native store, so a store
/// never sees the redis-py kwargs or the namespace.
#[pyclass(frozen, name = "NativeRedisClient")]
pub(crate) struct NativeRedisClient(
    #[expect(
        dead_code,
        reason = "the native stores built on this client read it, starting in #49525"
    )]
    pub(crate) RedisClient,
);

#[pymethods]
impl NativeRedisClient {
    /// `None` when Redis is not usable, which leaves every store given it on memory.
    #[staticmethod]
    #[pyo3(signature = (redis_kwargs, *, namespace = None))]
    fn connect(
        py: Python<'_>,
        redis_kwargs: RedisSpec,
        namespace: Option<String>,
    ) -> PyResult<Option<Self>> {
        Ok(shared_client(py, redis_kwargs, namespace)?.map(Self))
    }
}

/// The breaker state of every native Redis client this process built, so the Prometheus state
/// gauge counts them next to the Python `RedisCache` breakers.
#[pyfunction]
pub(crate) fn native_redis_breaker_states(py: Python<'_>) -> Vec<&'static str> {
    let clients = CLIENTS
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    clients
        .iter()
        .filter_map(|shared| shared.client.breaker_state())
        .map(|state| state_label(&state))
        .collect()
}

/// Every name the resolver may read, fetched once the way Python's `get_secret` fetches it: the
/// configured secret manager first, then the process environment.
fn secret_snapshot(
    py: Python<'_>,
    names: BTreeSet<String>,
) -> PyResult<Result<SecretSnapshot, litellm_secrets::Error>> {
    let source = crate::secrets::source(py)?;
    let owned: Vec<String> = names.iter().cloned().collect();
    let lookup = crate::execution::run_sync_value(py, async move {
        let borrowed: Vec<&str> = owned.iter().map(String::as_str).collect();
        Ok(source.resolve(&borrowed).await)
    })?;
    Ok(lookup.map(|lookup| SecretSnapshot::read(names, &*lookup)))
}

fn memory_only(py: Python<'_>, error: &(dyn std::error::Error + 'static)) -> Option<RedisClient> {
    let report = litellm_error::Report::new(error).to_string();
    crate::logger::capture(py).scope(|| {
        litellm_tracing::warn!(
            "Redis is not usable from the Rust stores, their state stays in memory: {report}"
        )
    });
    None
}
