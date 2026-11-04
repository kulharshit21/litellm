from typing import Final

from pydantic import TypeAdapter

from litellm.redis.native_settings import NativeRedisSettings
from litellm.rust_bridge._native import NativeRedisClient

_KWARGS: Final = TypeAdapter(dict[str, object])


def native_client(settings: NativeRedisSettings | None) -> NativeRedisClient | None:
    if settings is None:
        return None
    return NativeRedisClient.connect(_KWARGS.validate_python(settings.redis_kwargs), namespace=settings.namespace)
