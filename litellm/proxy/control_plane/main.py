"""Entrypoint for `litellm --control-plane`: the same image and CLI, a different app. Refuses to serve unlicensed."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from fastapi import FastAPI

from litellm.proxy.auth.litellm_license import LicenseCheck
from litellm.proxy.control_plane.app import License, create_app
from litellm.proxy.control_plane.container import build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings, load_settings


@dataclass(frozen=True, slots=True)
class Tls:
    keyfile: str
    certfile: str


class Runner(Protocol):
    def __call__(
        self,
        app: FastAPI,
        *,
        host: str,
        port: int,
        tls: Tls | None,
        limit_concurrency: int | None,
        log_config: str | None,
        keepalive_timeout: int | None,
        max_request_header_size: int | None,
    ) -> None: ...


def _uvicorn(
    app: FastAPI,
    *,
    host: str,
    port: int,
    tls: Tls | None,
    limit_concurrency: int | None,
    log_config: str | None,
    keepalive_timeout: int | None,
    max_request_header_size: int | None,
) -> None:
    import uvicorn

    from litellm.constants import DEFAULT_MAX_REQUEST_HEADER_SIZE
    from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager

    keyfile: Final = None if tls is None else tls.keyfile
    certfile: Final = None if tls is None else tls.certfile
    uvicorn.run(
        app=app,
        host=host,
        port=port,
        server_header=False,
        timeout_graceful_shutdown=round(GracefulShutdownManager.get_timeout()),
        log_config=log_config,
        timeout_keep_alive=5 if keepalive_timeout is None else keepalive_timeout,
        h11_max_incomplete_event_size=(
            DEFAULT_MAX_REQUEST_HEADER_SIZE if max_request_header_size is None else max_request_header_size
        ),
        ssl_keyfile=keyfile,
        ssl_certfile=certfile,
        limit_concurrency=limit_concurrency,
    )


def build_app(settings: ControlPlaneSettings, license_check: License) -> FastAPI:
    return create_app(build_container(settings), license_check)


def serve(
    host: str,
    port: int,
    tls: Tls | None = None,
    settings: Callable[[], ControlPlaneSettings] = load_settings,
    license_check: Callable[[], License] = LicenseCheck,
    run: Runner = _uvicorn,
    limit_concurrency: int | None = None,
    log_config: str | None = None,
    keepalive_timeout: int | None = None,
    max_request_header_size: int | None = None,
) -> None:
    app: Final = build_app(settings(), license_check())
    run(
        app,
        host=host,
        port=port,
        tls=tls,
        limit_concurrency=limit_concurrency,
        log_config=log_config,
        keepalive_timeout=keepalive_timeout,
        max_request_header_size=max_request_header_size,
    )
