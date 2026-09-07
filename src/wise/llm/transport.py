"""A deliberately small HTTP client that can only reach one local endpoint.

This is the whole network surface of :mod:`wise`. It is built on
:mod:`urllib.request` from the standard library — no new dependency — and it
is configured to be able to do almost nothing:

* the endpoint is validated when it is *configured*, not when it is used, and
  it must be a loopback literal unless a reviewed configuration says otherwise;
* the route is chosen from a declared allowlist, so no code path and certainly
  no model response can name a URL or a file;
* redirects are refused rather than followed — a redirect is how a local
  request becomes a remote one;
* the final response host is compared with the configured one, so a redirect
  that somehow happened still cannot deliver;
* inherited ``HTTP_PROXY``/``HTTPS_PROXY`` settings are dropped unless the
  configuration explicitly opts in, because a proxy is another way for a
  "local" call to leave the machine;
* the request has a byte budget, the response is read to its budget *plus one
  byte* and no further, so an endless reply is bounded rather than consumed;
* a timeout is mandatory, and a small bounded number of retries applies only
  to "not there" and "too slow", never to a boundary violation.

Nothing here starts a server, installs a model, or falls back to anything.
Absence is reported as :attr:`Outcome.UNAVAILABLE`; a boundary violation
raises :class:`~wise.errors.TransportError`.

The opener is injectable, which is how the tests exercise every path in this
module without opening a socket.

**Loopback is not confidentiality.** Binding to ``127.0.0.1`` means other
hosts cannot reach the port; it says nothing about other users on this
machine, about what the server logs, or about what the server itself may
contact. See ``docs/security/local-assistant.md``.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..errors import TransportError

#: The default endpoint: a loopback literal, never a name that could resolve
#: somewhere else, and never something a response can choose.
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

#: Host names accepted as loopback beside literal loopback addresses.
LOOPBACK_NAMES = ("localhost",)

#: The only routes this library ever posts to.
DEFAULT_ROUTES = ("/api/chat", "/api/embed")


class Outcome(str, Enum):
    """What became of one HTTP attempt."""

    OK = "ok"
    #: Connection refused, host unreachable, or the route answered 404 — the
    #: shape of "there is no server here" and "that model is not installed".
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    #: More bytes arrived than the budget allows; the rest was not read.
    OVERSIZE = "oversize"
    #: An HTTP status the caller has to interpret.
    HTTP_ERROR = "http_error"


def is_loopback(host: str) -> bool:
    """Whether ``host`` is a loopback literal or an accepted loopback name.

    A *name* is only accepted from :data:`LOOPBACK_NAMES`; nothing here
    resolves DNS, because resolution is exactly the step that could point a
    "local" endpoint somewhere else.

    >>> is_loopback("127.0.0.1"), is_loopback("::1"), is_loopback("localhost")
    (True, True, True)
    >>> is_loopback("10.0.0.5"), is_loopback("ollama.internal")
    (False, False)
    """
    text = str(host).strip().strip("[]")
    if text.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def check_endpoint(endpoint: str, *, allow_non_loopback: bool = False, reviewed_hosts: Sequence[str] = ()) -> str:
    """Validate a configured endpoint, returning its canonical origin.

    Raises :class:`~wise.errors.TransportError` for anything that is not a
    bare ``scheme://host[:port]`` origin on the loopback interface: another
    scheme, embedded credentials, a path, a query, a fragment, or a
    non-loopback host without both ``allow_non_loopback`` and an explicit
    entry in ``reviewed_hosts``.

    >>> check_endpoint("http://127.0.0.1:11434")
    'http://127.0.0.1:11434'
    >>> check_endpoint("http://example.invalid")
    Traceback (most recent call last):
        ...
    wise.errors.TransportError: endpoint 'http://example.invalid': host 'example.invalid' is not loopback; a non-loopback deployment needs allow_non_loopback=True, the host listed in reviewed_hosts, and its own authentication and network controls
    """
    text = str(endpoint).strip()
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https"):
        raise TransportError(f"endpoint {text!r}: scheme must be http or https, got {parts.scheme!r}")
    if parts.username or parts.password:
        raise TransportError(f"endpoint {text!r}: credentials in a URL are not accepted")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise TransportError(f"endpoint {text!r}: give a bare origin (scheme://host:port); the route is chosen by this library")
    host = parts.hostname
    if not host:
        raise TransportError(f"endpoint {text!r}: no host")
    try:
        port = parts.port
    except ValueError as exc:
        raise TransportError(f"endpoint {text!r}: {exc}") from exc
    if port is not None and not (1 <= port <= 65535):
        raise TransportError(f"endpoint {text!r}: port {port} is out of range")
    if not is_loopback(host) and not (allow_non_loopback and host in tuple(reviewed_hosts)):
        raise TransportError(
            f"endpoint {text!r}: host {host!r} is not loopback; a non-loopback deployment needs "
            "allow_non_loopback=True, the host listed in reviewed_hosts, and its own authentication "
            "and network controls"
        )
    netloc = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{netloc}" + (f":{port}" if port is not None else "")


@dataclass(frozen=True)
class TransportConfig:
    """Everything the transport is allowed to do, fixed when it is built.

    >>> TransportConfig().origin
    'http://127.0.0.1:11434'
    >>> TransportConfig(endpoint="https://ollama.corp:443")
    Traceback (most recent call last):
        ...
    wise.errors.TransportError: endpoint 'https://ollama.corp:443': host 'ollama.corp' is not loopback; a non-loopback deployment needs allow_non_loopback=True, the host listed in reviewed_hosts, and its own authentication and network controls
    """

    endpoint: str = DEFAULT_ENDPOINT
    timeout_s: float = 30.0
    max_request_bytes: int = 262_144
    max_response_bytes: int = 262_144
    max_retries: int = 1
    retry_delay_s: float = 0.0
    allow_non_loopback: bool = False
    reviewed_hosts: tuple[str, ...] = ()
    trust_environment_proxy: bool = False
    routes: tuple[str, ...] = DEFAULT_ROUTES
    headers: dict[str, str] = field(default_factory=dict)
    user_agent: str = "wise-pm (local, read-only)"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reviewed_hosts", tuple(str(h) for h in self.reviewed_hosts))
        object.__setattr__(self, "routes", tuple(str(r) for r in self.routes))
        object.__setattr__(self, "headers", dict(self.headers))
        origin = check_endpoint(self.endpoint, allow_non_loopback=self.allow_non_loopback, reviewed_hosts=self.reviewed_hosts)
        object.__setattr__(self, "endpoint", origin)
        for route in self.routes:
            if not route.startswith("/") or ".." in route or "//" in route[1:]:
                raise TransportError(f"route {route!r} must be an absolute path with no traversal")
        for name in ("timeout_s", "max_request_bytes", "max_response_bytes"):
            if getattr(self, name) <= 0:
                raise TransportError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.max_retries < 0:
            raise TransportError(f"max_retries must be non-negative, got {self.max_retries!r}")
        if self.retry_delay_s < 0:
            raise TransportError(f"retry_delay_s must be non-negative, got {self.retry_delay_s!r}")

    @property
    def origin(self) -> str:
        """The canonical ``scheme://host[:port]`` this transport may reach."""
        return self.endpoint

    @property
    def host(self) -> str:
        return urlsplit(self.endpoint).hostname or ""

    @property
    def loopback_only(self) -> bool:
        return is_loopback(self.host)

    def url_for(self, route: str) -> str:
        """The full URL of a declared route.

        >>> TransportConfig().url_for("/api/chat")
        'http://127.0.0.1:11434/api/chat'
        >>> TransportConfig().url_for("/api/pull")
        Traceback (most recent call last):
            ...
        wise.errors.TransportError: route '/api/pull' is not one of ('/api/chat', '/api/embed'); this library never names a route a caller or a response supplied
        """
        if route not in self.routes:
            raise TransportError(
                f"route {route!r} is not one of {self.routes}; this library never names a route a caller or a response supplied"
            )
        return f"{self.endpoint}{route}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "loopback_only": self.loopback_only,
            "timeout_s": self.timeout_s,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_retries": self.max_retries,
            "allow_non_loopback": self.allow_non_loopback,
            "reviewed_hosts": list(self.reviewed_hosts),
            "trust_environment_proxy": self.trust_environment_proxy,
            "routes": list(self.routes),
        }


@dataclass(frozen=True)
class TransportResponse:
    """One bounded attempt: what came back, how much of it, and how long it took."""

    outcome: Outcome
    status_code: int | None = None
    body: bytes = b""
    reason: str = ""
    url: str = ""
    elapsed_s: float = 0.0
    attempts: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", Outcome(self.outcome))

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK

    def json(self) -> Any:
        """Parse the body as JSON, or raise :class:`~wise.errors.TransportError`."""
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError(f"{self.url}: response is not valid JSON ({exc})") from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect.

    Installed as a subclass so :func:`urllib.request.build_opener` uses it
    instead of the default follower. A redirect is the ordinary way a request
    aimed at loopback ends up somewhere else, so it is refused rather than
    followed, and the refusal names the location it would have gone to.
    """

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise TransportError(
            f"{req.full_url}: the server answered {code} redirecting to {newurl!r}; "
            "the strict local transport does not follow redirects"
        )

    # urllib routes 301/302/303/307/308 through http_error_* before
    # redirect_request; refuse there too so no status slips past.
    def http_error_301(self, req: Any, fp: Any, code: int, msg: str, headers: Any) -> Any:
        return self.redirect_request(req, fp, code, msg, headers, headers.get("location", "") or "")

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


class _Opener(Protocol):
    """The one method this module needs, shaped like :class:`~urllib.request.OpenerDirector`.

    Naming it makes the opener injectable, which is how the tests drive every
    path in this module without a socket.
    """

    def open(self, fullurl: Any, data: Any = None, timeout: float | None = None) -> Any: ...


def build_opener(config: TransportConfig) -> urllib.request.OpenerDirector:
    """An opener that cannot follow a redirect and, by default, has no proxy.

    :class:`urllib.request.ProxyHandler` with an empty mapping replaces the
    default handler that reads ``HTTP_PROXY`` and friends from the
    environment, so an inherited proxy cannot silently carry a "local" request
    off the machine. ``trust_environment_proxy=True`` restores the ordinary
    behaviour for a deployment that has reviewed it.

    An empty :class:`~urllib.request.ProxyHandler` registers no ``*_open``
    method at all, so it does not appear in ``opener.handlers``; what it does
    is stop :func:`urllib.request.build_opener` from installing the default
    handler that would have read the environment.

    >>> opener = build_opener(TransportConfig())
    >>> any(isinstance(h, _NoRedirect) for h in opener.handlers)
    True
    >>> [h for h in opener.handlers if isinstance(h, urllib.request.ProxyHandler) and h.proxies]
    []
    """
    handlers: list[urllib.request.BaseHandler] = [_NoRedirect()]
    if not config.trust_environment_proxy:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = []
    return opener


class StrictLocalTransport:
    """POST JSON to one declared route of one validated local endpoint.

    ``opener`` exists so the tests can drive every path in this class without
    a socket; leave it unset and the transport builds its own.

    >>> transport = StrictLocalTransport(TransportConfig())
    >>> transport.config.origin
    'http://127.0.0.1:11434'
    """

    def __init__(self, config: TransportConfig | None = None, *, opener: _Opener | None = None) -> None:
        self.config = config if config is not None else TransportConfig()
        self._opener = opener
        #: routes actually posted to, in order — provenance, not a transcript
        self.calls: list[str] = []

    @property
    def opener(self) -> _Opener:
        """The opener, built on first use. Building it opens no connection."""
        if self._opener is None:
            built: _Opener = build_opener(self.config)
            self._opener = built
        return self._opener

    def post_json(self, route: str, payload: Mapping[str, Any], *, timeout_s: float | None = None) -> TransportResponse:
        """One bounded POST. Absence is an outcome; a boundary breach raises."""
        url = self.config.url_for(route)
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        if len(body) > self.config.max_request_bytes:
            raise TransportError(
                f"{url}: request of {len(body)} bytes exceeds the configured limit of {self.config.max_request_bytes}"
            )
        timeout = float(self.config.timeout_s if timeout_s is None else timeout_s)
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": self.config.user_agent}
        headers.update(self.config.headers)
        started = time.monotonic()
        last = TransportResponse(Outcome.UNAVAILABLE, url=url, reason="no attempt was made")
        for attempt in range(1, self.config.max_retries + 2):
            self.calls.append(route)
            last = self._attempt(url, body, headers, timeout, attempt, started)
            if last.outcome not in (Outcome.UNAVAILABLE, Outcome.TIMEOUT):
                return last
            if attempt <= self.config.max_retries and self.config.retry_delay_s:
                time.sleep(self.config.retry_delay_s)
        return last

    # ------------------------------------------------------------- internals
    def _attempt(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout: float, attempt: int, started: float
    ) -> TransportResponse:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        limit = self.config.max_response_bytes
        try:
            with self.opener.open(request, timeout=timeout) as response:
                self._check_final_host(url, response)
                chunk = response.read(limit + 1)
                status = int(getattr(response, "status", 0) or 0)
                if len(chunk) > limit:
                    return TransportResponse(
                        Outcome.OVERSIZE,
                        status_code=status,
                        body=chunk[:limit],
                        reason=f"the reply exceeded {limit} bytes and was not read further",
                        url=url,
                        elapsed_s=time.monotonic() - started,
                        attempts=attempt,
                    )
                return TransportResponse(
                    Outcome.OK,
                    status_code=status,
                    body=chunk,
                    url=url,
                    elapsed_s=time.monotonic() - started,
                    attempts=attempt,
                )
        except TransportError:
            raise
        except urllib.error.HTTPError as exc:
            detail = b""
            try:
                detail = exc.read(limit + 1)[:limit]
            except Exception:  # pragma: no cover - a closed error body
                detail = b""
            outcome = Outcome.UNAVAILABLE if exc.code == 404 else Outcome.HTTP_ERROR
            return TransportResponse(
                outcome,
                status_code=int(exc.code),
                body=detail,
                reason=f"HTTP {exc.code}",
                url=url,
                elapsed_s=time.monotonic() - started,
                attempts=attempt,
            )
        except (TimeoutError, socket.timeout) as exc:  # noqa: UP041 - socket.timeout is an alias, kept for older openers
            return TransportResponse(
                Outcome.TIMEOUT,
                reason=f"no reply within {timeout}s ({exc})",
                url=url,
                elapsed_s=time.monotonic() - started,
                attempts=attempt,
            )
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError | socket.timeout):
                return TransportResponse(
                    Outcome.TIMEOUT,
                    reason=f"no reply within {timeout}s",
                    url=url,
                    elapsed_s=time.monotonic() - started,
                    attempts=attempt,
                )
            return TransportResponse(
                Outcome.UNAVAILABLE,
                reason=f"cannot reach {url}: {exc.reason}",
                url=url,
                elapsed_s=time.monotonic() - started,
                attempts=attempt,
            )
        except OSError as exc:
            return TransportResponse(
                Outcome.UNAVAILABLE,
                reason=f"cannot reach {url}: {exc}",
                url=url,
                elapsed_s=time.monotonic() - started,
                attempts=attempt,
            )

    def _check_final_host(self, url: str, response: Any) -> None:
        """Refuse a reply that came from anywhere but the configured origin."""
        final = getattr(response, "url", None) or getattr(response, "geturl", lambda: None)()
        if not final:
            return
        expected = urlsplit(url)
        actual = urlsplit(str(final))
        if (actual.scheme, actual.hostname, actual.port) != (expected.scheme, expected.hostname, expected.port):
            raise TransportError(
                f"{url}: the reply came from {final!r}, which is not the configured endpoint {self.config.origin!r}; refusing it"
            )


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_ROUTES",
    "LOOPBACK_NAMES",
    "Outcome",
    "StrictLocalTransport",
    "TransportConfig",
    "TransportResponse",
    "build_opener",
    "check_endpoint",
    "is_loopback",
]
