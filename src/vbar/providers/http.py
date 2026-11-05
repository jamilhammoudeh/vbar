"""JSON over HTTPS for the provider adapters: no redirects, plain HTTP only to this computer.

Redirects are refused because urllib turns a POST answered with 301, 302 or 303 into a GET to
the new address, which may be another host, and sends the same headers, the key among them.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import logging
import math
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import provider_error

logger = logging.getLogger(__name__)

# Rate limits and server errors are retried this many times after the first try.
MAX_RETRIES = 2
# Longest wait before a retry, s, whatever Retry-After asks for.
MAX_WAIT_S = 30.0
RETRIED_KINDS = frozenset({"rate", "server"})
# Largest response body read, bytes; a reply this long is far past any reply limit V-BAR sets.
MAX_BODY_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Answer:
    """A provider's HTTP answer: status, headers (names in lower case) and the parsed JSON body.

    `body` is None when the body is not JSON.
    """

    status: int
    headers: Mapping[str, str]
    body: Any


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_url(url: str) -> None:
    """Refuse a URL that is not HTTPS, or plain HTTP to this computer, before any connection.

    Parameters
    ----------
    url : str
        The request address.

    Raises
    ------
    ValueError
        If the address holds a user name or password, the scheme is not https, or it is http with
        a host other than localhost or a loopback address.
    """
    parts = urllib.parse.urlsplit(url)
    # urllib would send user:password as a header; the messages below quote the address.
    if "@" in parts.netloc:
        raise ValueError("the address must not hold a user name or password")
    if not parts.hostname:
        raise ValueError(f"no host in the address {url!r}")
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and _loopback(parts.hostname):
        return
    raise ValueError(f"{url!r} must use https; plain http is allowed only to this computer")


def check_key(key: str) -> None:
    """Refuse an API key that is empty or holds a control or non-ASCII character.

    http.client would refuse such a header itself, with an error that quotes the key.

    Raises
    ------
    ValueError
        With a message that does not quote the key.
    """
    if not key or not key.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise ValueError("the API key is empty or holds a control or non-ASCII character")


def check_reply_limit(max_output_tokens: int) -> None:
    """Refuse a reply limit that is not a positive integer.

    Raises
    ------
    ValueError
        If `max_output_tokens` is not a positive integer.
    """
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ValueError(f"max_output_tokens must be a positive integer, got {max_output_tokens!r}")


def token_count(usage: Any, name: str) -> int:
    """The whole count `usage[name]`, or 0 when it is missing or not an integer."""
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def error_field(body: Any, name: str) -> Any:
    """`body["error"][name]`, or None when the body has no such field."""
    error = body.get("error") if isinstance(body, dict) else None
    return error.get(name) if isinstance(error, dict) else None


def _opener(url: str) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = [
        _RefuseRedirects(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    ]
    if urllib.parse.urlsplit(url).scheme == "http":
        # A proxy from the environment would see a plain-HTTP request, key and all.
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def _lower(headers: Any) -> dict[str, str]:
    return {name.lower(): value for name, value in headers.items()} if headers else {}


def _parse(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None


def post_json(*, url: str, headers: Mapping[str, str], body: Any, timeout: float) -> Answer:
    """POST `body` as JSON and return the answer, whatever its status.

    Parameters
    ----------
    url : str
        The endpoint; https, or http to localhost or a loopback address.
    headers : Mapping of str to str
        Request headers, the authorization header among them.
    body : JSON-serializable
        The request body.
    timeout : float
        Seconds to wait for the connection and for each read.

    Returns
    -------
    Answer
        The status, headers and parsed body, for 2xx and error statuses alike.

    Raises
    ------
    ValueError
        If the URL is refused by `check_url`.
    ProviderError
        Kind "network" for a refused connection, a timeout, a dropped connection or a redirect;
        kind "tls" when the certificate cannot be verified; kind "server" for a body longer
        than `MAX_BODY_BYTES`.
    """
    check_url(url)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
        method="POST",
    )
    host = urllib.parse.urlsplit(url).hostname
    t0 = time.perf_counter()
    try:
        with _opener(url).open(request, timeout=timeout) as response:
            status, received = response.status, response.headers
            raw = response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as error:
        status, received = error.code, error.headers
        try:
            raw = error.read(MAX_BODY_BYTES + 1)
        except (OSError, http.client.HTTPException):
            raw = b""
        error.close()
    except urllib.error.URLError as error:
        tls = isinstance(error.reason, ssl.SSLCertVerificationError)
        logger.debug("POST to %s failed: %s", host, type(error.reason).__name__)
        raise provider_error("tls" if tls else "network") from None
    except ssl.SSLCertVerificationError:
        logger.debug("POST to %s failed: certificate not verified", host)
        raise provider_error("tls") from None
    except (OSError, http.client.HTTPException) as error:
        logger.debug("POST to %s failed: %s", host, type(error).__name__)
        raise provider_error("network") from None
    logger.debug("POST to %s: %d in %.2f s", host, status, time.perf_counter() - t0)
    if len(raw) > MAX_BODY_BYTES:
        logger.debug("answer from %s longer than %d bytes", host, MAX_BODY_BYTES)
        raise provider_error("server")
    answer = Answer(status, _lower(received), _parse(raw))
    if 300 <= answer.status < 400:
        logger.debug("redirect from %s refused", host)
        raise provider_error("network")
    return answer


def _wait(answer: Answer, attempt: int) -> float:
    """Seconds before retry `attempt` (from 0): Retry-After in seconds, else 1, 2, 4..."""
    given = answer.headers.get("retry-after", "")
    try:
        wait = float(given)
    except ValueError:
        wait = math.nan
    if not (math.isfinite(wait) and wait >= 0.0):
        wait = 2.0**attempt
    return min(wait, MAX_WAIT_S)


def post_with_retries(
    *,
    url: str,
    headers: Mapping[str, str],
    body: Any,
    timeout: float,
    classify: Callable[[Answer], str | None],
    sleep: Callable[[float], None] = time.sleep,
    model: str = "",
) -> Answer:
    """POST through `post_json`, retrying rate limits and server errors at most twice.

    Parameters
    ----------
    url, headers, body, timeout
        As in `post_json`.
    classify : callable
        Maps an answer to a kind of `MESSAGES`, or None for a usable answer.
    sleep : callable, optional
        Waits the given seconds between tries.
    model : str, optional
        The model name, for the "model" sentence.

    Returns
    -------
    Answer
        The first answer `classify` accepts.

    Raises
    ------
    ProviderError
        With the kind `classify` gives, once retries are spent or for a kind that is not retried,
        or as raised by `post_json`.
    """
    attempt = 0
    while True:
        answer = post_json(url=url, headers=headers, body=body, timeout=timeout)
        kind = classify(answer)
        if kind is None:
            return answer
        if kind not in RETRIED_KINDS or attempt == MAX_RETRIES:
            raise provider_error(kind, model=model, status=answer.status)
        wait = _wait(answer, attempt)
        logger.debug("answer %d; retrying in %.1f s", answer.status, wait)
        sleep(wait)
        attempt += 1
