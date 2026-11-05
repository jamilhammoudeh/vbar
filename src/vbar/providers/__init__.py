"""Model providers reached over HTTP: the wire formats, the error kinds and the stored key.

A provider failure raises `ProviderError` with a `kind` from `MESSAGES` and that kind's sentence
as its only text, so neither the key nor the provider's own error text reaches a log or a screen.
Rate limits, network failures and server errors are also `InfrastructureError`, which the
supervisor study reruns instead of counting against the model.
"""

from __future__ import annotations

from ..agents.llm import InfrastructureError

MESSAGES = {
    "key": "The provider refused the API key; run `vbar setup` and paste a valid key.",
    "credit": (
        "The provider says the account has no credit left; add credit on its billing page and "
        "run again."
    ),
    "rate": "The provider is limiting requests; wait a minute and run again.",
    "network": "V-BAR could not reach the provider; check the internet connection and run again.",
    "model": (
        "The provider does not know the model '{m}'; run `vbar setup` and pick one from the list."
    ),
    "server": "The provider had an internal error; try again in a few minutes.",
    "tls": "Python could not verify the provider's certificate; see docs/setup.md.",
    "cap": "The ${x} cap was reached; the remaining runs were not flown.",
    "refused": (
        "The provider refused the request (status {status}); run `vbar setup --show` to check "
        "the model, and report it if the problem stays."
    ),
}
# Failures of the provider's service rather than of the request or the account.
INFRASTRUCTURE_KINDS = frozenset({"rate", "network", "server"})


class ProviderError(RuntimeError):
    """A provider refused or failed a request; `kind` is a key of `MESSAGES` other than "cap".

    Parameters
    ----------
    kind : str
        The failure's kind.
    model : str, optional
        The model name, shown in the "model" sentence.
    status : int, optional
        The HTTP status, shown in the "refused" sentence.

    Raises
    ------
    ValueError
        If `kind` is not a provider failure kind.
    """

    def __init__(self, kind: str, *, model: str = "", status: int = 0):
        if kind not in MESSAGES or kind == "cap":
            raise ValueError(f"unknown provider error kind {kind!r}")
        text = MESSAGES[kind]
        if kind == "model":
            text = text.format(m=model)
        elif kind == "refused":
            text = text.format(status=int(status))
        super().__init__(text)
        self.kind = kind


class ProviderUnavailable(ProviderError, InfrastructureError):
    """A rate limit, network failure or server error: the provider gave no answer to judge."""


def provider_error(kind: str, *, model: str = "", status: int = 0) -> ProviderError:
    """The error for `kind`: a `ProviderUnavailable` for the infrastructure kinds."""
    if kind in INFRASTRUCTURE_KINDS:
        return ProviderUnavailable(kind, model=model, status=status)
    return ProviderError(kind, model=model, status=status)


__all__ = [
    "INFRASTRUCTURE_KINDS",
    "MESSAGES",
    "ProviderError",
    "ProviderUnavailable",
    "provider_error",
]
