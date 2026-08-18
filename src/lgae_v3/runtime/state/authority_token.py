"""Authority capability token (v5.11 Phase 2).

The _AuthorityCapability is an internal token that must be presented
to mutate authoritative state. Exactly one instance is generated per
runtime. No external code can produce one.

This enforces physical exclusivity of the mutation path:

    ΔS_authority ⇒ valid authority capability

Without the capability, mutation methods raise UnauthorizedMutationError.
"""
from __future__ import annotations


class _AuthorityCapability:
    """Internal capability token for authoritative state mutation.

    This class is intentionally not exported in __all__. Only the
    runtime's internal authority layer can create instances.

    The token is compared by identity (is), not by value, so it cannot
    be forged or reconstructed.
    """

    __slots__ = ("_runtime_id",)

    def __init__(self, runtime_id: int) -> None:
        object.__setattr__(self, "_runtime_id", runtime_id)

    @property
    def runtime_id(self) -> int:
        return object.__getattribute__(self, "_runtime_id")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("_AuthorityCapability is immutable")

    def __delattr__(self, name: str) -> None:
        raise TypeError("_AuthorityCapability is immutable")

    def __repr__(self) -> str:
        return f"_AuthorityCapability(runtime_id={self.runtime_id})"
