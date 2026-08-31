"""Method-owned model and trainer packages behind the shared train contract."""

from .registry import METHOD_REGISTRY, MethodSpec, method_spec

__all__ = ["METHOD_REGISTRY", "MethodSpec", "method_spec"]
