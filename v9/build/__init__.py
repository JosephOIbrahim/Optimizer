"""V9 build layer — the build-and-cache half of the optimizer.

Config tuning sets an env var and is a microsecond side-effect. Building is
the other mechanism: compile a TRT engine / install SageAttention / build a
CUTLASS kernel — minutes of work that yields a cached artifact needing
invalidation, background scheduling, and safe fallback.

This package owns:
  * EngineCacheManager / BuildArtifact — the driver-keyed, LRU, self-healing
    artifact cache (cache.py).
  * BackgroundCompiler / BackendRegistry — the idle-gated, preemptible
    compile worker (compiler.py).
"""
from __future__ import annotations

from v9.build.cache import BuildArtifact, EngineCacheManager
from v9.build.compiler import BackendRegistry, BackgroundCompiler, CompileJob

__all__ = [
    "BuildArtifact",
    "EngineCacheManager",
    "BackendRegistry",
    "BackgroundCompiler",
    "CompileJob",
]
