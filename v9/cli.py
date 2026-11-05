"""V9 — the single orchestrator + CLI entry point.

One orchestrator, one enum, one config (DESIGN_V9 §4.6). Subcommands:

    once               run one optimization cycle, then clean shutdown
    watch [--interval] loop the cycle forever (Ctrl-C to stop)
    doctor             honest health/capability report (see v9.doctor)
    benchmark          measured-speedup harness (see v9.benchmark)
    evolve             nightly genetic parameter evolution (stub for now)

A cycle is:
    gather(profile, config) -> attach orchestrator telemetry onto ctx
        -> [advisor.analyze() in parallel] -> aggregate() -> apply top-N

Degrade-safe throughout: a missing SDK disables a backend, a failing advisor
contributes zero recs, and `once` always returns a clean exit code.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from v9 import context as ctx_mod
from v9 import paths
from v9.apply import Applier, ApplyResult
from v9.config import Config, load_config
from v9.hardware import HardwareProfile, detect
from v9.recommend import Recommendation, aggregate

log = logging.getLogger("v9.cli")

_ADVISOR_SECTIONS = ("gpu", "system", "comfyui", "houdini", "ue5", "coordinator")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """Owns the profile, config, advisors, and build infrastructure for the
    lifetime of a run. `watch` reuses one Orchestrator across cycles so the
    advisors' in-memory state (ComfyUI hot-shape ring) and the background
    compiler persist."""

    def __init__(self, config: Config, top_n: int = 8,
                 dry_run: bool = False) -> None:
        self.config = config
        self.profile: HardwareProfile = config.profile
        self.top_n = top_n
        self.dry_run = dry_run

        # --- build infrastructure (degrade-safe construction) -------------
        self.cache = self._build_cache()
        self.registry = self._build_registry()
        self.compiler = self._build_compiler()
        self.applier = Applier(compiler=self.compiler, cache=self.cache,
                               registry=self.registry, config=config,
                               dry_run=dry_run)
        self.advisors = self._build_advisors()

        # startup hygiene: drop TRT plans built for a different driver
        self._invalidate_stale_cache()

    # ---- construction helpers (each never raises) --------------------
    def _build_cache(self) -> Any:
        try:
            from v9.build.cache import EngineCacheManager
            trt = self.config.get("tensorrt", {}) or {}
            max_gb = float(trt.get("cache_size_gb", 200))
            return EngineCacheManager(
                hardware=self.profile,
                max_disk_bytes=int(max_gb * 1024 ** 3),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("engine cache unavailable: %s", exc)
            return None

    def _build_registry(self) -> Any:
        try:
            from v9.accel.registry import build_default_registry
            return build_default_registry(self.profile, self.config)
        except Exception as exc:  # noqa: BLE001
            log.warning("backend registry unavailable: %s", exc)
            return None

    def _build_compiler(self) -> Any:
        if self.cache is None:
            return None
        try:
            from v9.build.compiler import BackgroundCompiler, BackendRegistry
            spec = (self.config.get("comfyui", {}) or {}).get(
                "speculative_compile", {}) or {}
            # bridge the accel registry into the compiler's expected interface:
            # both expose register()/get()/names(); accel registry lacks names()
            compiler_registry = BackendRegistry()
            if self.registry is not None:
                try:
                    for backend in self.registry.all_backends():
                        compiler_registry.register(backend)
                except Exception:  # noqa: BLE001
                    pass
            return BackgroundCompiler(
                engine_cache=self.cache,
                idle_threshold_pct=float(spec.get("only_when_gpu_idle_pct", 5)),
                idle_dwell_seconds=int(spec.get("idle_dwell_seconds", 30)),
                max_concurrent=int(spec.get("max_concurrent_compiles", 1)),
                registry=compiler_registry,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("background compiler unavailable: %s", exc)
            return None

    def _build_advisors(self) -> list:
        from v9.advisors.gpu import GpuAdvisor
        from v9.advisors.system import SystemAdvisor
        from v9.advisors.comfyui import ComfyUIAdvisor
        from v9.advisors.houdini import HoudiniAdvisor
        from v9.advisors.ue5 import UE5Advisor
        from v9.advisors.coordinator import CoordinatorAdvisor

        classes = {
            "gpu": GpuAdvisor,
            "system": SystemAdvisor,
            "comfyui": ComfyUIAdvisor,
            "houdini": HoudiniAdvisor,
            "ue5": UE5Advisor,
            "coordinator": CoordinatorAdvisor,
        }
        advisors = []
        for section, cls in classes.items():
            try:
                cfg = dict(self.config.get(section, {}) or {})
                # advisors that need TRT version/cache-budget read a nested
                # 'tensorrt' block; surface the global one if absent.
                cfg.setdefault("tensorrt", self.config.get("tensorrt", {}) or {})
                # the Houdini advisor's CUTLASS-Vellum rule reads the global
                # acceleration block; surface it so config opt-in reaches it.
                cfg.setdefault("acceleration",
                               self.config.get("acceleration", {}) or {})
                advisor = cls(config=cfg, cache_handle=self.cache,
                              quality_ledger=self._quality_ledger())
                advisors.append(advisor)
            except Exception as exc:  # noqa: BLE001
                log.warning("advisor %s failed to construct: %s", section, exc)
        return advisors

    def _quality_ledger(self) -> Any:
        """The quality module is function-based; advisors only call optional
        methods on the handle, so a None handle is the conservative default."""
        return None

    def _invalidate_stale_cache(self) -> None:
        if self.cache is None:
            return
        try:
            n = self.cache.invalidate_on_driver_change()
            if n:
                log.info("invalidated %d cache entries from a prior driver", n)
        except Exception as exc:  # noqa: BLE001
            log.debug("driver-change invalidation skipped: %s", exc)

    # ---- lifecycle ---------------------------------------------------
    def start(self) -> None:
        if self.compiler is not None:
            try:
                self.compiler.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("compiler failed to start: %s", exc)

    def shutdown(self) -> None:
        if self.compiler is not None:
            try:
                self.compiler.stop(timeout=10.0)
            except Exception as exc:  # noqa: BLE001
                log.warning("compiler failed to stop: %s", exc)

    # ---- one cycle ---------------------------------------------------
    def run_cycle(self) -> dict[str, Any]:
        ctx = ctx_mod.gather(profile=self.profile,
                             config=self.config.as_dict(),
                             timestamp=time.time())
        self._attach_compiler_telemetry(ctx)

        recs = self._run_advisors_parallel(ctx)
        ranked = aggregate(recs, min_confidence=0.0)
        results = self.applier.apply_all(ranked, top_n=self.top_n)
        return {
            "ctx": ctx,
            "recommendations": ranked,
            "results": results,
            "advisor_count": len(self.advisors),
            "rec_count": len(ranked),
        }

    def _attach_compiler_telemetry(self, ctx: Any) -> None:
        """The coordinator advisor reads compile_jobs_active / idle_dwell /
        usage_history off ctx. Supply them from the live compiler."""
        if self.compiler is None:
            return
        try:
            status = self.compiler.status()
            ctx.compile_jobs_active = dict(
                status.get("queue_depth_per_app", {}) or {})
            for job in status.get("current_jobs", []):
                app = job.get("app", "")
                if app:
                    ctx.compile_jobs_active[app] = \
                        ctx.compile_jobs_active.get(app, 0) + 1
            ctx.idle_dwell_seconds = float(
                status.get("idle_state", {}).get("dwell_seconds_observed", 0.0))
        except Exception as exc:  # noqa: BLE001
            log.debug("compiler telemetry attach skipped: %s", exc)

    def _run_advisors_parallel(self, ctx: Any) -> list[Recommendation]:
        recs: list[Recommendation] = []
        if not self.advisors:
            return recs
        with ThreadPoolExecutor(max_workers=len(self.advisors),
                                thread_name_prefix="advisor") as pool:
            futures = {pool.submit(a.analyze, ctx): a for a in self.advisors}
            for fut in as_completed(futures):
                advisor = futures[fut]
                try:
                    recs.extend(fut.result() or [])
                except Exception as exc:  # noqa: BLE001 — base guards already
                    log.warning("advisor %s analyze() raised in pool: %s",
                                getattr(advisor, "name", "?"), exc)
        return recs


# ---------------------------------------------------------------------------
# cycle reporting
# ---------------------------------------------------------------------------
def _print_cycle(report: dict[str, Any], cycle_index: int = 0) -> None:
    ctx = report["ctx"]
    ranked: list[Recommendation] = report["recommendations"]
    results: list[ApplyResult] = report["results"]
    applied = sum(1 for r in results if r.applied and not r.advisory)
    advisory = sum(1 for r in results if r.applied and r.advisory)
    failed = sum(1 for r in results if not r.applied)

    print(f"=== Cycle {cycle_index}: {ctx.workflow_mode} mode | "
          f"GPU {ctx.gpu_utilization_pct:.0f}% | "
          f"VRAM {ctx.vram_used_gb:.1f}/{ctx.vram_total_gb:.0f} GB | "
          f"running={[k for k, v in ctx.running.items() if v] or 'none'} ===")
    print(f"  {report['advisor_count']} advisors -> "
          f"{report['rec_count']} recommendations -> "
          f"applied {applied}, advised {advisory}, failed {failed}")
    for rec, res in zip(ranked, results):
        mark = "OK " if res.applied else "XX "
        tag = "advice" if res.advisory else "applied"
        if not res.applied:
            tag = "FAILED"
        print(f"  [{mark}] {rec.action.value:20s} {rec.target:34s} "
              f"({tag}) {res.message}")


# ---------------------------------------------------------------------------
# subcommand handlers
# ---------------------------------------------------------------------------
def _cmd_once(args: argparse.Namespace, config: Config) -> int:
    orch = Orchestrator(config, top_n=args.top_n, dry_run=args.dry_run)
    orch.start()
    try:
        report = orch.run_cycle()
        _print_cycle(report, cycle_index=1)
    finally:
        orch.shutdown()
    print("once: complete (clean shutdown).")
    return 0


def _cmd_watch(args: argparse.Namespace, config: Config) -> int:
    orch = Orchestrator(config, top_n=args.top_n, dry_run=args.dry_run)
    orch.start()
    interval = max(1.0, float(args.interval))
    print(f"watch: cycling every {interval:.0f}s (Ctrl-C to stop).")
    i = 0
    try:
        while True:
            i += 1
            report = orch.run_cycle()
            _print_cycle(report, cycle_index=i)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nwatch: interrupted; shutting down cleanly.")
    finally:
        orch.shutdown()
    return 0


def _cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    from v9 import doctor
    return doctor.run(config)


def _cmd_benchmark(args: argparse.Namespace, config: Config) -> int:
    from v9 import benchmark
    return benchmark.run(config, iterations=args.iterations)


def _cmd_pbench(args: argparse.Namespace, config: Config) -> int:
    from v9 import pipeline_benchmark
    return pipeline_benchmark.run(config, seqs=args.seqs, iters=args.iterations,
                                  tc_mode=args.tc_mode)


def _cmd_ibench(args: argparse.Namespace, config: Config) -> int:
    from v9 import image_benchmark
    return image_benchmark.run(config, model_path=args.model, prompt=args.prompt,
                               steps=args.steps, size=args.size,
                               tc_mode=args.tc_mode)


def _cmd_roofline(args: argparse.Namespace, config: Config) -> int:
    from v9 import hwmodel
    return hwmodel.run(config, measure=not args.no_measure,
                       gemm_n=args.gemm_n, iters=args.iterations)


def _cmd_analyze(args: argparse.Namespace, config: Config) -> int:
    from v9 import control
    return control.analyze(config, workload=args.workload,
                           precision=args.precision, iters=args.iterations)


def _cmd_autotune(args: argparse.Namespace, config: Config) -> int:
    from v9 import control
    return control.autotune(config, n=args.gemm_n, baseline=args.baseline)


def _cmd_cutlass(args: argparse.Namespace, config: Config) -> int:
    """Manage the CUTLASS ingredient: bootstrap | status | build."""
    profile = config.profile
    sm_arch = getattr(getattr(profile, "gpu", None), "sm_arch", None)
    op = args.op

    if op == "bootstrap":
        from v9.accel import cutlass_bootstrap as cb
        accel = (config.get("acceleration", {}) or {}).get("cutlass", {}) or {}
        tag = args.tag or str(accel.get("version_tag", "v3.8.0"))
        install_dir = accel.get("install_dir") or None
        print(f"cutlass bootstrap: cloning + verifying (sm_{sm_arch}, tag {tag})...")
        res = cb.bootstrap(sm_arch=sm_arch, install_dir=install_dir, tag=tag)
        for step in res.steps:
            print(f"  - {step}")
        print(res.summary())
        if res.ok:
            print("\nNext: set acceleration.cutlass.enabled: true in config.yaml, "
                  "then run `v9 benchmark` to prove the FP8 GEMM speedup.")
        return 0 if res.ok else 1

    if op == "status":
        from v9.accel.registry import build_default_registry
        reg = build_default_registry(profile, config)
        be = reg.get("cutlass_fp8_gemm")
        if be is None:
            print("cutlass: backend not registered.")
            return 1
        print(f"cutlass_fp8_gemm: enabled={be.enabled} version={be.version}")
        if not be.enabled:
            print(f"  reason: {getattr(be, '_unavailable_reason', 'unknown')}")
        return 0

    if op == "build":
        from v9.accel.vellum import make_accelerator, default_gemm
        acc = make_accelerator(config, profile)
        mnk = tuple(args.mnk) if args.mnk else default_gemm(config)
        print(f"cutlass build: Vellum/Pyro FP8 GEMM kernel for {mnk}...")
        res = acc.build(mnk)  # type: ignore[arg-type]
        print(f"  backend={res.used_backend} precision={res.precision}")
        print(f"  {res.message}")
        if res.artifact_path:
            print(f"  cached: {res.artifact_path}")
        return 0 if res.ok else 1

    print(f"unknown cutlass op: {op}")
    return 2


def _cmd_render(args: argparse.Namespace, config: Config) -> int:
    """Apply the recipe, then launch + track a Houdini/ComfyUI render."""
    import shlex
    from v9 import render
    extra = shlex.split(args.xargs) if args.xargs else []
    return render.run(args.app, args.file, config,
                      extra_args=extra,
                      dry_run=args.dry_run, out=args.out, frames=args.frames)


def _cmd_evolve(args: argparse.Namespace, config: Config) -> int:
    # Genetic evolution is a Phase-2 component; the harness exists but the
    # tuner is not ported yet. Be honest rather than fake a result.
    print("evolve: genetic tuner not yet ported to V9 (Phase 2). "
          "No evolution performed.")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="v9", description="Optimizer V9 — detect-don't-hardcode GPU "
        "optimization for VFX (ComfyUI / Houdini / UE5).")
    p.add_argument("--config", default=None,
                   help="path to config.yaml (defaults to v9/config.yaml; "
                        "runs headless on built-in defaults if absent)")
    p.add_argument("--log-level", default="INFO",
                   help="logging level (DEBUG/INFO/WARNING/ERROR)")
    sub = p.add_subparsers(dest="command")

    once = sub.add_parser("once", help="run one optimization cycle then exit")
    once.add_argument("--top-n", type=int, default=8,
                      help="apply at most N ranked recommendations")
    once.add_argument("--dry-run", action="store_true",
                      help="report what would change without mutating anything")
    once.set_defaults(func=_cmd_once)

    watch = sub.add_parser("watch", help="loop the cycle continuously")
    watch.add_argument("--interval", type=float, default=60.0,
                       help="seconds between cycles (default 60)")
    watch.add_argument("--top-n", type=int, default=8)
    watch.add_argument("--dry-run", action="store_true")
    watch.set_defaults(func=_cmd_watch)

    doc = sub.add_parser("doctor", help="honest health + capability report")
    doc.set_defaults(func=_cmd_doctor)

    bench = sub.add_parser("benchmark", help="measured-speedup harness")
    bench.add_argument("--iterations", type=int, default=20,
                       help="timed iterations per measurement")
    bench.set_defaults(func=_cmd_benchmark)

    pbench = sub.add_parser("pbench", help="end-to-end pipeline benchmark "
                            "(diffusion-shaped; measures Sage + torch.compile)")
    pbench.add_argument("--seqs", type=int, nargs="+", default=[1024, 4096],
                        help="sequence lengths (latent tokens) to measure")
    pbench.add_argument("--iterations", type=int, default=20,
                        help="timed iterations per measurement")
    pbench.add_argument("--tc-mode", dest="tc_mode", default="reduce-overhead",
                        help="torch.compile mode for the run "
                             "(reduce-overhead|max-autotune|default)")
    pbench.set_defaults(func=_cmd_pbench)

    ibench = sub.add_parser("ibench", help="REAL-image benchmark (SD1.5; "
                            "measured seconds/image + LPIPS)")
    ibench.add_argument("--model", default=None,
                        help="single-file checkpoint (.safetensors); "
                             "defaults to the local SD1.5")
    ibench.add_argument("--prompt", default=None, help="generation prompt")
    ibench.add_argument("--steps", type=int, default=25)
    ibench.add_argument("--size", type=int, default=512)
    ibench.add_argument("--tc-mode", dest="tc_mode", default="default",
                        help="torch.compile mode (default|reduce-overhead|max-autotune)")
    ibench.set_defaults(func=_cmd_ibench)

    roof = sub.add_parser("roofline", help="derive + measure the GPU roofline "
                          "(compute/memory ceilings + bound classifier)")
    roof.add_argument("--no-measure", action="store_true",
                      help="derive theoretical ceilings only; skip microbench")
    roof.add_argument("--gemm-n", type=int, default=8192,
                      help="square GEMM dimension for the compute-peak microbench")
    roof.add_argument("--iterations", type=int, default=30,
                      help="timed iterations per microbench")
    roof.set_defaults(func=_cmd_roofline)

    an = sub.add_parser("analyze", help="closed loop: profile a workload, "
                        "roofline-classify it, emit derived recommendations")
    an.add_argument("workload", nargs="?", default="dit",
                    help="gemm | elementwise | dit")
    an.add_argument("--precision", default="fp16", help="fp16|bf16|fp32")
    an.add_argument("--iterations", type=int, default=15)
    an.set_defaults(func=_cmd_analyze)

    at = sub.add_parser("autotune", help="roofline-guided precision search on a "
                        "GEMM, admitted by the Pareto gate")
    at.add_argument("--gemm-n", type=int, default=8192, help="square GEMM dim")
    at.add_argument("--baseline", default="fp16", help="baseline precision")
    at.set_defaults(func=_cmd_autotune)

    evolve = sub.add_parser("evolve", help="nightly genetic evolution (Phase 2)")
    evolve.set_defaults(func=_cmd_evolve)

    cut = sub.add_parser("cutlass",
                         help="manage the CUTLASS ingredient (bootstrap/status/build)")
    cut.add_argument("op", choices=["bootstrap", "status", "build"],
                     help="bootstrap=install+verify, status=capability, "
                          "build=compile+cache the Vellum FP8 GEMM kernel")
    cut.add_argument("--tag", default=None, help="CUTLASS git tag (default from config)")
    cut.add_argument("--mnk", type=int, nargs=3, metavar=("M", "N", "K"),
                     default=None, help="GEMM dims for `build` (default from config)")
    cut.set_defaults(func=_cmd_cutlass)

    rnd = sub.add_parser("render",
                         help="apply the recipe then launch+track a render")
    rnd.add_argument("app", choices=["houdini", "comfyui"],
                     help="render engine: houdini (husk) or comfyui (comfy CLI)")
    rnd.add_argument("file", help="scene/USD (.hip/.usd) or workflow JSON")
    rnd.add_argument("--out", default=None, help="output path/dir override")
    rnd.add_argument("--frames", default=None,
                     help="frame range for Houdini, e.g. 1-100 or 1-100x2")
    rnd.add_argument("--dry-run", action="store_true",
                     help="show the recipe + command without launching")
    rnd.add_argument("--xargs", default="",
                     help='extra args passed verbatim to husk/comfy, quoted: '
                          '--xargs "--samples 512"')
    rnd.set_defaults(func=_cmd_render)
    return p


def _setup_logging(level: str, config: Optional[Config] = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        file_path = paths.log_path("optimizer")
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0

    _setup_logging(args.log_level)
    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 — config must never crash the CLI
        log.error("config load failed, using defaults: %s", exc)
        from v9.config import _defaults  # type: ignore
        config = Config(_defaults(), detect())

    return int(args.func(args, config))


if __name__ == "__main__":
    raise SystemExit(main())
