"""Combined pipeline CLI.

Runs the complete analysis: the single-session vision pipeline (formerly
``PART2.py``), the validation battery, the model-based population analyses
(Poisson GLM encoding, reduced-rank regression, latent dynamics) and the
multi-session / multi-subject aggregation.

Examples
--------
Single session, everything::

    uv run python combined_pipeline.py

Chosen stages only::

    uv run python combined_pipeline.py --stages single,validate

Multi-session run over the first six sessions, four processes, no figures::

    uv run python combined_pipeline.py --stages multisession --limit 6 --jobs 4 \\
        --no-figures

Multi-session run for two subjects, ignoring the cache::

    uv run python combined_pipeline.py --stages multisession \\
        --subjects Cori,Forssmann --force

Parameter override (any :class:`~vision_pipeline.AnalysisConfig` field)::

    uv run python combined_pipeline.py --stages single --set fdr_alpha=0.01 \\
        --set stim_window=0.06,0.30

Every stage writes a ``manifest.json`` recording the configuration, library
versions and git revision, so results can be traced back to the exact code and
parameters that produced them.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

from vision_pipeline import AnalysisConfig, load_session_alf
from vision_pipeline.config import (DEFAULT_DATA_ROOT, DEFAULT_SESSION,
                                    DEFAULT_OUTPUT_DIR, MODEL_OUTPUT_DIR,
                                    MULTISESSION_OUTPUT_DIR,
                                    VALIDATION_OUTPUT_DIR)
from vision_pipeline.pipeline import (run_model_stage, run_multisession,
                                      run_part2, run_validation_stage)

ALL_STAGES = ("single", "validate", "model", "multisession")
DEFAULT_STAGES = "single,validate,model"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="combined_pipeline.py",
        description="Full vision-population analysis pipeline "
                    "(Steinmetz et al. 2019).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stages", default=DEFAULT_STAGES,
                        help=f"comma-separated stages from {ALL_STAGES} "
                             f"(default: {DEFAULT_STAGES})")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION,
                        help="path to a single session folder (with an 'alf' "
                             "sub-folder)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Subjects root used by the multisession stage")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="root directory for the per-stage output folders "
                             "(default: repository root)")
    parser.add_argument("--limit", type=int, default=None,
                        help="analyse only the first N sessions (multisession)")
    parser.add_argument("--subjects", default=None,
                        help="comma-separated subject names (multisession)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel worker processes (multisession)")
    parser.add_argument("--force", action="store_true",
                        help="ignore the multisession cache and recompute")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip figure generation")
    parser.add_argument("--n-perm-reg", type=int, default=0,
                        help="permutations for the per-neuron regression null "
                             "(0 disables)")
    parser.add_argument("--n-perm-validation", type=int, default=50,
                        help="permutations for the validation null checks")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override an AnalysisConfig field, e.g. "
                             "--set fdr_alpha=0.01 (repeatable)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def apply_overrides(cfg: AnalysisConfig, overrides) -> AnalysisConfig:
    """Apply ``--set key=value`` overrides, parsing values with ``ast.literal_eval``."""
    values = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        try:
            values[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            values[key] = raw            # plain string
    return cfg.with_overrides(**values)


def main(argv=None) -> int:
    args = parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; choose from {ALL_STAGES}")

    verbose = not args.quiet
    cfg = apply_overrides(AnalysisConfig(), args.overrides)

    root = args.out_root or DEFAULT_OUTPUT_DIR.parent
    out_dirs = {
        "single": root / DEFAULT_OUTPUT_DIR.name,
        "multisession": root / MULTISESSION_OUTPUT_DIR.name,
        "model": root / MODEL_OUTPUT_DIR.name,
        "validate": root / VALIDATION_OUTPUT_DIR.name,
    }
    if args.out_root:
        out_dirs = {k: root / v.name for k, v in out_dirs.items()}

    print("=" * 78)
    print("COMBINED VISION PIPELINE")
    print("=" * 78)
    print(f"  stages    : {', '.join(stages)}")
    print(f"  outputs   : {root}")
    if verbose:
        print(f"  config    : seed={cfg.random_seed}, stim_window={cfg.stim_window}, "
              f"fdr_alpha={cfg.fdr_alpha}")

    t_start = time.perf_counter()
    part2 = None

    # ---- single-session stage (also feeds validate/model) ----
    if "single" in stages or "validate" in stages or "model" in stages:
        if not args.session or not Path(args.session).exists():
            raise SystemExit(f"session folder not found: {args.session}")
        session = load_session_alf(args.session, verbose=verbose)
        part2 = run_part2(session, out_dirs["single"], cfg,
                          make_figures=not args.no_figures,
                          n_perm_reg=args.n_perm_reg, verbose=verbose)

    if "validate" in stages:
        print("\n" + "#" * 78)
        print("# STAGE 7: PIPELINE VALIDATION")
        print("#" * 78)
        run_validation_stage(part2, out_dirs["validate"], cfg,
                             n_perm_null=args.n_perm_validation, verbose=verbose)

    if "model" in stages:
        run_model_stage(part2, out_dirs["model"], cfg,
                        make_figures=not args.no_figures, verbose=verbose)

    if "multisession" in stages:
        print("\n" + "#" * 78)
        print("# STAGE 8: MULTI-SESSION / MULTI-SUBJECT ANALYSIS")
        print("#" * 78)
        subjects = ([s.strip() for s in args.subjects.split(",") if s.strip()]
                    if args.subjects else None)
        run_multisession(args.data_root, out_dirs["multisession"], cfg,
                         limit=args.limit, subjects=subjects, jobs=args.jobs,
                         force=args.force, verbose=verbose)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"DONE in {elapsed / 60:.1f} min")
    for stage in stages:
        print(f"  {stage:12s} -> {out_dirs[stage]}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())