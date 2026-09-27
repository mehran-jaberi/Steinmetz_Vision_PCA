"""PART2.py -- Part 2: Vision-specific population analysis (single session).

This file is now a thin, backwards-compatible entry point: the analysis itself
lives in the :mod:`vision_pipeline` package (see ``vision_pipeline/__init__.py``
for the module map). Running::

    uv run python PART2.py

reproduces the original outputs in ``part2_outputs/``. For the full pipeline
(multi-session aggregation, model-based analyses, validation) use
``combined_pipeline.py`` instead.

The session path is configured below; change it to analyse a different
recording.
"""

from __future__ import annotations

from pathlib import Path

from vision_pipeline import AnalysisConfig, load_session_alf, run_part2

WORKSPACE = Path(__file__).resolve().parent

#: Session to analyse (see README > Dataset for how to obtain the data).
SESSION = WORKSPACE / "Steinmetz_et_al_2019_9974357" / "nicklab" / "Subjects" \
    / "Cori" / "2016-12-14" / "001"

#: Output directory for this stage.
OUTPUT_DIR = WORKSPACE / "part2_outputs"


def main(session_path=SESSION, output_dir=OUTPUT_DIR, config=None):
    print("=" * 78)
    print("PART 2 - VISION-SPECIFIC POPULATION ANALYSIS")
    print("=" * 78)
    cfg = config or AnalysisConfig()
    session = load_session_alf(session_path)
    return run_part2(session, output_dir, cfg)


if __name__ == "__main__":
    main()