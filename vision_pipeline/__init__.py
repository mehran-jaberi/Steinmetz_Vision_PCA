"""Vision population analysis pipeline for the Steinmetz et al. (2019) dataset.

Module map
----------
``config``           :class:`AnalysisConfig` — all tunable parameters
``io``               ALF session loading and session discovery
``stimuli``          grating / sparse-noise stimulus reconstruction
``units``            refined visual-neuron identification
``responses``        response matrices, PSTHs, binned tensors
``encoding``         contrast tuning, regression, decoding
``receptive_fields`` spike-triggered-average receptive fields
``population``       PCA and condition structure
``glm``              Poisson stimulus encoding models (model-based stage)
``latent``           reduced-rank regression and latent dynamics
``validation``       integrity, alignment and null-calibration checks
``figures``          all plots
``pipeline``         stage orchestration
``reporting``        run manifests (config, versions, git revision)

Typical use::

    from vision_pipeline import AnalysisConfig, load_session_alf, run_part2
    cfg = AnalysisConfig()
    session = load_session_alf(".../Cori/2016-12-14/001")
    result = run_part2(session, "part2_outputs", cfg)
"""

from __future__ import annotations

from .config import (AnalysisConfig, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR,
                     DEFAULT_SESSION, MODEL_OUTPUT_DIR, MULTISESSION_OUTPUT_DIR,
                     VALIDATION_OUTPUT_DIR, VISUAL_AREAS)
from .io import SessionData, SessionRef, discover_sessions, load_session_alf
from .pipeline import (ModelResult, Part2Result, run_model_stage,
                       run_multisession, run_part2, run_validation_stage,
                       session_metrics)

__all__ = [
    "AnalysisConfig",
    "VISUAL_AREAS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_SESSION",
    "DEFAULT_OUTPUT_DIR",
    "MULTISESSION_OUTPUT_DIR",
    "MODEL_OUTPUT_DIR",
    "VALIDATION_OUTPUT_DIR",
    "SessionData",
    "SessionRef",
    "load_session_alf",
    "discover_sessions",
    "Part2Result",
    "ModelResult",
    "run_part2",
    "run_model_stage",
    "run_validation_stage",
    "run_multisession",
    "session_metrics",
]

__version__ = "0.2.0"
