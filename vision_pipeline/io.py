"""ALF-format data loading and session discovery.

The loader in this module is a drop-in replacement for the original
``PART2.py::load_session_alf`` with one important efficiency change: spikes are
grouped by cluster **once**, using a single stable argsort, instead of masking
the full spike array for every unit. That makes it cheap to iterate over all
clusters of a session (needed by the multi-session stage) while producing
identical per-unit spike times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Trial fields loaded from ``trials.*.npy`` and used downstream.
TRIAL_FILES = {
    "goCue_times": "goCue_times",
    "visualStim_times": "visualStim_times",
    "response_times": "response_times",
    "feedback_times": "feedback_times",
    "visualStim_contrastLeft": "visualStim_contrastLeft",
    "visualStim_contrastRight": "visualStim_contrastRight",
    "response_choice": "response_choice",
    "feedbackType": "feedbackType",
    "repNum": "repNum",
    "included": "included",
}

#: Files that must exist for a session to be analysed.
REQUIRED_FILES = (
    "spikes.times.npy",
    "spikes.clusters.npy",
    "clusters.depths.npy",
    "clusters.peakChannel.npy",
    "clusters.probes.npy",
    "clusters._phy_annotation.npy",
    "channels.sitePositions.npy",
    "channels.probe.npy",
    "channels.brainLocation.tsv",
    "probes.rawFilename.tsv",
    "probes.insertion.tsv",
    "trials.visualStim_times.npy",
    "trials.visualStim_contrastLeft.npy",
    "trials.visualStim_contrastRight.npy",
    "trials.goCue_times.npy",
    "trials.response_choice.npy",
    "trials.included.npy",
    "trials.intervals.npy",
    "sparseNoise.times.npy",
    "sparseNoise.positions.npy",
)


@dataclass
class SessionRef:
    """Lightweight pointer to a session on disk."""

    subject: str
    date: str
    number: str
    path: Path

    @property
    def name(self) -> str:
        return f"{self.subject}/{self.date}/{self.number}"

    @property
    def label(self) -> str:
        return f"{self.subject} {self.date}"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


@dataclass
class SessionData:
    """In-memory representation of one ALF session.

    Attributes mirror the original ``PART2.py`` dict, plus a precomputed spike
    index that makes ``unit_spike_times`` O(1) look-up after a first touch.
    """

    session: str
    spike_times: np.ndarray
    spike_clusters: np.ndarray
    n_clusters: int
    cluster_depths: np.ndarray
    cluster_peak_channel: np.ndarray
    cluster_probes: np.ndarray
    cluster_waveform_duration: np.ndarray
    cluster_annotation: np.ndarray
    channel_positions: np.ndarray
    channel_probe: np.ndarray
    channel_region: np.ndarray
    channel_ccf: np.ndarray
    probe_names: list
    probe_insertion: pd.DataFrame
    trial_info: dict
    n_trials: int
    passiveVisual: dict
    sparseNoise: dict
    # --- derived, not part of the raw data ---
    _spike_order: np.ndarray = field(repr=False, default=None)
    _spike_offsets: np.ndarray = field(repr=False, default=None)
    _unit_spikes: dict = field(repr=False, default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def n_spikes_per_cluster(self) -> np.ndarray:
        """Total spike count per cluster."""
        if not hasattr(self, "_n_spikes"):
            self._n_spikes = np.diff(self._spike_offsets)
        return self._n_spikes

    def unit_spike_times(self, unit_id: int) -> np.ndarray:
        """Sorted spike times of one unit (cached)."""
        unit_id = int(unit_id)
        cached = self._unit_spikes.get(unit_id)
        if cached is None:
            if unit_id < 0 or unit_id >= self.n_clusters:
                raise IndexError(f"cluster id {unit_id} out of range")
            a, b = self._spike_offsets[unit_id], self._spike_offsets[unit_id + 1]
            cached = self.spike_times[self._spike_order[a:b]]
            self._unit_spikes[unit_id] = cached
        return cached

    def all_unit_spike_times(self, unit_ids) -> dict:
        """Bulk cache access for a set of units."""
        return {int(u): self.unit_spike_times(int(u)) for u in unit_ids}

    @property
    def duration(self) -> float:
        """Recording duration in seconds (trial interval span)."""
        try:
            intervals = self.trial_info["intervals"]
            return float(np.max(intervals[:, 1]) - np.min(intervals[:, 0]))
        except Exception:
            return float(self.spike_times[-1] - self.spike_times[0])

    @property
    def mean_firing_rate(self) -> float:
        dur = self.duration
        if dur <= 0:
            return float("nan")
        return float(len(self.spike_times) / dur / max(self.n_clusters, 1))

    def region_of_unit(self, unit_id: int) -> str:
        """Brain-region label assigned via the unit's peak channel."""
        return str(self.channel_region[int(self.cluster_peak_channel[unit_id])])

    def region_of_units(self, unit_ids) -> np.ndarray:
        unit_ids = np.asarray(unit_ids, dtype=int)
        return self.channel_region[self.cluster_peak_channel[unit_ids]]


def _build_spike_index(cluster_ids: np.ndarray, n_clusters: int):
    """Single stable argsort grouping spikes by cluster, preserving time order."""
    order = np.argsort(cluster_ids, kind="stable")
    counts = np.bincount(cluster_ids, minlength=n_clusters)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return order.astype(np.int64), offsets


def load_session_alf(session_folder, verbose=True) -> SessionData:
    """Load an ALF-format session.

    Parameters
    ----------
    session_folder : str or Path
        Session directory that contains an ``alf`` sub-folder.
    verbose : bool
        Print a short summary of what was loaded.

    Raises
    ------
    FileNotFoundError
        If the folder or a required file is missing, with the missing files
        listed explicitly (this is what the validation stage reports).
    """
    folder = Path(session_folder) / "alf"
    if not folder.exists():
        raise FileNotFoundError(f"No 'alf' folder found at {folder}")

    missing = [f for f in REQUIRED_FILES if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Session {folder} is missing required files: {', '.join(missing)}"
        )

    spike_times = np.load(folder / "spikes.times.npy").ravel().astype(float)
    spike_clusters = np.load(folder / "spikes.clusters.npy").ravel().astype(np.int64)
    n_clusters = len(np.load(folder / "clusters.depths.npy"))
    order, offsets = _build_spike_index(spike_clusters, n_clusters)

    trial_info = {}
    for key, fname in TRIAL_FILES.items():
        trial_info[key] = np.load(folder / f"trials.{fname}.npy").ravel()
    trial_info["intervals"] = np.load(folder / "trials.intervals.npy")

    session = SessionData(
        session=str(Path(session_folder)),
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        n_clusters=n_clusters,
        cluster_depths=np.load(folder / "clusters.depths.npy").ravel(),
        cluster_peak_channel=np.load(folder / "clusters.peakChannel.npy")
        .ravel()
        .astype(int),
        cluster_probes=np.load(folder / "clusters.probes.npy").ravel().astype(int),
        cluster_waveform_duration=np.load(folder / "clusters.waveformDuration.npy")
        .ravel(),
        cluster_annotation=np.load(
            folder / "clusters._phy_annotation.npy", allow_pickle=True
        )
        .ravel()
        .astype(int),
        channel_positions=np.load(folder / "channels.sitePositions.npy"),
        channel_probe=np.load(folder / "channels.probe.npy").ravel().astype(int),
        channel_region=pd.read_csv(folder / "channels.brainLocation.tsv", sep="\t")[
            "allen_ontology"
        ]
        .astype(str)
        .values,
        channel_ccf=pd.read_csv(folder / "channels.brainLocation.tsv", sep="\t")[
            ["ccf_ap", "ccf_dv", "ccf_lr"]
        ].values,
        probe_names=[
            str(n)
            for n in pd.read_csv(folder / "probes.rawFilename.tsv", sep="\t")[
                "rawFilename"
            ].values
        ],
        probe_insertion=pd.read_csv(folder / "probes.insertion.tsv", sep="\t"),
        trial_info=trial_info,
        n_trials=len(trial_info["goCue_times"]),
        passiveVisual={
            "times": np.load(folder / "passiveVisual.times.npy").ravel()
            if (folder / "passiveVisual.times.npy").exists()
            else np.array([]),
            "contrastLeft": np.load(folder / "passiveVisual.contrastLeft.npy").ravel()
            if (folder / "passiveVisual.contrastLeft.npy").exists()
            else np.array([]),
            "contrastRight": np.load(folder / "passiveVisual.contrastRight.npy").ravel()
            if (folder / "passiveVisual.contrastRight.npy").exists()
            else np.array([]),
        },
        sparseNoise={
            "positions": np.load(folder / "sparseNoise.positions.npy"),
            "times": np.load(folder / "sparseNoise.times.npy").ravel(),
        },
        _spike_order=order,
        _spike_offsets=offsets,
    )

    # sanity checks (fail fast on corrupt/incompatible sessions)
    if spike_clusters.max(initial=0) >= n_clusters:
        raise ValueError("spike cluster id exceeds number of clusters")
    if session.cluster_peak_channel.max(initial=0) >= len(session.channel_region):
        raise ValueError("peak channel id exceeds number of channels")

    if verbose:
        print(f"Loaded session: {session.session}")
        print(
            f"  {session.n_trials} trials | {session.n_clusters} clusters | "
            f"{len(session.spike_times):,} spikes"
        )
    return session


def discover_sessions(data_root, require=REQUIRED_FILES) -> list[SessionRef]:
    """Find every ``<subject>/<date>/<number>`` session under ``data_root``.

    Only sessions that contain all ``require`` files are returned, so the
    multi-session stage never fails half-way through a batch.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    refs = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for date_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            for num_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
                alf = num_dir / "alf"
                if not alf.is_dir():
                    continue
                if all((alf / f).exists() for f in require):
                    refs.append(
                        SessionRef(subject_dir.name, date_dir.name, num_dir.name,
                                   num_dir)
                    )
    return refs