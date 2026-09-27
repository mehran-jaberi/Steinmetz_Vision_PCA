"""Temporary helper: inventory ALF sessions and their visual-related files."""
from pathlib import Path

ROOT = Path("Steinmetz_et_al_2019_9974357/nicklab/Subjects")
KEYS = [
    "spikes.times.npy", "spikes.clusters.npy", "clusters.depths.npy",
    "clusters.peakChannel.npy", "clusters._phy_annotation.npy",
    "channels.brainLocation.tsv", "probes.rawFilename.tsv",
    "trials.visualStim_times.npy", "trials.visualStim_contrastLeft.npy",
    "trials.goCue_times.npy", "trials.response_choice.npy",
    "sparseNoise.times.npy", "sparseNoise.positions.npy",
    "passiveVisual.times.npy",
]

sessions = []
for subj in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for date in sorted(p for p in subj.iterdir() if p.is_dir()):
        for num in sorted(p for p in date.iterdir() if p.is_dir()):
            alf = num / "alf"
            if not alf.is_dir():
                continue
            present = {k: (alf / k).exists() for k in KEYS}
            sessions.append((subj.name, f"{date.name}/{num.name}", present))

print(f"total sessions with alf: {len(sessions)}\n")
full = 0
for subj, date, present in sessions:
    missing = [k for k, v in present.items() if not v]
    has_vis = present["sparseNoise.times.npy"] and present["trials.visualStim_times.npy"]
    tag = "VIS" if has_vis else "   "
    if not missing:
        full += 1
    print(f"{tag} {subj:12s} {date}  missing={len(missing)}")
    if missing and has_vis:
        print(f"        -> {missing[:6]}")

print(f"\nsessions with all keys: {full}/{len(sessions)}")
