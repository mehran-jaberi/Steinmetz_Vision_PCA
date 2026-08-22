import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_session(folder_path="."):
    """
    Load a single recording session from .npy files (and optional .mat trial info).

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder containing spike_times.npy, spike_clusters.npy,
        templates.npy, channel_positions.npy, channel_map.npy,
        and trial_info.npy (or trial_info.mat).

    Returns
    -------
    data : dict
        Dictionary with the following keys:
        - spike_times        : ndarray (n_spikes,)        – spike times (seconds)
        - spike_clusters     : ndarray (n_spikes,)        – unit ID for each spike
        - templates          : ndarray (n_units, n_timepoints, n_channels)
        - channel_positions  : ndarray (n_channels, 2)    – (x, y) in µm
        - channel_map        : ndarray (n_channels,)      – which channels were saved
        - trial_info         : structured ndarray         – trial metadata (see paper)
    """
    folder = Path(folder_path)

    data = {}
    data["spike_times"]        = np.load(folder / "spike_times.npy")
    data["spike_clusters"]     = np.load(folder / "spike_clusters.npy")
    data["templates"]          = np.load(folder / "templates.npy")
    data["channel_positions"]  = np.load(folder / "channel_positions.npy")
    data["channel_map"]        = np.load(folder / "channel_map.npy")

    # Trial information – try .npy first, then .mat
    trial_npy = folder / "trial_info.npy"
    trial_mat = folder / "trial_info.mat"
    if trial_npy.exists():
        data["trial_info"] = np.load(trial_npy, allow_pickle=True)
    elif trial_mat.exists():
        mat = loadmat(trial_mat, struct_as_record=False, squeeze_me=True)
        data["trial_info"] = mat["trial_info"]
    else:
        raise FileNotFoundError("No trial_info.npy or trial_info.mat found.")

    # Optional: some sessions provide a cluster info file
    clusters_file = folder / "clusters.npy"
    if clusters_file.exists():
        data["clusters"] = np.load(clusters_file, allow_pickle=True)

    return data


# ============================================================
# 2. SPIKE TIME RETRIEVAL
# ============================================================
def get_unit_spike_times(data, unit_id, time_range=None):
    """
    Extract spike times for a single unit, optionally within a time window.

    Parameters
    ----------
    data : dict
        Session dictionary (output of load_session).
    unit_id : int
        Cluster ID of the unit.
    time_range : tuple (start, end), optional
        Restrict to times in this interval (seconds).

    Returns
    -------
    times : ndarray
        Spike times of the selected unit.
    """
    mask = data["spike_clusters"] == unit_id
    times = data["spike_times"][mask]
    if time_range is not None:
        times = times[(times >= time_range[0]) & (times <= time_range[1])]
    return times


# ============================================================
# 3. PERI-STIMULUS TIME HISTOGRAM (PSTH)
# ============================================================
def compute_psth(data, unit_id, align_event="go_cue", trial_mask=None,
                 bin_width=0.01, time_window=(-0.5, 1.0)):
    """
    Compute the average PSTH of one unit aligned to a trial event.

    Parameters
    ----------
    data : dict
    unit_id : int
    align_event : str
        Field name in trial_info used for alignment (e.g., 'go_cue', 'stim_on').
    trial_mask : boolean ndarray, optional
        Subset of trials to include.
    bin_width : float
        Bin size in seconds.
    time_window : tuple (start, end)
        Time relative to the event.

    Returns
    -------
    t_centers : ndarray
        Time bin centers.
    psth : ndarray
        Mean firing rate (spikes/s) across the selected trials.
    """
    trials = data["trial_info"]
    if trial_mask is not None:
        trials = trials[trial_mask]

    align_times = trials[align_event]                # one per trial
    spike_times = get_unit_spike_times(data, unit_id)

    bins = np.arange(time_window[0], time_window[1] + bin_width, bin_width)
    counts = np.zeros((len(trials), len(bins) - 1))

    for i, t0 in enumerate(align_times):
        aligned = spike_times - t0
        in_window = (aligned >= time_window[0]) & (aligned < time_window[1])
        counts[i, :], _ = np.histogram(aligned[in_window], bins=bins)

    psth = counts.mean(axis=0) / bin_width           # spikes per second
    t_centers = bins[:-1] + bin_width / 2
    return t_centers, psth


# ============================================================
# 4. POPULATION RESPONSE MATRIX
# ============================================================
def build_population_response(data, unit_ids, time_range, align_event="go_cue"):
    """
    Create a [trials x units] matrix of spike counts in a given time window.

    Parameters
    ----------
    data : dict
    unit_ids : list or ndarray of ints
    time_range : tuple (start, end) relative to align_event
    align_event : str

    Returns
    -------
    response : ndarray (n_trials, n_units)
        Spike counts per trial.
    """
    trials = data["trial_info"]
    align_times = trials[align_event]
    n_trials = len(trials)
    n_units = len(unit_ids)
    response = np.zeros((n_trials, n_units))

    for i, t0 in enumerate(align_times):
        for j, uid in enumerate(unit_ids):
            sp = get_unit_spike_times(data, uid)
            aligned = sp - t0
            response[i, j] = np.sum((aligned >= time_range[0]) & (aligned < time_range[1]))
    return response


# ============================================================
# 5. DECODING BEHAVIOR (EXAMPLE: CHOICE)
# ============================================================
def decode_choice(data, unit_ids, time_range=(0.05, 0.35), cv_folds=5):
    """
    Train a logistic regression classifier to decode the animal's choice (left/right)
    from population activity in a given time window.

    Parameters
    ----------
    data : dict
    unit_ids : array-like of int
    time_range : tuple
    cv_folds : int
        Number of cross-validation folds.

    Returns
    -------
    mean_score : float
    std_score : float
    """
    trials = data["trial_info"]

    # Define trial types: high‑contrast left vs high‑contrast right (no bilateral)
    left_trials = (trials["contrast_left"] > 0.5) & (trials["contrast_right"] == 0)
    right_trials = (trials["contrast_right"] > 0.5) & (trials["contrast_left"] == 0)
    trial_mask = left_trials | right_trials

    X = build_population_response(data, unit_ids, time_range, align_event="go_cue")
    X = X[trial_mask]
    y = np.where(right_trials[trial_mask], 1, 0)   # 0 = left, 1 = right

    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores.mean(), scores.std()


# ============================================================
# 6. VISUALIZATION HELPERS
# ============================================================
def plot_channel_map(data):
    """Quick scatter plot of recording channel positions."""
    pos = data["channel_positions"]
    plt.figure(figsize=(4, 8))
    plt.scatter(pos[:, 0], pos[:, 1], s=2)
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Channel positions")
    plt.axis("equal")
    plt.show()


def plot_template(data, template_id):
    """
    Show the mean spike waveform of a template on a few channels
    near its peak channel.
    """
    template = data["templates"][template_id]   # (time, channels)
    peak_chan = np.argmax(np.abs(template).max(axis=0))
    ch_start = max(0, peak_chan - 4)
    ch_end = min(template.shape[1], peak_chan + 5)
    channels_to_plot = range(ch_start, ch_end)

    plt.figure()
    for ch in channels_to_plot:
        plt.plot(template[:, ch], label=f"ch {ch}")
    plt.legend()
    plt.title(f"Template {template_id}")
    plt.xlabel("Time samples")
    plt.ylabel("Amplitude (a.u.)")
    plt.show()