"""Temporary: debug RRR R^2 and unit stability."""
import numpy as np

from vision_pipeline import AnalysisConfig, load_session_alf
from vision_pipeline.latent import reduced_rank_regression, rrr_score
from vision_pipeline.responses import bin_spike_counts, population_time_series
from vision_pipeline.stimuli import build_stimulus_movie, reconstruct_sparse_noise
from vision_pipeline.units import identify_visual_neurons

cfg = AnalysisConfig(rf_permutations=0, n_permutations=0)
s = load_session_alf("Steinmetz_et_al_2019_9974357/nicklab/Subjects/Cori/2016-12-14/001",
                     verbose=False)
u = identify_visual_neurons(s, cfg, verbose=False)
flashes, grid = reconstruct_sparse_noise(s)
edges, movie = build_stimulus_movie(flashes, grid, bin_size=cfg.glm_bin)
print("movie", movie.shape, "edges", edges.shape, "flags per bin:",
      movie.sum(1).min(), movie.sum(1).max())

Y = population_time_series(s, u.refined_ids, edges)
lag = 1
X = movie[: movie.shape[0] - lag].astype(float)
Yl = Y[lag:]
print("Y", Yl.shape, "X", X.shape, "Y mean", Yl.mean().round(4))

# direct OLS
A = np.column_stack([np.ones(len(X)), X])
B, *_ = np.linalg.lstsq(A, Yl, rcond=None)
pred = A @ B
sst = np.sum((Yl - Yl.mean(0)) ** 2)
print("direct OLS R2 (with intercept) =", round(1 - np.sum((Yl - pred) ** 2) / sst, 6))

# via reduced_rank_regression full rank
fit = reduced_rank_regression(Yl, X, min(X.shape[1], Yl.shape[1]))
print("RRR full-rank R2               =", round(rrr_score(Yl, X, fit), 6))
fit1 = reduced_rank_regression(Yl, X, 1)
print("RRR rank-1 R2                  =", round(rrr_score(Yl, X, fit1), 6))
fit12 = reduced_rank_regression(Yl, X, 12)
print("RRR rank-12 R2                 =", round(rrr_score(Yl, X, fit12), 6))

# sanity: synthetic data with a known rank-2 relationship
rng = np.random.default_rng(0)
n, p, q = 2000, 50, 20
Xs = rng.standard_normal((n, p))
W = rng.standard_normal((p, 2)) @ rng.standard_normal((2, q))
Ys = Xs @ W + 0.5 * rng.standard_normal((n, q))
print("\nsynthetic: OLS R2 =",
      round(rrr_score(Ys, Xs, reduced_rank_regression(Ys, Xs, min(p, q))), 4),
      "| rank-2 R2 =", round(rrr_score(Ys, Xs, reduced_rank_regression(Ys, Xs, 2)), 4),
      "| rank-1 R2 =", round(rrr_score(Ys, Xs, reduced_rank_regression(Ys, Xs, 1)), 4))

# --- unit stability ---
print("\nrate stability per unit (10 bins, first half vs second half):")
e = np.linspace(s.spike_times.min(), s.spike_times.max(), 11)
profiles = []
for uid in u.refined_ids:
    c = bin_spike_counts(s.unit_spike_times(int(uid)), e)
    profiles.append(c)
P = np.array(profiles)
print("population rate profile (Hz):", np.round(P.mean(0) / 106.7, 2))
cs = [np.corrcoef(p[:5], p[5:])[0, 1] for p in P]
print("median split-half r =", round(float(np.median(cs)), 3))
print("first-half vs second-half population means:",
      round(P[:, :5].mean(), 1), round(P[:, 5:].mean(), 1))
