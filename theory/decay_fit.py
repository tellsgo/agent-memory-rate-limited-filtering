#!/usr/bin/env python3
"""Fits of the reader decay models in theory.tex Sec. 5 to the batch_01 P0 AUCs (equal-variance
binormal link AUC = Phi(d'/sqrt 2)). Leave-last-out: fit on T = 16, 64, predict T = 256."""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

T = np.array([16., 64., 256.])
AUC = {"7B": np.array([0.774, 0.636, 0.488]), "14B": np.array([0.795, 0.701, 0.549])}
N_POS = np.array([50, 150, 50])            # APPROVE episodes per T (= DENY episodes)

def hanley_mcneil(a, n1, n2):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))

dprime = lambda a: np.sqrt(2) * norm.ppf(a)
auc = lambda d: norm.cdf(d / np.sqrt(2))

MODELS = {   # name: (d'(T; theta), initial theta)
    "interference  d0/sqrt(1+T/Tc)": (lambda th, t: th[0] / np.sqrt(1 + t / th[1]), [1.5, 20.]),
    "contraction   d0*exp(-T/tau)":  (lambda th, t: th[0] * np.exp(-t / th[1]), [1.5, 60.]),
    "power         a*T^-alpha":      (lambda th, t: th[0] * t ** (-th[1]), [3., 0.5]),
}

for name_r, a in AUC.items():
    se = hanley_mcneil(a, N_POS, N_POS)
    print(f"\n== {name_r}: AUC {a}, SE {np.round(se, 3)}, d' {np.round(dprime(a), 3)}")
    for mname, (f, th0) in MODELS.items():
        # leave-last-out: exact fit through the first two points (in AUC space)
        r2 = least_squares(lambda th: auc(f(th, T[:2])) - a[:2], th0, bounds=([0, 1e-3], [20, 1e4]))
        pred = auc(f(r2.x, T[2]))
        # all three points, weighted by SE
        r3 = least_squares(lambda th: (auc(f(th, T)) - a) / se, th0, bounds=([0, 1e-3], [20, 1e4]))
        chi2 = float(np.sum(r3.fun ** 2))
        print(f"  {mname}: LOO theta={np.round(r2.x, 3)} pred AUC(256)={pred:.3f} "
              f"err={pred - a[2]:+.3f} ({(pred - a[2]) / se[2]:+.2f} SE) | full theta={np.round(r3.x, 3)} chi2={chi2:.2f}")


# ---- Model B (binding-error mixture): BA* - 1/2 = (1 - w_T) (BA*_c - 1/2), 1 - w_T = exp(-T/tau);
# DENY rate q_T = q_z + (q_c - q_z) exp(-T/tau) with the same tau.
BA = {"7B": np.array([0.76, 0.64, 0.54]), "14B": np.array([0.77, 0.67, 0.57])}
QDENY = {"7B": np.array([(0.80 + 0.52) / 2, (0.77 + 0.67) / 2, (0.78 + 0.86) / 2]),
         "14B": np.array([0.46, 0.65, 0.79])}
N_EP = 2 * N_POS
rng = np.random.default_rng(0)
print("\n== Model B")
for name_r in AUC:
    a, b, q = AUC[name_r], BA[name_r], QDENY[name_r]
    se = hanley_mcneil(a, N_POS, N_POS)
    # tau from AUC - 1/2 via the first two points (upper envelope: single exponential)
    tau_auc = (T[1] - T[0]) / np.log((a[0] - .5) / (a[1] - .5))
    pred_auc = .5 + (a[1] - .5) * np.exp(-(T[2] - T[1]) / tau_auc)
    tau_ba = (T[1] - T[0]) / np.log((b[0] - .5) / (b[1] - .5))
    pred_ba = .5 + (b[1] - .5) * np.exp(-(T[2] - T[1]) / tau_ba)
    # parametric bootstrap CI for tau_auc
    taus = []
    for _ in range(20000):
        x = a[:2] + rng.normal(0, se[:2])
        if x[1] > .5 and x[0] > x[1]:
            taus.append((T[1] - T[0]) / np.log((x[0] - .5) / (x[1] - .5)))
    lo, hi = np.percentile(taus, [2.5, 97.5])
    # shared-tau prediction of the DENY rate at T = 64 from T = 16 and 256
    e = np.exp(-T / tau_auc)
    dlt = (q[0] - q[2]) / (e[0] - e[2])
    qz = q[2] - dlt * e[2]
    pred_q = qz + dlt * e[1]
    se_q = np.sqrt(q[1] * (1 - q[1]) / N_EP[1])
    print(f"  {name_r}: tau_AUC={tau_auc:.1f} (95% param. bootstrap {lo:.0f}-{hi:.0f}, valid draws {len(taus)}) "
          f"pred AUC(256)={pred_auc:.3f} obs {a[2]} | tau_BA*={tau_ba:.1f} pred BA*(256)={pred_ba:.3f} obs {b[2]} | "
          f"DENY rate T=64 pred {pred_q:.3f} obs {q[1]:.3f} (SE {se_q:.3f}), q_c={dlt + qz:.2f} q_z={qz:.2f}")
