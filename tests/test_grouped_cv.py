"""Smoke test for Group A statistical fixes (no RDKit/Vina required).

Demonstrates that GroupKFold removes trajectory leakage, and that the
scaffold-stratified permutation + effective-n helpers run correctly on
synthetic trajectory-structured data.

Run: python -m tests.test_grouped_cv
"""
import numpy as np

from descartes_pharma_docking.probing.probe_runner import (
    cv_ridge_r2,
    permutation_test,
    one_per_trajectory_indices,
)
from descartes_pharma_docking.probing.scaffold_permutation import hardened_probe


def make_trajectory_data(n_ep=30, steps=20, dim=32, seed=0):
    """Each episode has a distinct hidden 'signature' and a random target.

    Within an episode, consecutive timesteps are near-identical (as in real
    docking trajectories). The per-episode target is independent random, so it
    is NOT predictable from a held-out episode's signature -- any apparent
    predictive power across folds is pure leakage.
    """
    rng = np.random.default_rng(seed)
    H, y, ep = [], [], []
    for e in range(n_ep):
        sig = rng.normal(size=dim)
        tgt = rng.normal()
        for _ in range(steps):
            H.append(sig + 0.01 * rng.normal(size=dim))
            y.append(tgt + 0.001 * rng.normal())
            ep.append(e)
    return (
        np.array(H, np.float32),
        np.array(y, np.float32),
        np.array(ep, np.int64),
    )


def main():
    H, y, ep = make_trajectory_data()

    r2_leak = cv_ridge_r2(H, y, groups=None)   # shuffled KFold -> leaks
    r2_clean = cv_ridge_r2(H, y, groups=ep)    # GroupKFold -> honest
    print(f"R2 shuffled-KFold (leaky):  {r2_leak:.3f}")
    print(f"R2 GroupKFold    (honest):  {r2_clean:.3f}")
    assert r2_leak > 0.5, "expected leakage to inflate shuffled-KFold R2"
    assert r2_clean < r2_leak - 0.3, "GroupKFold should drop sharply (leak removed)"

    idx = one_per_trajectory_indices(ep, seed=1)
    print(f"one-per-trajectory n:       {len(idx)} (expected 30)")
    assert len(idx) == 30 and len(np.unique(ep[idx])) == 30

    scaf = ep % 6  # fake scaffolds: 6 groups
    p = permutation_test(H, y, n_perms=50, scaffolds=scaf, groups=ep)
    print(f"scaffold-strat perm p:      {p:.3f}")
    assert 0.0 <= p <= 1.0

    H_un = np.random.default_rng(7).normal(0, 1, H.shape).astype(np.float32)
    d, pv, enc = hardened_probe(H, H_un, y, scaf, n_perms=50, groups=ep)
    print(f"hardened_probe:             dR2={d:.3f} p={pv:.3f} encoded={enc}")

    print("\nGROUP A SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
