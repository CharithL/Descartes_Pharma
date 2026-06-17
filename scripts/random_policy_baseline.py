"""A4: Random-action baseline for the Docking Game Agent.

Proves the trained policy learned a real search strategy rather than just
benefiting from random hill-climbing. Runs an UNTRAINED policy that takes
uniformly random actions through the *identical* DockingEnv, and reports the
same metrics used for the trained policy (mean initial Vina, mean final Vina,
fraction improved). Optionally compares against a Vina-dock-only reference.

Can be used two ways:
  * Imported by the pipeline (run_docking_game Phase 3 eval) to print a
    Trained vs Random-action comparison table.
  * Run standalone:  python -m scripts.random_policy_baseline
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def random_action_rollout(env, ligands, max_steps=30, n_actions=22,
                          seed=0, cap=50):
    """Run uniformly-random actions through the env for each ligand.

    Returns dict with mean_init, mean_final, frac_improved, and the per-ligand
    initial/final Vina scores.
    """
    rng = np.random.default_rng(seed)
    inits, finals = [], []

    for lig in ligands[:cap]:
        env.reset(lig)
        init_score = env.current_score
        for _ in range(max_steps):
            action = int(rng.integers(n_actions))
            _, _, done, _ = env.step(action)
            if done:
                break
        inits.append(init_score)
        finals.append(env.best_score)

    inits, finals = np.array(inits), np.array(finals)
    return {
        "mean_init": float(np.mean(inits)) if len(inits) else 0.0,
        "mean_final": float(np.mean(finals)) if len(finals) else 0.0,
        "frac_improved": float(np.mean(finals < inits)) if len(inits) else 0.0,
        "inits": inits,
        "finals": finals,
    }


def vina_dock_only(vina_world_model, env, ligands, cap=50, exhaustiveness=8):
    """Reference: let Vina's own search dock each ligand (no agent).

    Returns mean best docked energy. Skips ligands that fail to dock.
    """
    energies = []
    for lig in ligands[:cap]:
        try:
            coords = getattr(lig, "conformer_coords", None)
            if coords is None:
                continue
            pdbqt = env._coords_to_pdbqt(coords)
            results = vina_world_model.dock_ligand(
                pdbqt, n_poses=1, exhaustiveness=exhaustiveness
            )
            if results:
                energies.append(results[0].total_energy)
        except Exception:
            continue
    return float(np.mean(energies)) if energies else float("nan")


def print_policy_comparison(trained, random_action, vina_dock=None):
    """Print a Trained vs Random-action (vs Vina-dock-only) table + warning.

    `trained` and `random_action` are dicts with mean_init/mean_final/
    frac_improved. `vina_dock` is an optional mean best docked energy (float).
    """
    print()
    print("    " + "-" * 64)
    print("    POLICY COMPARISON (lower Vina = better; more negative = tighter)")
    print("    " + "-" * 64)
    print(f"    {'Method':<22}{'mean_init':>11}{'mean_final':>12}{'improved':>11}")
    for name, s in [("Trained policy", trained),
                    ("Random actions", random_action)]:
        print(f"    {name:<22}{s['mean_init']:>11.3f}"
              f"{s['mean_final']:>12.3f}{s['frac_improved']:>10.1%}")
    if vina_dock is not None and not np.isnan(vina_dock):
        print(f"    {'Vina-dock-only':<22}{'-':>11}{vina_dock:>12.3f}{'-':>11}")

    # Flag loudly if the trained policy is not clearly better than random.
    margin = random_action["mean_final"] - trained["mean_final"]
    if margin <= 0.5:  # trained not meaningfully lower (better) than random
        print()
        print("    WARNING: trained policy not clearly better than random "
              "actions")
        print(f"             (final Vina: trained={trained['mean_final']:.3f} "
              f"vs random={random_action['mean_final']:.3f}, "
              f"margin={margin:+.3f})")
    print("    " + "-" * 64)


def main():
    """Standalone: build the env via the pipeline and run the random baseline."""
    from scripts.run_docking_game import run_phase1, run_phase2
    from descartes_pharma_docking.training.docking_env import DockingEnv

    pocket, vina_model, _, _ = run_phase1()
    _, val_ligands, _, _, _, _ = run_phase2()

    env = DockingEnv(vina_world_model=vina_model, pocket_features=pocket,
                     max_steps=30, score_history_len=10)

    print("\n[A4] Random-action baseline (standalone)")
    rand_stats = random_action_rollout(env, val_ligands, max_steps=30)
    vdock = vina_dock_only(vina_model, env, val_ligands)
    # No trained policy in standalone mode -- show random vs Vina-dock-only.
    print_policy_comparison(rand_stats, rand_stats, vina_dock=vdock)


if __name__ == "__main__":
    main()
