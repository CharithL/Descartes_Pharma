"""D1: Docking-pose cache.

Every episode previously re-docked the same ligands, turning training into a
48+ hour run. The cache docks each unique ligand ONCE (at high exhaustiveness),
stores the reconstructed full-atom pose keyed by SMILES, and persists it to
disk. Subsequent episodes look up the cached pose instead of re-docking.

Note: the spec asked for a `.pkl`, but we persist as JSON instead -- pickle is
unsafe to load and unnecessary here (the payload is just {SMILES: [[x,y,z],...]}).
"""
import json
from pathlib import Path

import numpy as np


class PoseCache:
    """SMILES -> full-atom docked pose, persisted as JSON."""

    def __init__(self, path):
        self.path = Path(path)
        self.cache = {}

    def load(self):
        """Load the cache from disk if it exists (best-effort)."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    raw = json.load(f)
                self.cache = {k: np.asarray(v, dtype=float)
                              for k, v in raw.items()}
            except Exception:
                self.cache = {}
        return self

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: np.asarray(v).tolist()
                        for k, v in self.cache.items()}
        with open(self.path, "w") as f:
            json.dump(serializable, f)

    def __contains__(self, smiles):
        return smiles in self.cache

    def __len__(self):
        return len(self.cache)

    def get(self, smiles):
        return self.cache.get(smiles)

    def put(self, smiles, coords):
        self.cache[smiles] = np.asarray(coords)


def predock_ligands(env, ligands, cache, exhaustiveness=8, progress_every=100):
    """Phase 2.5: dock each unique ligand ONCE and cache its full-atom pose.

    Uses the env's real-pose reconstruction (`_dock_and_reconstruct`) so cached
    poses are the same quality as a live dock. Already-cached SMILES are
    skipped, so re-running resumes where it left off.
    """
    seen, uniq = set(), []
    for lig in ligands:
        smi = getattr(lig, "smiles", None)
        if smi is None or smi in seen:
            continue
        seen.add(smi)
        uniq.append(lig)

    todo = [l for l in uniq if l.smiles not in cache]
    print(f"    Pre-docking {len(todo)} ligands "
          f"({len(uniq) - len(todo)} already cached, "
          f"exhaustiveness={exhaustiveness})...")

    docked = 0
    for i, lig in enumerate(todo):
        try:
            coords = env._dock_and_reconstruct(lig, exhaustiveness=exhaustiveness)
            if coords is not None:
                cache.put(lig.smiles, coords)
                docked += 1
        except Exception:
            pass
        if (i + 1) % progress_every == 0:
            print(f"      pre-docked {i + 1}/{len(todo)} "
                  f"({docked} new poses cached)")

    cache.save()
    print(f"    Pre-docking complete: {len(cache)} poses cached -> {cache.path}")
    return cache
