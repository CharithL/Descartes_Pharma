"""
DESCARTES-PHARMA: C2 Drug Candidate Factory.

Evolutionary factory for generating, mutating, and selecting drug candidate
model architectures via Thompson sampling and genetic operators. Each genome
encodes a complete model specification -- architecture, representation,
training hyperparameters, and mechanistic-awareness flags -- enabling
automated neural architecture search across the drug discovery design space.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import copy
import uuid

import numpy as np


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_ARCHITECTURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gcn": {
        "family": "graph",
        "representation": "graph",
        "supports_3d": False,
        "default_layers": 3,
        "default_embedding_dim": 128,
        "description": "Graph Convolutional Network (Kipf & Welling).",
    },
    "gat": {
        "family": "graph",
        "representation": "graph",
        "supports_3d": False,
        "default_layers": 3,
        "default_embedding_dim": 128,
        "description": "Graph Attention Network with multi-head attention.",
    },
    "mpnn": {
        "family": "graph",
        "representation": "graph",
        "supports_3d": False,
        "default_layers": 4,
        "default_embedding_dim": 256,
        "description": "Message Passing Neural Network (Gilmer et al.).",
    },
    "schnet": {
        "family": "geometric",
        "representation": "3d_conformer",
        "supports_3d": True,
        "default_layers": 6,
        "default_embedding_dim": 256,
        "description": "SchNet continuous-filter convolutional network for 3D molecular geometry.",
    },
    "dimenet": {
        "family": "geometric",
        "representation": "3d_conformer",
        "supports_3d": True,
        "default_layers": 6,
        "default_embedding_dim": 256,
        "description": "DimeNet with directional message passing on 3D structure.",
    },
    "transformer": {
        "family": "sequence",
        "representation": "smiles",
        "supports_3d": False,
        "default_layers": 6,
        "default_embedding_dim": 512,
        "description": "Transformer encoder over SMILES token sequences.",
    },
    "fingerprint_mlp": {
        "family": "classical",
        "representation": "fingerprint",
        "supports_3d": False,
        "default_layers": 3,
        "default_embedding_dim": 512,
        "description": "MLP baseline operating on molecular fingerprint vectors.",
    },
    "neural_ode": {
        "family": "continuous",
        "representation": "graph",
        "supports_3d": False,
        "default_layers": 4,
        "default_embedding_dim": 128,
        "description": "Neural ODE with continuous-depth graph dynamics.",
    },
}

VALID_ARCHITECTURES = list(_ARCHITECTURE_REGISTRY.keys())
VALID_REPRESENTATIONS = ["graph", "smiles", "fingerprint", "3d_conformer"]
VALID_READOUTS = ["mean", "sum", "attention", "set2set"]
VALID_FINGERPRINTS = ["morgan", "rdkit", "maccs", "topological_torsion"]
VALID_OPTIMIZERS = ["adam", "adamw", "sgd", "lamb"]
VALID_SPLIT_STRATEGIES = ["scaffold_split", "random", "temporal", "cluster"]
VALID_LOSSES = ["bce", "mse", "focal", "evidential"]


# ---------------------------------------------------------------------------
# Drug Candidate Genome
# ---------------------------------------------------------------------------

@dataclass
class DrugCandidateGenome:
    """Complete specification for a drug-candidate predictive model.

    Encodes every axis of variation in the Descartes-Pharma C2 search space:
    architecture family, molecular representation, training recipe,
    regularisation strategy, and mechanistic-awareness options.

    Attributes
    ----------
    genome_id : str
        Unique identifier for this genome instance.
    parent_ids : List[str]
        IDs of parent genomes (empty for seed population).
    generation : int
        Evolutionary generation number (0 for seed).
    architecture : str
        Model architecture key. One of: gcn, gat, mpnn, schnet, dimenet,
        transformer, fingerprint_mlp, neural_ode.
    embedding_dim : int
        Dimensionality of the learned molecular embedding.
    n_layers : int
        Number of message-passing / encoder layers.
    dropout : float
        Dropout probability applied throughout the network.
    readout : str
        Graph-level readout strategy (mean, sum, attention, set2set).
    representation : str
        Molecular input representation (graph, smiles, fingerprint,
        3d_conformer).
    fingerprint_type : Optional[str]
        Fingerprint variant when representation == 'fingerprint'.
    fingerprint_bits : int
        Length of the fingerprint bit vector.
    use_3d : bool
        Whether to incorporate 3D conformer geometry.
    learning_rate : float
        Initial learning rate for the optimizer.
    batch_size : int
        Training mini-batch size.
    max_epochs : int
        Maximum training epochs (early stopping may terminate sooner).
    optimizer : str
        Optimizer algorithm (adam, adamw, sgd, lamb).
    split_strategy : str
        Data split strategy (scaffold_split, random, temporal, cluster).
    primary_loss : str
        Primary training loss (bce, mse, focal, evidential).
    auxiliary_mechanism_loss : str
        Auxiliary loss targeting mechanism prediction.
    aux_mechanism_weight : float
        Weighting coefficient for the auxiliary mechanism loss.
    aux_mechanism_targets : List[str]
        Mechanism-of-action labels used by the auxiliary head.
    weight_decay : float
        L2 regularisation coefficient.
    embedding_l1 : float
        L1 sparsity penalty on the embedding layer.
    information_bottleneck : bool
        Whether to apply a variational information bottleneck.
    ib_beta : float
        Beta coefficient for the information bottleneck.
    physics_priors : bool
        Inject physics-informed inductive biases (e.g. Coulomb features).
    scaffold_awareness : bool
        Enable scaffold-conditioned prediction heads.
    pharmacophore_features : bool
        Include pharmacophore-derived descriptors as auxiliary input.
    chirality_aware : bool
        Model stereochemistry / chirality information.
    dataset : str
        Target benchmark dataset identifier.
    task : str
        Prediction task (e.g. 'classification', 'regression').
    """

    # -- identity --
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0

    # -- architecture --
    architecture: str = "gcn"
    embedding_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.1
    readout: str = "mean"

    # -- representation --
    representation: str = "graph"
    fingerprint_type: Optional[str] = None
    fingerprint_bits: int = 2048
    use_3d: bool = False

    # -- training recipe --
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    optimizer: str = "adamw"
    split_strategy: str = "scaffold_split"

    # -- loss --
    primary_loss: str = "bce"
    auxiliary_mechanism_loss: str = "bce"
    aux_mechanism_weight: float = 0.1
    aux_mechanism_targets: List[str] = field(default_factory=list)

    # -- regularisation --
    weight_decay: float = 1e-4
    embedding_l1: float = 0.0
    information_bottleneck: bool = False
    ib_beta: float = 1e-3

    # -- mechanistic awareness --
    physics_priors: bool = False
    scaffold_awareness: bool = False
    pharmacophore_features: bool = False
    chirality_aware: bool = False

    # -- task --
    dataset: str = "clintox"
    task: str = "classification"


# ---------------------------------------------------------------------------
# Drug Candidate Factory
# ---------------------------------------------------------------------------

class DrugCandidateFactory:
    """Evolutionary factory for drug-candidate model genomes.

    Implements a Thompson-sampling-guided evolutionary loop:
      1. ``initialize_population`` -- seed a diverse population.
      2. ``thompson_sampling_select`` -- pick parents weighted by posterior
         performance estimates.
      3. ``crossover`` / ``mutate_genome`` -- recombine and perturb.
      4. ``evolve_generation`` -- orchestrate one full generation step.

    Parameters
    ----------
    population_size : int
        Number of genomes per generation.
    mutation_rate : float
        Per-gene probability of mutation during ``mutate_genome``.
    crossover_rate : float
        Probability of applying crossover vs. cloning a single parent.
    rng_seed : Optional[int]
        Seed for the internal NumPy random generator.
    """

    def __init__(
        self,
        population_size: int = 32,
        mutation_rate: float = 0.25,
        crossover_rate: float = 0.5,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(rng_seed)
        self._generation = 0

    # -- public helpers -----------------------------------------------------

    @staticmethod
    def get_architecture_registry() -> Dict[str, Dict[str, Any]]:
        """Return a copy of the architecture registry."""
        return copy.deepcopy(_ARCHITECTURE_REGISTRY)

    # -- population initialisation ------------------------------------------

    def initialize_population(
        self,
        dataset: str = "clintox",
        task: str = "classification",
    ) -> List[DrugCandidateGenome]:
        """Create a diverse seed population of drug-candidate genomes.

        The factory ensures every architecture family is represented at least
        once and fills remaining slots with uniformly sampled random genomes.

        Parameters
        ----------
        dataset : str
            Dataset identifier to assign to every genome.
        task : str
            Task identifier to assign to every genome.

        Returns
        -------
        List[DrugCandidateGenome]
            Seed population of size ``self.population_size``.
        """
        population: List[DrugCandidateGenome] = []

        # Guarantee at least one genome per architecture
        for arch in VALID_ARCHITECTURES:
            genome = self._random_genome(architecture=arch)
            genome.dataset = dataset
            genome.task = task
            population.append(genome)

        # Fill remaining slots with random genomes
        while len(population) < self.population_size:
            arch = self.rng.choice(VALID_ARCHITECTURES)
            genome = self._random_genome(architecture=arch)
            genome.dataset = dataset
            genome.task = task
            population.append(genome)

        self._generation = 0
        return population[: self.population_size]

    # -- mutation -----------------------------------------------------------

    def mutate_genome(self, genome: DrugCandidateGenome) -> DrugCandidateGenome:
        """Return a mutated copy of *genome*.

        Each mutable field is independently perturbed with probability
        ``self.mutation_rate``.  Continuous hyper-parameters are log-normally
        jittered; categorical parameters are uniformly re-sampled.

        Parameters
        ----------
        genome : DrugCandidateGenome
            The parent genome (not modified in place).

        Returns
        -------
        DrugCandidateGenome
            A new genome with mutations applied.
        """
        child = copy.deepcopy(genome)
        child.genome_id = uuid.uuid4().hex[:12]
        child.parent_ids = [genome.genome_id]

        # -- categorical mutations --
        if self.rng.random() < self.mutation_rate:
            child.architecture = self.rng.choice(VALID_ARCHITECTURES)
            reg = _ARCHITECTURE_REGISTRY[child.architecture]
            child.representation = reg["representation"]
            child.use_3d = reg["supports_3d"]

        if self.rng.random() < self.mutation_rate:
            child.readout = self.rng.choice(VALID_READOUTS)

        if self.rng.random() < self.mutation_rate:
            child.optimizer = self.rng.choice(VALID_OPTIMIZERS)

        if self.rng.random() < self.mutation_rate:
            child.split_strategy = self.rng.choice(VALID_SPLIT_STRATEGIES)

        if self.rng.random() < self.mutation_rate:
            child.primary_loss = self.rng.choice(VALID_LOSSES)

        if child.representation == "fingerprint" and self.rng.random() < self.mutation_rate:
            child.fingerprint_type = self.rng.choice(VALID_FINGERPRINTS)

        # -- continuous mutations (log-normal jitter) --
        child.embedding_dim = self._mutate_int(child.embedding_dim, low=32, high=1024)
        child.n_layers = self._mutate_int(child.n_layers, low=1, high=12)
        child.dropout = self._mutate_float(child.dropout, low=0.0, high=0.6)
        child.learning_rate = self._mutate_float(child.learning_rate, low=1e-5, high=1e-1, log=True)
        child.batch_size = self._mutate_int(child.batch_size, low=16, high=512)
        child.max_epochs = self._mutate_int(child.max_epochs, low=50, high=500)
        child.aux_mechanism_weight = self._mutate_float(child.aux_mechanism_weight, low=0.0, high=1.0)
        child.weight_decay = self._mutate_float(child.weight_decay, low=0.0, high=1e-1, log=True)
        child.embedding_l1 = self._mutate_float(child.embedding_l1, low=0.0, high=1e-1, log=True)
        child.ib_beta = self._mutate_float(child.ib_beta, low=1e-5, high=1.0, log=True)
        child.fingerprint_bits = self._mutate_int(child.fingerprint_bits, low=256, high=4096)

        # -- boolean mutations --
        for flag in ("information_bottleneck", "physics_priors", "scaffold_awareness",
                     "pharmacophore_features", "chirality_aware"):
            if self.rng.random() < self.mutation_rate:
                setattr(child, flag, not getattr(child, flag))

        return child

    # -- crossover ----------------------------------------------------------

    def crossover(
        self,
        parent_a: DrugCandidateGenome,
        parent_b: DrugCandidateGenome,
    ) -> DrugCandidateGenome:
        """Uniform crossover between two parent genomes.

        Each field is independently drawn from one of the two parents with
        equal probability.  Architecture-representation consistency is
        enforced after the swap.

        Parameters
        ----------
        parent_a, parent_b : DrugCandidateGenome
            Two parent genomes.

        Returns
        -------
        DrugCandidateGenome
            A new child genome.
        """
        child = DrugCandidateGenome()
        child.genome_id = uuid.uuid4().hex[:12]
        child.parent_ids = [parent_a.genome_id, parent_b.genome_id]

        fields_to_cross = [
            "architecture", "embedding_dim", "n_layers", "dropout", "readout",
            "fingerprint_type", "fingerprint_bits", "use_3d",
            "learning_rate", "batch_size", "max_epochs", "optimizer",
            "split_strategy", "primary_loss", "auxiliary_mechanism_loss",
            "aux_mechanism_weight", "aux_mechanism_targets",
            "weight_decay", "embedding_l1", "information_bottleneck",
            "ib_beta", "physics_priors", "scaffold_awareness",
            "pharmacophore_features", "chirality_aware",
        ]

        for f in fields_to_cross:
            donor = parent_a if self.rng.random() < 0.5 else parent_b
            setattr(child, f, copy.deepcopy(getattr(donor, f)))

        # Enforce architecture-representation consistency
        reg = _ARCHITECTURE_REGISTRY[child.architecture]
        child.representation = reg["representation"]
        child.use_3d = reg["supports_3d"]

        # Inherit task metadata from parent_a
        child.dataset = parent_a.dataset
        child.task = parent_a.task

        return child

    # -- Thompson sampling selection ----------------------------------------

    def thompson_sampling_select(
        self,
        population: List[DrugCandidateGenome],
        fitness_scores: np.ndarray,
        n_select: int,
    ) -> List[DrugCandidateGenome]:
        """Select parents via Thompson sampling on Beta-posteriors.

        Each genome's fitness score (assumed in [0, 1]) parameterises a
        Beta distribution.  We draw one sample per genome and pick the
        ``n_select`` genomes with the highest samples, balancing
        exploitation and exploration.

        Parameters
        ----------
        population : List[DrugCandidateGenome]
            Current generation of genomes.
        fitness_scores : np.ndarray
            Array of shape ``(len(population),)`` with fitness values in
            [0, 1].
        n_select : int
            Number of parents to select.

        Returns
        -------
        List[DrugCandidateGenome]
            Selected parent genomes.
        """
        scores = np.asarray(fitness_scores, dtype=np.float64)
        scores = np.clip(scores, 1e-6, 1.0 - 1e-6)

        # Beta posterior parameters (pseudo-counts)
        alpha = 1.0 + scores * 10.0
        beta_param = 1.0 + (1.0 - scores) * 10.0

        samples = self.rng.beta(alpha, beta_param)
        indices = np.argsort(samples)[::-1][:n_select]
        return [population[i] for i in indices]

    # -- full generation step -----------------------------------------------

    def evolve_generation(
        self,
        population: List[DrugCandidateGenome],
        fitness_scores: np.ndarray,
    ) -> List[DrugCandidateGenome]:
        """Produce the next generation via selection, crossover, and mutation.

        Workflow:
          1. Select ``population_size`` parents via Thompson sampling.
          2. For each offspring slot, either crossover two parents (with
             probability ``crossover_rate``) or clone one parent.
          3. Mutate every offspring.

        Parameters
        ----------
        population : List[DrugCandidateGenome]
            Current generation.
        fitness_scores : np.ndarray
            Fitness array of shape ``(len(population),)``.

        Returns
        -------
        List[DrugCandidateGenome]
            New generation of ``self.population_size`` genomes.
        """
        self._generation += 1
        n_parents = max(2, self.population_size // 2)
        parents = self.thompson_sampling_select(population, fitness_scores, n_parents)

        offspring: List[DrugCandidateGenome] = []
        while len(offspring) < self.population_size:
            if self.rng.random() < self.crossover_rate and len(parents) >= 2:
                idx = self.rng.choice(len(parents), size=2, replace=False)
                child = self.crossover(parents[idx[0]], parents[idx[1]])
            else:
                parent = parents[self.rng.integers(len(parents))]
                child = copy.deepcopy(parent)
                child.genome_id = uuid.uuid4().hex[:12]
                child.parent_ids = [parent.genome_id]

            child = self.mutate_genome(child)
            child.generation = self._generation
            offspring.append(child)

        return offspring[: self.population_size]

    # -- private helpers ----------------------------------------------------

    def _random_genome(self, architecture: Optional[str] = None) -> DrugCandidateGenome:
        """Sample a fully random genome, optionally fixing the architecture."""
        arch = architecture or self.rng.choice(VALID_ARCHITECTURES)
        reg = _ARCHITECTURE_REGISTRY[arch]

        fp_type = self.rng.choice(VALID_FINGERPRINTS) if reg["representation"] == "fingerprint" else None

        return DrugCandidateGenome(
            genome_id=uuid.uuid4().hex[:12],
            generation=0,
            architecture=arch,
            embedding_dim=int(self.rng.choice([64, 128, 256, 512])),
            n_layers=int(self.rng.integers(1, 9)),
            dropout=float(self.rng.uniform(0.0, 0.5)),
            readout=self.rng.choice(VALID_READOUTS),
            representation=reg["representation"],
            fingerprint_type=fp_type,
            fingerprint_bits=int(self.rng.choice([512, 1024, 2048, 4096])),
            use_3d=reg["supports_3d"],
            learning_rate=float(10 ** self.rng.uniform(-5, -2)),
            batch_size=int(self.rng.choice([16, 32, 64, 128, 256])),
            max_epochs=int(self.rng.integers(50, 401)),
            optimizer=self.rng.choice(VALID_OPTIMIZERS),
            split_strategy=self.rng.choice(VALID_SPLIT_STRATEGIES),
            primary_loss=self.rng.choice(VALID_LOSSES),
            auxiliary_mechanism_loss="bce",
            aux_mechanism_weight=float(self.rng.uniform(0.0, 0.5)),
            aux_mechanism_targets=[],
            weight_decay=float(10 ** self.rng.uniform(-6, -2)),
            embedding_l1=float(self.rng.uniform(0.0, 0.01)),
            information_bottleneck=bool(self.rng.random() < 0.3),
            ib_beta=float(10 ** self.rng.uniform(-4, 0)),
            physics_priors=bool(self.rng.random() < 0.3),
            scaffold_awareness=bool(self.rng.random() < 0.3),
            pharmacophore_features=bool(self.rng.random() < 0.3),
            chirality_aware=bool(self.rng.random() < 0.3),
        )

    def _mutate_float(
        self, value: float, low: float, high: float, log: bool = False,
    ) -> float:
        """Conditionally jitter a continuous hyper-parameter."""
        if self.rng.random() >= self.mutation_rate:
            return value
        if log and value > 0:
            log_val = np.log10(value) + self.rng.normal(0, 0.3)
            return float(np.clip(10 ** log_val, low, high))
        perturbed = value + self.rng.normal(0, 0.1) * (high - low)
        return float(np.clip(perturbed, low, high))

    def _mutate_int(self, value: int, low: int, high: int) -> int:
        """Conditionally jitter a discrete hyper-parameter."""
        if self.rng.random() >= self.mutation_rate:
            return value
        perturbed = value + int(self.rng.normal(0, 0.2) * (high - low))
        return int(np.clip(perturbed, low, high))
