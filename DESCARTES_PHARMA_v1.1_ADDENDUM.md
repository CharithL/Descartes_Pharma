# DESCARTES-PHARMA v1.1 ADDENDUM

## LLM Integration & AlphaFold Zombie Detection Module

### Extending DESCARTES-PHARMA with Large Language Model Co-Pilots and AlphaFold/DeepMind Structural Biology Integration

*March 2026*

---

## Table of Contents

A1. [What v1.0 Was Missing](#a1)
A2. [LLM Balloon Expansion: Full Implementation](#a2)
A3. [AlphaFold as DESCARTES-PHARMA Target: Is AlphaFold a Zombie?](#a3)
A4. [AlphaFold as DESCARTES-PHARMA Component: Structure-Guided Probing](#a4)
A5. [Evoformer SAE Decomposition: What Does AlphaFold Actually Know?](#a5)
A6. [AlphaFold Confidence as Zombie Indicator](#a6)
A7. [Integrated Pipeline: LLM + AlphaFold + DESCARTES-PHARMA](#a7)
A8. [New Test Datasets Enabled by AlphaFold](#a8)
A9. [Updated Implementation Roadmap](#a9)

---

## A1. What v1.0 Was Missing

DESCARTES-PHARMA v1.0 referenced LLM Balloon Expansion in the co-evolution section but did not include:

1. **Full LLM system prompts** for both C1 (probe) and C2 (drug candidate) balloon expansion
2. **LLM-guided gap analysis** for coverage detection
3. **AlphaFold integration** — the single most important structural biology tool for drug discovery
4. **Evoformer representation probing** — testing whether AlphaFold's internal representations are zombies
5. **Structure-guided mechanistic probing** — using AlphaFold-predicted structures as biological ground truth

This addendum adds all five.

---

## A2. LLM Balloon Expansion: Full Implementation

### A2.1 When LLM Balloon Activates

The LLM balloon fires when compositional search stalls — after N rounds without improvement in the mechanistic validation score. The LLM proposes novel:

- **For C2 (Drug Candidate Factory):** New model architectures, training strategies, molecular representations, loss functions
- **For C1 (Probing Factory):** New probe methods, transforms, null distributions, conditional probing strategies

### A2.2 C2 Balloon: Novel Drug Model Architectures

```python
SYSTEM_DRUG_MODEL_BALLOON = """You are a computational chemist and deep 
learning architect advising the DESCARTES-PHARMA Dual Factory. The factory 
is searching for drug discovery AI models that are NOT "pharmaceutical 
zombies" — models whose internal embeddings genuinely correspond to 
molecular mechanisms of action, not just statistical shortcuts.

The factory has exhausted compositional search of standard architectures.

Key context:
{dataset_context}
Target property: {target_property}
Known mechanisms: {known_mechanisms}

Results so far:
- Architectures tested: {architectures_tested}
- Best mechanistic score: {best_mechanism_score}
- Best prediction accuracy: {best_accuracy}
- Total genomes evaluated: {n_genomes}
- Zombie verdict distribution: {verdict_distribution}

Key insight from DESCARTES-PHARMA: GNN models with explicit 3D geometry
(SchNet, DimeNet, GemNet) tend to encode real biophysics, while 2D 
fingerprint models find shortcuts. The question: is there an architecture
between physics-based docking (too slow) and fingerprint-MLP (too zombie)
that naturally recovers molecular mechanisms?

AlphaFold 3 is available as a structural oracle. Its Evoformer 
representations, pair representations, and predicted structures can be 
used as features, teacher signals, or validation targets.

Propose 3-5 NOVEL model architectures or training strategies. Think about:
- AlphaFold Evoformer features as pre-trained molecular representations
- Equivariant neural networks with AlphaFold-predicted binding pockets
- Contrastive learning between 2D graphs and AlphaFold 3D structures
- Physics-informed GNNs with AlphaFold confidence as regularizer
- Multi-scale models: atom-level + residue-level + pocket-level
- Knowledge distillation from AlphaFold to lightweight drug models
- Attention over AlphaFold pair representations for binding prediction
- Hybrid: molecular dynamics refinement of AlphaFold poses + ML scoring

Respond with ONLY valid JSON. Array of 3-5 proposed architectures, each with:
  - name: descriptive identifier
  - architecture_type: base class to extend
  - alphafold_integration: how AlphaFold is used (features/teacher/validator/none)
  - modification: what makes this novel
  - loss_recipe: proposed training losses
  - hypothesis: why this might avoid zombie solutions
  - implementation_sketch: key PyTorch patterns (3-5 lines)
  - estimated_zombie_risk: LOW/MEDIUM/HIGH with rationale
"""


SYSTEM_DRUG_MODEL_GAP = """You are analyzing gaps in the DESCARTES-PHARMA 
factory's search of drug discovery model architectures.

Current coverage:
{coverage_summary}

AlphaFold integration status:
- AlphaFold features used: {af_features_tested}
- AlphaFold as teacher: {af_teacher_tested}
- AlphaFold as validator: {af_validator_tested}

Results:
- Best mechanistic score: {best_score}
- Architecture families tested: {families}
- Molecular representations tested: {representations}
- Loss functions tested: {losses}

Gaps detected:
{gap_list}

For each gap, propose a specific model genome that would fill it.
Consider whether AlphaFold integration could address the gap.

Key principle: zombie risk decreases with:
1. Explicit 3D geometry (vs 2D graph topology only)
2. Physics-based priors (electrostatics, van der Waals, solvation)
3. AlphaFold structural grounding (predicted pose as constraint)
4. Multi-resolution representations (atom + residue + pocket)
5. Mechanistic auxiliary losses (predict binding pose, not just affinity)

Respond with ONLY valid JSON."""
```

### A2.3 C1 Balloon: Novel Probing Strategies

```python
SYSTEM_PROBE_BALLOON_PHARMA = """You are a mechanistic interpretability 
researcher advising the DESCARTES-PHARMA factory. The probing factory is 
searching for methods to detect molecular mechanism encoding in drug 
discovery model embeddings.

Current probe results for this model:
{probe_results_summary}

Methods already tried:
{methods_tried}

The following mechanisms remain undetected (zombie classification):
{zombie_mechanisms}

Known about these mechanisms:
{mechanism_properties}

AlphaFold 3 provides:
- Predicted binding poses for all compounds in dataset
- Evoformer pair representations (interaction features)
- Per-residue confidence scores (pLDDT)
- Predicted aligned error (PAE) for interaction quality

Propose 3-5 NOVEL probing strategies. Consider:
- The mechanism might be encoded in a rotated coordinate system
  → Use DAS (Distributed Alignment Search) to find encoding direction
- The mechanism might be in polypharmacological superposition
  → Use SAE with higher expansion factor (16×) focused on this target
- The mechanism might only be encoded for specific scaffolds
  → Scaffold-conditional probing
- AlphaFold binding pose features as alternative ground truth
  → Probe for binding pose correspondence instead of descriptor correspondence
- The mechanism might be in attention weights, not embeddings
  → Attention-based probing of GNN message passing
- The mechanism might require 3D spatial encoding detection
  → Probe for distance/angle/dihedral encoding

Respond with ONLY valid JSON."""
```

### A2.4 LLM Balloon Expander Implementation

```python
class PharmaLLMBalloonExpander:
    """
    LLM balloon expansion for DESCARTES-PHARMA.
    
    Uses Anthropic Claude API for generating novel:
    - Drug model architectures (C2 factory)
    - Probing strategies (C1 factory)
    - AlphaFold integration ideas
    - Gap-filling configurations
    """
    
    def __init__(self, model='claude-sonnet-4-20250514'):
        self.model = model
    
    def propose_novel_architectures(self, factory_state, dataset_context,
                                      known_mechanisms):
        """Ask LLM to propose novel drug model architectures."""
        import json, requests
        
        prompt = SYSTEM_DRUG_MODEL_BALLOON.format(
            dataset_context=dataset_context,
            target_property=factory_state.get('target', 'toxicity'),
            known_mechanisms=known_mechanisms,
            architectures_tested=factory_state['architectures_tested'],
            best_mechanism_score=factory_state['best_mechanism_score'],
            best_accuracy=factory_state['best_accuracy'],
            n_genomes=factory_state['n_genomes'],
            verdict_distribution=factory_state['verdict_distribution']
        )
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={'Content-Type': 'application/json'},
            json={
                'model': self.model,
                'max_tokens': 2000,
                'messages': [{'role': 'user', 'content': prompt}]
            }
        )
        
        data = response.json()
        text = data['content'][0]['text']
        
        try:
            proposals = json.loads(text)
            return [self._proposal_to_genome(p) for p in proposals]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"LLM balloon parse error: {e}")
            return []
    
    def propose_novel_probes(self, probe_results, zombie_mechanisms,
                               alphafold_available=True):
        """Ask LLM to propose novel probing strategies."""
        import json, requests
        
        prompt = SYSTEM_PROBE_BALLOON_PHARMA.format(
            probe_results_summary=str(probe_results),
            methods_tried=list(probe_results.keys()),
            zombie_mechanisms=zombie_mechanisms,
            mechanism_properties="AlphaFold structural data available" 
                if alphafold_available else "No structural data"
        )
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={'Content-Type': 'application/json'},
            json={
                'model': self.model,
                'max_tokens': 2000,
                'messages': [{'role': 'user', 'content': prompt}]
            }
        )
        
        data = response.json()
        text = data['content'][0]['text']
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return []
    
    def _proposal_to_genome(self, proposal):
        """Convert LLM proposal to DrugCandidateGenome."""
        from dataclasses import fields
        genome = DrugCandidateGenome()
        
        arch = proposal.get('architecture_type', 'gcn')
        if arch in ['gcn', 'gat', 'mpnn', 'schnet', 'dimenet',
                     'transformer', 'fingerprint_mlp', 'neural_ode']:
            genome.architecture = arch
        
        af_integration = proposal.get('alphafold_integration', 'none')
        if af_integration != 'none':
            genome.physics_priors = True
            genome.use_3d = True
        
        if 'bottleneck' in str(proposal.get('modification', '')).lower():
            genome.information_bottleneck = True
        
        if 'mechanism' in str(proposal.get('loss_recipe', '')).lower():
            genome.auxiliary_mechanism_loss = True
            genome.aux_mechanism_weight = 0.1
        
        return genome
```

---

## A3. AlphaFold as DESCARTES-PHARMA Target: Is AlphaFold a Zombie?

### A3.1 The Critical Question

AlphaFold 3 predicts protein-ligand binding poses with remarkable accuracy — 50% more accurate than the best traditional methods on the PoseBusters benchmark. But does AlphaFold's Evoformer actually encode real binding physics, or does it find statistical shortcuts in the training data?

AF3 excels at predicting static protein-ligand interactions with minimal conformational changes, significantly outperforming traditional docking methods in side-chain orientation accuracy. However, AF3 struggles with protein-ligand complexes involving significant conformational changes (>5Å RMSD) and demonstrates a persistent bias toward active GPCR conformations.

This conformational bias is a **zombie signature**: AlphaFold predicts the most common conformation in the training data rather than the biophysically correct conformation for a given ligand. It produces the right output (close-to-correct structure) for the wrong internal reason (memorized conformation, not physics).

### A3.2 AlphaFold Zombie Detection Framework

```python
class AlphaFoldZombieDetector:
    """
    Apply DESCARTES-PHARMA probing to AlphaFold's internal representations.
    
    AlphaFold's architecture has clear internal state variables:
    - Single representations (per-residue features) ~ analogous to LSTM hidden states
    - Pair representations (residue-residue features) ~ analogous to attention matrices
    - Evoformer outputs (final processed features)
    - Diffusion module intermediate states
    
    Known biophysical ground truth to probe for:
    - Electrostatic potential at binding site
    - Hydrophobic surface area
    - Hydrogen bond donor/acceptor geometry
    - Van der Waals complementarity
    - Solvation/desolvation energetics
    - Conformational strain energy
    - Binding pocket volume and shape
    
    The zombie question: does AlphaFold's Evoformer encode these
    biophysical features, or does it memorize structural templates?
    """
    
    def __init__(self, alphafold_model, device='cpu'):
        self.model = alphafold_model
        self.device = device
    
    def extract_evoformer_representations(self, input_features):
        """
        Extract internal representations from AlphaFold's Evoformer.
        
        Returns:
            single_repr: (N_residues, D_single) per-residue features
            pair_repr: (N_residues, N_residues, D_pair) pairwise features
            These are the "hidden states" to probe for biophysics.
        """
        # Hook into Evoformer intermediate layers
        representations = {}
        
        def hook_single(module, input, output):
            representations['single'] = output.detach().cpu().numpy()
        
        def hook_pair(module, input, output):
            representations['pair'] = output.detach().cpu().numpy()
        
        # Register hooks on Evoformer output
        # (Exact layer names depend on AlphaFold implementation)
        handles = []
        for name, module in self.model.named_modules():
            if 'evoformer' in name and 'single' in name:
                handles.append(module.register_forward_hook(hook_single))
            if 'evoformer' in name and 'pair' in name:
                handles.append(module.register_forward_hook(hook_pair))
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_features)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        return representations
    
    def compute_biophysical_ground_truth(self, protein_structure, ligand):
        """
        Compute known biophysical features from experimental structure.
        These serve as ground truth for probing AlphaFold representations.
        
        Returns per-residue biophysical features to compare against
        AlphaFold's single representations.
        """
        # Requires: biopython, rdkit, or similar
        features = {}
        
        # 1. Electrostatic potential (Coulomb energy per residue)
        features['electrostatic'] = self._compute_electrostatics(
            protein_structure, ligand)
        
        # 2. Hydrophobic contact area per residue
        features['hydrophobic_contact'] = self._compute_hydrophobic_contact(
            protein_structure, ligand)
        
        # 3. H-bond geometry per residue
        features['hbond_score'] = self._compute_hbond_scores(
            protein_structure, ligand)
        
        # 4. VdW complementarity per residue
        features['vdw_complementarity'] = self._compute_vdw(
            protein_structure, ligand)
        
        # 5. Solvent accessibility change upon binding
        features['delta_sasa'] = self._compute_delta_sasa(
            protein_structure, ligand)
        
        # 6. Distance to binding site centroid
        features['binding_site_distance'] = self._compute_binding_distance(
            protein_structure, ligand)
        
        # 7. B-factor / flexibility
        features['flexibility'] = self._extract_bfactors(protein_structure)
        
        return features
    
    def probe_evoformer_for_biophysics(self, representations, 
                                         biophysical_features):
        """
        Run DESCARTES-PHARMA probing on AlphaFold representations.
        
        This is the core zombie test for AlphaFold:
        Do Evoformer single representations encode electrostatics,
        hydrophobic contacts, H-bonds, etc.?
        
        Uses identical probe methods as DESCARTES-PHARMA v1.0:
        Ridge ΔR², MLP ΔR², SAE decomposition, CCA, RSA, CKA,
        resample ablation, MINE, MDL.
        """
        single_repr = representations['single']
        
        results = {}
        for feature_name, feature_values in biophysical_features.items():
            # Ridge ΔR²
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score
            
            ridge = Ridge(alpha=1.0)
            scores = cross_val_score(ridge, single_repr, feature_values, cv=5)
            ridge_r2 = np.mean(scores)
            
            # MLP ΔR² (mandatory companion)
            mlp_r2 = self._mlp_probe(single_repr, feature_values)
            
            # Random baseline
            random_repr = np.random.randn(*single_repr.shape)
            random_scores = cross_val_score(Ridge(1.0), random_repr, 
                                             feature_values, cv=5)
            random_r2 = np.mean(random_scores)
            
            delta_r2_ridge = ridge_r2 - random_r2
            delta_r2_mlp = mlp_r2 - random_r2
            
            results[feature_name] = {
                'ridge_r2': ridge_r2,
                'mlp_r2': mlp_r2,
                'random_r2': random_r2,
                'delta_r2_ridge': delta_r2_ridge,
                'delta_r2_mlp': delta_r2_mlp,
                'encoding_type': self._classify_encoding(
                    delta_r2_ridge, delta_r2_mlp),
            }
        
        # Aggregate verdict
        n_encoded = sum(1 for r in results.values() 
                        if r['encoding_type'] != 'ZOMBIE')
        n_total = len(results)
        
        results['aggregate'] = {
            'n_encoded': n_encoded,
            'n_total': n_total,
            'fraction_encoded': n_encoded / n_total,
            'verdict': ('CONFIRMED_MECHANISTIC' if n_encoded >= n_total * 0.7
                       else 'PARTIAL_ZOMBIE' if n_encoded >= n_total * 0.3
                       else 'CONFIRMED_ZOMBIE')
        }
        
        return results
    
    def _classify_encoding(self, ridge_delta, mlp_delta, threshold=0.05):
        if ridge_delta > threshold and mlp_delta > threshold:
            return 'LINEAR_ENCODED' if mlp_delta < ridge_delta + 0.1 \
                else 'NONLINEAR_ENCODED'
        elif mlp_delta > threshold:
            return 'NONLINEAR_ONLY'
        return 'ZOMBIE'
    
    def _mlp_probe(self, representations, target, hidden_dim=64, epochs=50):
        """MLP probe with controlled capacity."""
        # [Same as DESCARTES v3.0 MLP probe implementation]
        pass
```

### A3.3 Known AlphaFold Zombie Signatures

Based on published benchmarks, AlphaFold 3 shows specific zombie-like behaviors that DESCARTES-PHARMA probing should detect:

| Behavior | Zombie Type | DESCARTES-PHARMA Probe |
|---|---|---|
| Bias toward active GPCR conformations regardless of ligand | Template memorization zombie | Conditional probing: ligand-class-stratified R² |
| Difficulty predicting dynamic systems and disordered regions | Static-bias zombie | Temporal/dynamical probes on MD trajectories |
| Struggles with conformational changes >5Å RMSD | Distribution-edge zombie | Scaffold-resolved R² for flexible vs rigid targets |
| Predicts E3 ligases always in closed conformation | Majority-class zombie | SAE feature probing for open/closed state encoding |
| Requires complementary approaches for affinity ranking | Rank-order zombie | Transfer entropy: structure → affinity direction |

---

## A4. AlphaFold as DESCARTES-PHARMA Component: Structure-Guided Probing

### A4.1 AlphaFold Provides Ground Truth for Probing

Instead of only probing drug models for RDKit descriptor correspondence (MW, logP, TPSA), AlphaFold provides **3D structural ground truth** that is far more mechanistically relevant:

```python
class AlphaFoldGroundTruthGenerator:
    """
    Use AlphaFold 3 to generate structural ground truth features
    for DESCARTES-PHARMA mechanistic probing.
    
    For each drug-target pair in the dataset:
    1. Predict binding pose with AlphaFold 3
    2. Extract 3D structural features from the predicted complex
    3. Use these features as "biological variables" for probing
    
    This is MUCH better than probing for RDKit descriptors because:
    - RDKit descriptors are input features (trivially decodable)
    - AlphaFold structural features are EMERGENT (binding pose, 
      interaction geometry) — encoding them requires genuine 
      understanding of molecular interactions
    """
    
    def __init__(self, alphafold_server_url=None):
        """
        Can use:
        1. AlphaFold Server API (free, non-commercial)
        2. Local AlphaFold 3 installation (academic weights available since Nov 2024)
        3. Pre-computed AlphaFold DB structures
        """
        self.server_url = alphafold_server_url
    
    def generate_structural_ground_truth(self, protein_sequence, 
                                           ligand_smiles_list):
        """
        Generate 3D structural features for a set of ligands
        binding to a target protein.
        
        Returns per-compound structural features to use as
        probe targets (biological variables) in DESCARTES-PHARMA.
        """
        structural_features = []
        feature_names = []
        
        for smiles in ligand_smiles_list:
            # Predict binding pose
            pose = self._predict_binding_pose(protein_sequence, smiles)
            
            if pose is None:
                structural_features.append([np.nan] * 15)
                continue
            
            features = self._extract_binding_features(pose)
            structural_features.append(features)
        
        feature_names = [
            'binding_pocket_volume',        # Å³
            'binding_pocket_depth',         # Å
            'n_protein_contacts',           # Count of contacts < 4Å
            'n_hbonds_predicted',           # Predicted H-bonds
            'buried_surface_area',          # Å²
            'shape_complementarity',        # Lawrence & Colman Sc score
            'electrostatic_complementarity', # Charge matching at interface
            'hydrophobic_contact_area',     # Å² of nonpolar-nonpolar contact
            'ligand_strain_energy',         # kcal/mol, conformational penalty
            'binding_site_flexibility',     # Average B-factor / pLDDT inverse
            'water_displacement_count',     # Predicted displaced waters
            'entrance_channel_width',       # Å, accessibility metric
            'aromatic_stacking_count',      # Pi-pi and CH-pi interactions
            'metal_coordination_count',     # Metal-ligand bonds
            'allosteric_distance',          # Å to known allosteric sites
        ]
        
        return np.array(structural_features), feature_names
    
    def _predict_binding_pose(self, protein_seq, ligand_smiles):
        """Call AlphaFold 3 for structure prediction."""
        # Option 1: AlphaFold Server API
        # Option 2: Local inference with open-sourced weights
        # Option 3: Pre-computed from AlphaFold DB
        pass
    
    def _extract_binding_features(self, predicted_complex):
        """Extract 15 structural features from predicted binding pose."""
        pass
```

### A4.2 Three Modes of AlphaFold Integration

```
MODE 1: ALPHAFOLD AS GROUND TRUTH ORACLE
  Input: drug-target pairs from dataset
  AlphaFold: predicts binding poses
  Output: structural features used as probe targets
  Question: "Does the drug model encode binding geometry?"
  
MODE 2: ALPHAFOLD AS PRE-TRAINED ENCODER
  Input: protein sequences + ligand SMILES
  AlphaFold: generates Evoformer representations
  Output: pre-trained features used as model inputs
  Question: "Does fine-tuning preserve AlphaFold's biophysics?"
  
MODE 3: ALPHAFOLD AS TEACHER (DISTILLATION)
  Input: drug-target pairs
  AlphaFold: teacher model predicting structures
  Drug model: student model predicting activity
  Loss: activity_loss + α × evoformer_alignment_loss
  Question: "Does distillation transfer mechanism or just accuracy?"
```

---

## A5. Evoformer SAE Decomposition

### A5.1 The Key Experiment

Apply SAE decomposition (from DESCARTES v3.0 Section 3) directly to AlphaFold's Evoformer representations. This is the **direct analog** of the InterPLM study (Nature Methods 2025) that found up to 2,548 interpretable features per ESM-2 layer using SAEs.

```python
def evoformer_sae_analysis(alphafold_model, protein_ligand_pairs,
                             expansion_factor=8, k=30, device='cpu'):
    """
    Train SAE on AlphaFold Evoformer single representations.
    
    Then probe SAE features for known biophysical properties.
    
    Expected findings:
    - Some SAE features = electrostatic complementarity
    - Some SAE features = hydrophobic packing
    - Some SAE features = H-bond networks
    - Some SAE features = TEMPLATE MEMORIZATION (zombie features!)
    - Some SAE features = DATASET ARTIFACTS (PDB deposition bias)
    
    The ratio of biophysical to artifact features is the 
    mechanistic encoding score.
    """
    # 1. Extract Evoformer representations for all pairs
    all_representations = []
    detector = AlphaFoldZombieDetector(alphafold_model, device)
    
    for protein_seq, ligand_smiles in protein_ligand_pairs:
        features = prepare_alphafold_input(protein_seq, ligand_smiles)
        repr_dict = detector.extract_evoformer_representations(features)
        
        # Pool single representations to per-complex vector
        single = repr_dict['single']  # (N_residues, D)
        # Take binding site residues only (within 8Å of ligand)
        binding_site = select_binding_site_residues(single, repr_dict)
        pooled = binding_site.mean(axis=0)  # (D,)
        all_representations.append(pooled)
    
    all_repr = np.stack(all_representations)
    
    # 2. Train SAE
    sae, loss_history = train_sae(
        [all_repr], all_repr.shape[1],
        expansion_factor=expansion_factor, k=k, device=device)
    
    # 3. Probe SAE features for biophysical ground truth
    # (Uses AlphaFold's OWN structural predictions as ground truth,
    #  creating a self-consistency check)
    af_ground_truth = AlphaFoldGroundTruthGenerator()
    structural_features, feature_names = af_ground_truth\
        .generate_structural_ground_truth(
            protein_sequences, ligand_smiles_list)
    
    # 4. Probe
    results = sae_probe_molecular_mechanisms(
        sae, all_repr, structural_features, feature_names, device)
    
    return results
```

---

## A6. AlphaFold Confidence as Zombie Indicator

### A6.1 pLDDT and PAE as Zombie Flags

AlphaFold provides per-residue confidence (pLDDT) and predicted aligned error (PAE). These can serve as **zombie indicators**:

```python
def alphafold_confidence_zombie_flags(plddt_scores, pae_matrix,
                                        binding_site_residues):
    """
    AlphaFold's own confidence scores flag potential zombie predictions.
    
    Low pLDDT at binding site → AlphaFold is uncertain about 
    binding geometry → structure may be memorized template, not physics.
    
    High PAE between protein and ligand → AlphaFold doesn't understand
    the protein-ligand spatial relationship → zombie binding prediction.
    """
    # Binding site pLDDT
    bs_plddt = plddt_scores[binding_site_residues]
    mean_bs_plddt = np.mean(bs_plddt)
    
    # Protein-ligand PAE (cross-chain)
    # PAE[i,j] = expected position error of residue j when aligned on residue i
    # High cross-chain PAE = uncertain relative positioning
    ligand_residue_idx = len(plddt_scores) - 1  # Last "residue" = ligand
    cross_pae = pae_matrix[binding_site_residues, ligand_residue_idx]
    mean_cross_pae = np.mean(cross_pae)
    
    flags = {
        'binding_site_plddt': float(mean_bs_plddt),
        'cross_chain_pae': float(mean_cross_pae),
        'confident_binding': mean_bs_plddt > 70 and mean_cross_pae < 10,
        'zombie_risk': (
            'LOW' if mean_bs_plddt > 80 and mean_cross_pae < 5 else
            'MEDIUM' if mean_bs_plddt > 60 and mean_cross_pae < 15 else
            'HIGH'
        ),
        'interpretation': (
            'AlphaFold is confident in this binding prediction'
            if mean_bs_plddt > 70 and mean_cross_pae < 10
            else 'AlphaFold may be template-matching rather than computing physics'
        )
    }
    
    return flags
```

---

## A7. Integrated Pipeline: LLM + AlphaFold + DESCARTES-PHARMA

### A7.1 The Complete Workflow

```
╔══════════════════════════════════════════════════════════════════════╗
║         DESCARTES-PHARMA v1.1: INTEGRATED PIPELINE                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  INPUT: Drug-target dataset (SMILES + target sequences + labels)    ║
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │ STAGE 0: ALPHAFOLD STRUCTURAL ENRICHMENT                    │    ║
║  │                                                             │    ║
║  │  For each drug-target pair:                                 │    ║
║  │  1. AlphaFold 3 → predicted binding pose                   │    ║
║  │  2. Extract 15 structural ground truth features             │    ║
║  │  3. Extract pLDDT / PAE confidence flags                    │    ║
║  │  4. Extract Evoformer representations (if probing AF itself)│    ║
║  └──────────────────────────┬──────────────────────────────────┘    ║
║                             │                                        ║
║  ┌──────────────────────────▼──────────────────────────────────┐    ║
║  │ STAGE 1: DUAL FACTORY CO-EVOLUTION                          │    ║
║  │                                                             │    ║
║  │  C2 (Drug Model Factory)     C1 (Probing Factory)          │    ║
║  │  ├── GCN, GAT, MPNN         ├── Ridge ΔR² + MLP ΔR²       │    ║
║  │  ├── SchNet, DimeNet         ├── SAE polypharmacology       │    ║
║  │  ├── AF-pretrained encoder   ├── CCA / RSA / CKA            │    ║
║  │  ├── AF-distilled models     ├── Resample ablation          │    ║
║  │  └── LLM Balloon → novel    └── LLM Balloon → novel        │    ║
║  │                                                             │    ║
║  │  Probe targets now include:                                 │    ║
║  │  1. RDKit descriptors (baseline)                            │    ║
║  │  2. AlphaFold structural features (3D ground truth)         │    ║
║  │  3. Known pathway activations (biological ground truth)     │    ║
║  └──────────────────────────┬──────────────────────────────────┘    ║
║                             │                                        ║
║  ┌──────────────────────────▼──────────────────────────────────┐    ║
║  │ STAGE 2: STATISTICAL HARDENING (13 methods)                 │    ║
║  │  + AlphaFold confidence-weighted significance                │    ║
║  │  + Scaffold-stratified permutation                          │    ║
║  │  + FDR across all mechanisms × models × datasets            │    ║
║  └──────────────────────────┬──────────────────────────────────┘    ║
║                             │                                        ║
║  ┌──────────────────────────▼──────────────────────────────────┐    ║
║  │ STAGE 3: ZOMBIE VERDICT GENERATOR                           │    ║
║  │                                                             │    ║
║  │  Per model × per mechanism × per dataset:                   │    ║
║  │  → CAUSALLY_VALIDATED / CONFIRMED_MECHANISTIC /             │    ║
║  │    POLYPHARMACOLOGY_DETECTED / NONLINEAR_MECHANISM /        │    ║
║  │    CANDIDATE_MECHANISTIC / SPURIOUS_SCAFFOLD /              │    ║
║  │    LIKELY_ZOMBIE / CONFIRMED_ZOMBIE                         │    ║
║  │                                                             │    ║
║  │  NEW: AlphaFold-informed confidence weighting               │    ║
║  │  → High-confidence AF poses weight verdict more heavily     │    ║
║  │  → Low-confidence AF poses flag uncertainty in ground truth │    ║
║  └──────────────────────────┬──────────────────────────────────┘    ║
║                             │                                        ║
║  OUTPUT: Mechanistic Validation Report                              ║
║  - Per-model zombie scores                                          ║
║  - Per-mechanism encoding maps                                      ║
║  - AlphaFold self-consistency scores                                ║
║  - Recommendation: ADVANCE / REDESIGN / KILL                        ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## A8. New Test Datasets Enabled by AlphaFold

### A8.1 PoseBusters Benchmark

AlphaFold 3 was benchmarked against PoseBusters — a curated set of protein-ligand co-crystal structures. This dataset has **experimental 3D ground truth** for both the protein AND the ligand, making it ideal for DESCARTES-PHARMA structure-level probing.

### A8.2 PDBbind

~23,000 protein-ligand complexes with experimentally measured binding affinities AND crystal structures. Train a model on molecular features → binding affinity, then probe whether internal representations encode the structural features visible in the crystal structures.

### A8.3 CASF Benchmark

The Comparative Assessment of Scoring Functions provides standardized tests for docking accuracy, binding affinity prediction, and ranking. Each test has known structural ground truth — perfect for zombie detection.

### A8.4 Updated Dataset Tier Table

```
TIER 1 — SYNTHETIC GROUND TRUTH (unchanged)
├── HH Simulator                      [2 hours, perfect ground truth]

TIER 2 — PHARMA BENCHMARKS (unchanged + additions)
├── TDC ClinTox / BBBP / Tox21       [30 min setup]
├── PharmaBench ADMET                 [1 hour setup]
+ ├── PoseBusters (AF3 benchmark)     [1 hour, 3D ground truth]
+ └── PDBbind refined set             [1 day, binding + structure]

TIER 3 — NEUROSCIENCE + STRUCTURAL (expanded)
├── Allen Brain Observatory           [1 day]
+ ├── AlphaFold DB structures         [free API, 200M structures]
+ └── CASF benchmark                  [1 day, scoring function validation]

TIER 4 — ADVANCED (unchanged + additions)
├── RxRx3-core / GDSC / BELKA        [1 day each]
+ └── Isomorphic Labs public data     [if/when released]
```

---

## A9. Updated Implementation Roadmap

### Week 1-2: Ground Truth + LLM Setup (same as v1.0, plus:)

```
Day 5 (NEW): LLM Balloon Infrastructure
  - Set up Anthropic API connection
  - Implement C1 and C2 balloon prompts
  - Test on HH simulator (ask LLM for novel probe strategies)
  - Validate: do LLM-proposed probes find m, h, n on HH data?
```

### Week 3-4: Pharma + AlphaFold Integration

```
Day 8-9 (UPDATED): TDC ClinTox with AlphaFold Ground Truth
  - Load ClinTox dataset
  - For each compound: query AlphaFold Server for target structure
  - Compute structural binding features as probe targets
  - Run probes on GCN embeddings vs BOTH RDKit + AlphaFold features
  → Compare: is structural probing more discriminative than descriptor probing?

Day 10-11 (NEW): AlphaFold Zombie Detection
  - Download AlphaFold 3 academic weights
  - Run on PoseBusters subset (50 complexes)
  - Extract Evoformer representations
  - Probe for biophysical features (electrostatics, H-bonds, hydrophobics)
  - Generate zombie verdict for AlphaFold itself
  → KEY RESULT: Is AlphaFold a zombie for specific biophysical features?

Day 12-14 (NEW): Evoformer SAE Decomposition
  - Train SAE on Evoformer single representations (expansion 8×)
  - Map SAE features to biophysical properties
  - Identify monosemantic vs polysemantic Evoformer features
  - Compare to InterPLM ESM-2 SAE results
  → What does AlphaFold ACTUALLY know about binding physics?
```

### Week 5-6: LLM-Guided Search Expansion

```
Day 15-16 (NEW): LLM Balloon for Drug Models
  - Run 50 C2 genomes through probing factory
  - Trigger LLM balloon when improvement stalls
  - LLM proposes AlphaFold-integrated architectures
  - Test: do LLM-proposed architectures score better on mechanism probes?

Day 17-18 (NEW): LLM Balloon for Probing Strategies
  - For mechanisms classified ZOMBIE across all models:
  - LLM proposes novel probe approaches
  - Implement and test top 3 LLM proposals
  - Update probe library with successful discoveries
```

### Week 7: Integration (same as v1.0, plus AlphaFold report)

---

## Summary: What This Addendum Adds

| Component | v1.0 Status | v1.1 Addition |
|---|---|---|
| LLM Balloon Expansion | Mentioned briefly | Full system prompts + implementation |
| LLM Gap Analysis | Not included | Complete gap detection + filling |
| AlphaFold as target | Not included | Full zombie detection framework for AF |
| AlphaFold as component | Not included | 3 integration modes (oracle/encoder/teacher) |
| Evoformer SAE | Not included | SAE decomposition of Evoformer features |
| AF confidence flags | Not included | pLDDT/PAE as zombie indicators |
| Structural ground truth | RDKit descriptors only | 15 AlphaFold structural features |
| PoseBusters/PDBbind | Not included | New Tier 2-3 test datasets |
| Integrated pipeline diagram | Separate components | Full Stage 0-3 workflow |

---

*This addendum transforms DESCARTES-PHARMA from a descriptor-probing framework into a structure-aware mechanistic validation system. By integrating AlphaFold as both target and component, and LLMs as search co-pilots, v1.1 can answer not just "does this model encode molecular properties?" but "does this model encode the 3D physics of molecular binding?" — the question that determines whether a drug discovery AI produces genuine mechanistic insight or sophisticated pattern matching.*
