# DESCARTES-PHARMA: Disease-Specific Discovery Probes
# From Zombie Detection → Drug Discovery
#
# The key shift: instead of probing for GENERIC molecular features 
# (LogP, TPSA — what we did on ClinTox/BBBP/Tox21),
# probe for DISEASE-SPECIFIC mechanisms.
#
# A model that genuinely encodes disease mechanisms (non-zombie)
# can be trusted to screen for new drug candidates.
# A model that doesn't (zombie) will produce candidates that 
# fail in clinical trials — the $42B Alzheimer's lesson.

"""
=================================================================
PART 1: ALZHEIMER'S DISEASE DISCOVERY PROBE
=================================================================

The Alzheimer's drug discovery graveyard has 6 major target classes.
Each has known molecular features that a non-zombie model MUST encode
to be trusted for that target.

For each target, we define:
  1. DATASET: where to get molecules with known activity at this target
  2. GROUND TRUTH FEATURES: computable molecular properties that 
     correlate with activity at this target for known mechanistic reasons
  3. ZOMBIE TEST: what a zombie model would encode instead (shortcuts)
  4. DISCOVERY MODE: how to use a validated non-zombie model to find
     new candidates
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DiseaseProbeSpec:
    """Specification for a disease-specific mechanistic probe."""
    
    disease: str
    target_name: str
    target_description: str
    
    # Where to get data
    datasets: List[str]
    dataset_source: str  # TDC, ChEMBL, PubChem, custom
    n_compounds_expected: int
    
    # Mechanistic ground truth features to probe for
    # These are the "biological variables" (like m, h, n in HH)
    mechanism_features: List[Dict[str, str]]
    # Each dict has: name, computation, biological_rationale, expected_direction
    
    # Known confounds to regress out
    confounds: List[str]
    
    # What a zombie would encode instead
    zombie_shortcuts: List[str]
    
    # How to use validated model for discovery
    discovery_strategy: str


# =================================================================
# ALZHEIMER'S PROBES
# =================================================================

ALZHEIMER_PROBES = [
    
    DiseaseProbeSpec(
        disease="Alzheimer's",
        target_name="BACE1_inhibition",
        target_description="""
        Beta-secretase 1 (BACE1) cleaves amyloid precursor protein (APP)
        to produce amyloid-beta peptides. BACE1 inhibitors reduce Aβ
        production. Multiple Phase 3 failures (verubecestat, atabecestat,
        lanabecestat) — but the TARGET is validated, the MOLECULES weren't
        selective enough. A non-zombie model could find better ones.
        """,
        
        datasets=[
            "ChEMBL BACE1 bioactivity (assay CHEMBL3650, ~4000 compounds)",
            "TDC HTS dataset (if BACE1 included)",
            "MoleculeNet BACE dataset (1,513 compounds, binary classification)",
        ],
        dataset_source="MoleculeNet BACE + ChEMBL",
        n_compounds_expected=4000,
        
        mechanism_features=[
            {
                "name": "aspartyl_protease_pharmacophore",
                "computation": """
                    BACE1 is an aspartyl protease. Inhibitors must have:
                    - A transition-state isostere (hydroxyethylamine, 
                      hydroxyethylene, or statine core)
                    - Hydrogen bond donors to catalytic Asp32 and Asp228
                    - Hydrophobic group filling S1' pocket
                    
                    Compute: presence of hydroxyethylamine/statine substructure
                    (SMARTS pattern matching via RDKit)
                """,
                "biological_rationale": "Direct interaction with catalytic site",
                "expected_direction": "positive (present → active)",
            },
            {
                "name": "S1_pocket_complementarity",
                "computation": """
                    BACE1 S1 pocket is hydrophobic (Leu30, Tyr71, Phe108).
                    Compute: hydrophobic surface area of the portion of 
                    the molecule expected to occupy S1 pocket.
                    Proxy: aromatic ring count in the "west" portion of 
                    the molecule (relative to pharmacophore core).
                """,
                "biological_rationale": "Shape complementarity with binding site",
                "expected_direction": "positive (more hydrophobic → better fit)",
            },
            {
                "name": "brain_penetration_score",
                "computation": """
                    BACE1 is in the brain — drug MUST cross BBB.
                    Compute: CNS MPO score (Wager et al. 2010):
                    weighted sum of LogP, LogD, MW, TPSA, HBD, pKa
                    Score 4-6 = good CNS penetration.
                """,
                "biological_rationale": "Must reach brain target",
                "expected_direction": "optimal range (not linear)",
            },
            {
                "name": "selectivity_vs_BACE2",
                "computation": """
                    BACE2 is a close homolog. Off-target BACE2 inhibition
                    causes skin depigmentation (seen in clinical trials).
                    Compute: molecular features that distinguish BACE1 vs
                    BACE2 binders (specific substructure patterns from SAR).
                    
                    Key difference: BACE1 has Thr232 at the flap,
                    BACE2 has Ala. Bulky groups near this position
                    should discriminate.
                """,
                "biological_rationale": "Safety — avoid BACE2 side effects",
                "expected_direction": "selectivity features present → safer",
            },
            {
                "name": "herg_liability",
                "computation": """
                    Many BACE1 inhibitors are basic amines → hERG risk.
                    Compute: basic nitrogen count, pKa of most basic center,
                    presence of difluorophenyl (known hERG pharmacophore).
                """,
                "biological_rationale": "Safety — avoid cardiac toxicity",
                "expected_direction": "negative (hERG features → toxic)",
            },
        ],
        
        confounds=["MW", "NumHeavyAtoms", "total_charge"],
        
        zombie_shortcuts=[
            "Molecular weight (bigger molecules tend to be more potent in assays)",
            "LogP alone (hydrophobic molecules stick to everything)",
            "Aromatic ring count (correlates with potency but not mechanism)",
            "Training set frequency (common scaffolds score higher)",
        ],
        
        discovery_strategy="""
        1. Train GNN on BACE1 activity data (active/inactive classification)
        2. Probe: does GNN encode the 5 mechanism features above?
        3. If NON-ZOMBIE for mechanism features:
           a. Use GNN to screen large virtual library (ZINC20, Enamine REAL)
           b. Filter: top 1% predicted active
           c. SECOND FILTER (novel): extract GNN embeddings for hits,
              verify mechanism features are in the "correct encoding zone"
              (embedding region where mechanism features have high ΔR²)
           d. This second filter catches molecules that score high for
              wrong reasons (zombie predictions within a non-zombie model)
        4. If ZOMBIE for mechanism features:
           a. Do NOT trust this model for BACE1 screening
           b. Try: 3D-equivariant GNN (SchNet/DimeNet) with AlphaFold
              BACE1 structure as input
           c. Re-probe: does 3D model encode pocket complementarity?
        """,
    ),
    
    DiseaseProbeSpec(
        disease="Alzheimer's",
        target_name="tau_aggregation_inhibition",
        target_description="""
        Tau protein hyperphosphorylation and aggregation forms 
        neurofibrillary tangles — the other hallmark of AD.
        Unlike amyloid, tau pathology correlates with cognitive decline.
        Tau aggregation inhibitors are a hot target class.
        """,
        
        datasets=[
            "ChEMBL tau aggregation assays",
            "Published tau ThT fluorescence screening data",
        ],
        dataset_source="ChEMBL + literature",
        n_compounds_expected=2000,
        
        mechanism_features=[
            {
                "name": "beta_sheet_intercalation",
                "computation": """
                    Tau aggregation inhibitors often intercalate between
                    beta-sheet layers. Compute: planarity (fraction of 
                    atoms in aromatic plane), molecular length along
                    principal axis (must fit between tau beta-sheets,
                    ~4.7 Å spacing).
                """,
                "biological_rationale": "Physical mechanism of inhibition",
                "expected_direction": "planar molecules intercalate better",
            },
            {
                "name": "cysteine_reactivity",
                "computation": """
                    Some tau inhibitors work by covalently modifying Cys291
                    and Cys322 in the tau repeat domain.
                    Compute: electrophilic warhead presence (Michael acceptors,
                    acrylamides, vinyl sulfonamides) via SMARTS patterns.
                """,
                "biological_rationale": "Covalent mechanism of action",
                "expected_direction": "warhead present → active (for covalent class)",
            },
            {
                "name": "blood_brain_barrier",
                "computation": "Same CNS MPO score as BACE1 probe",
                "biological_rationale": "Must reach brain",
                "expected_direction": "optimal range 4-6",
            },
        ],
        
        confounds=["MW", "NumHeavyAtoms", "LogP"],
        
        zombie_shortcuts=[
            "Dye-like molecules (ThT assay artifact — fluorescent compounds interfere)",
            "Aggregators (colloidal aggregation gives false positives in biochemical assays)",
            "MW correlation (larger molecules often look more active in aggregation assays)",
        ],
        
        discovery_strategy="""
        1. Critical: filter out PAINS (pan-assay interference compounds)
           and aggregators BEFORE training
        2. Train GNN on filtered tau aggregation data
        3. Probe for beta-sheet intercalation and cysteine reactivity features
        4. The ThT fluorescence assay artifact is a KNOWN zombie source —
           probe specifically for ThT-like features as a zombie indicator
        5. Non-zombie model screens for genuine tau binders
        """,
    ),
    
    DiseaseProbeSpec(
        disease="Alzheimer's",
        target_name="neuroinflammation_modulation",
        target_description="""
        Microglial activation and neuroinflammation are increasingly
        recognized as drivers of AD progression. TREM2, CD33, and
        complement pathway are key targets.
        """,
        
        datasets=[
            "ChEMBL TREM2 modulators",
            "ChEMBL CD33 inhibitors",
            "Anti-inflammatory compound libraries (published screens)",
        ],
        dataset_source="ChEMBL + published screens",
        n_compounds_expected=1500,
        
        mechanism_features=[
            {
                "name": "immune_modulator_pharmacophore",
                "computation": """
                    Anti-inflammatory pharmacophores: presence of
                    sulfonamide, carboxylic acid, or hydroxamic acid
                    (common in COX/LOX/cytokine inhibitors).
                    Compute via SMARTS.
                """,
                "biological_rationale": "Common anti-inflammatory warheads",
                "expected_direction": "positive",
            },
            {
                "name": "microglial_penetration",
                "computation": """
                    Must penetrate BBB AND enter microglia.
                    Compute: CNS MPO + cLogP (moderate lipophilicity
                    for cell membrane penetration).
                """,
                "biological_rationale": "Must reach target cells in brain",
                "expected_direction": "optimal range",
            },
        ],
        
        confounds=["MW", "NumHeavyAtoms"],
        zombie_shortcuts=["General anti-inflammatory features not specific to neuroinflammation"],
        discovery_strategy="Train on neuroinflammation-specific data, probe for target-specific features",
    ),
]


# =================================================================
# CANCER PROBES
# =================================================================

CANCER_PROBES = [
    
    DiseaseProbeSpec(
        disease="Cancer",
        target_name="kinase_selectivity",
        target_description="""
        Kinase inhibitors are the largest class of targeted cancer drugs
        (imatinib, erlotinib, crizotinib, etc.). The key challenge is 
        SELECTIVITY — hitting the target kinase without hitting hundreds
        of off-target kinases that cause toxicity.
        
        A non-zombie model must encode selectivity-determining features,
        not just general kinase binding features.
        """,
        
        datasets=[
            "TDC Kinase Inhibitor dataset",
            "ChEMBL kinase panel data (Davis, Metz, KIBA datasets)",
            "Published kinase selectivity panels (Karaman et al. 2008)",
        ],
        dataset_source="TDC + ChEMBL + KIBA",
        n_compounds_expected=10000,
        
        mechanism_features=[
            {
                "name": "hinge_binder_type",
                "computation": """
                    ALL kinase inhibitors bind the hinge region (connects
                    N-lobe and C-lobe). The hinge binder motif determines 
                    Type I vs Type II binding.
                    
                    Compute: presence of hinge-binding motifs:
                    - Aminopyrimidine (Type I)
                    - Aminopyridine (Type I)
                    - Quinazoline (Type I)
                    - Urea/amide extending to DFG-out pocket (Type II)
                    
                    Use SMARTS pattern matching.
                """,
                "biological_rationale": "Primary binding interaction",
                "expected_direction": "hinge binder present → active",
            },
            {
                "name": "dfg_out_occupancy",
                "computation": """
                    Type II inhibitors occupy the DFG-out pocket 
                    (allosteric pocket behind the ATP site).
                    This gives selectivity because DFG-out conformation
                    varies between kinases.
                    
                    Compute: presence of hydrophobic tail extending 
                    beyond hinge binder (molecular length > 15Å along
                    principal axis, with terminal hydrophobic group).
                """,
                "biological_rationale": "Selectivity-determining interaction",
                "expected_direction": "DFG-out extension → more selective",
            },
            {
                "name": "gatekeeper_complementarity",
                "computation": """
                    The gatekeeper residue (position varies: Thr315 in 
                    ABL, Met790 in EGFR) controls access to hydrophobic
                    pocket behind ATP site. T315I mutation in BCR-ABL
                    causes imatinib resistance.
                    
                    Compute: size of substituent that projects toward
                    gatekeeper position (from crystal structure alignment).
                    Proxy: molecular volume in the "northeast" quadrant.
                """,
                "biological_rationale": "Resistance and selectivity determinant",
                "expected_direction": "depends on target kinase gatekeeper",
            },
            {
                "name": "solvent_front_exposure",
                "computation": """
                    The solvent-exposed front of the binding site varies
                    between kinases and determines selectivity.
                    
                    Compute: polar surface area of the portion of molecule
                    expected to face solvent (from docking pose or 
                    AlphaFold-predicted binding mode).
                """,
                "biological_rationale": "Selectivity from solvent interactions",
                "expected_direction": "target-specific",
            },
            {
                "name": "lipophilic_efficiency",
                "computation": """
                    LipE = pIC50 - LogP. High LipE means potency comes
                    from specific interactions, not just lipophilic binding.
                    Low LipE = the molecule is potent because it's greasy,
                    not because it fits the target well.
                    
                    Compute: LipE directly (requires both potency and LogP).
                    For probe: use LogP as inverse proxy — if model encodes
                    potency INDEPENDENTLY of LogP, it understands specificity.
                """,
                "biological_rationale": "Quality metric separating specific from nonspecific binding",
                "expected_direction": "high LipE → quality drug candidate",
            },
        ],
        
        confounds=["MW", "NumHeavyAtoms", "LogP", "total_aromatic_area"],
        
        zombie_shortcuts=[
            "LogP alone (greasy molecules bind everything)",
            "Molecular size (bigger molecules make more contacts)",
            "Aromatic ring count (flat molecules fit ATP sites generically)",
            "Training set scaffold frequency (common kinase inhibitor scaffolds)",
        ],
        
        discovery_strategy="""
        1. Train GNN on kinase selectivity panel (multi-task: predict 
           activity against 5-10 kinases simultaneously)
        2. Probe: does GNN encode hinge binder type, DFG-out occupancy,
           gatekeeper complementarity, AND lipophilic efficiency?
        3. Key zombie test: probe for LogP encoding. If model predicts
           kinase activity primarily through LogP → ZOMBIE (predicts
           binding from greasiness, not selectivity)
        4. Non-zombie model: screen virtual library, but FILTER by
           requiring high LipE in the embedding space — this selects
           for molecules that are potent for mechanistic reasons
        5. AlphaFold integration: use AF3 predicted binding poses as
           additional ground truth for pocket complementarity probes
        """,
    ),
    
    DiseaseProbeSpec(
        disease="Cancer",
        target_name="immune_checkpoint_modulation",
        target_description="""
        PD-1/PD-L1 and CTLA-4 checkpoint inhibitors have revolutionized
        cancer treatment. Small-molecule PD-L1 inhibitors are an active
        area (vs current antibody therapies). The challenge: distinguishing
        genuine PD-L1 binding from nonspecific protein surface interactions.
        """,
        
        datasets=[
            "ChEMBL PD-L1 small molecule binders",
            "Published BMS/Incyte PD-L1 inhibitor series",
            "Patent data (WO2015034820, WO2017066227)",
        ],
        dataset_source="ChEMBL + patents",
        n_compounds_expected=500,
        
        mechanism_features=[
            {
                "name": "pdl1_dimerization_inducer",
                "computation": """
                    Known small-molecule PD-L1 inhibitors work by inducing
                    PD-L1 dimerization, burying the PD-1 binding face.
                    The BMS compound class (biphenyl core) wedges between 
                    two PD-L1 monomers.
                    
                    Compute: biphenyl/biaryl core presence, molecular 
                    symmetry, and size complementarity with PD-L1 dimer
                    interface (~800 Å² buried surface).
                """,
                "biological_rationale": "The actual mechanism of small-molecule PD-L1 inhibition",
                "expected_direction": "dimerization features → active",
            },
            {
                "name": "protein_surface_complementarity",
                "computation": """
                    PD-L1 surface is relatively flat (protein-protein 
                    interaction). Small molecules must have large, flat 
                    aromatic surfaces to make contacts.
                    
                    Compute: fraction of molecular surface that is flat 
                    aromatic, molecular planarity index.
                """,
                "biological_rationale": "PPI inhibition requires surface matching",
                "expected_direction": "more planar → better surface complementarity",
            },
        ],
        
        confounds=["MW", "LogP", "NumHeavyAtoms"],
        zombie_shortcuts=["Hydrophobic surface area (sticks to any protein surface)"],
        
        discovery_strategy="""
        1. Very small dataset (~500 compounds) — use transfer learning
           from larger kinase dataset, then fine-tune on PD-L1
        2. Probe: does model encode dimerization-inducing features
           or just general protein-surface sticking?
        3. AlphaFold critical here: predict PD-L1 dimer + small molecule
           complex, use as structural ground truth for probing
        4. Non-zombie model screens for genuine dimerization inducers
        """,
    ),

    DiseaseProbeSpec(
        disease="Cancer",
        target_name="synthetic_lethality",
        target_description="""
        Synthetic lethality: two genes where loss of EITHER alone is 
        survivable but loss of BOTH is lethal. PARP inhibitors in 
        BRCA-mutant cancers are the paradigm case. The approach:
        find gene pairs where the cancer has already lost gene A
        (through mutation), then inhibit gene B with a drug.
        
        This is the MOST mechanistically demanding target class because
        the model must understand GENE INTERACTIONS, not just single-target
        binding.
        """,
        
        datasets=[
            "GDSC (Genomics of Drug Sensitivity in Cancer) - 1000 cell lines",
            "DepMap CRISPR knockout data",
            "SynLethDB (synthetic lethal gene pairs)",
            "Published PARP inhibitor data",
        ],
        dataset_source="GDSC + DepMap + ChEMBL",
        n_compounds_expected=5000,
        
        mechanism_features=[
            {
                "name": "dna_damage_response_features",
                "computation": """
                    For PARP-like synthetic lethality, drug must disrupt
                    DNA repair. Compute: presence of DNA-binding motifs
                    (intercalators, minor groove binders, alkylating 
                    warheads) via SMARTS.
                """,
                "biological_rationale": "Mechanism of DNA repair disruption",
                "expected_direction": "DDR features → active in BRCA-mut context",
            },
            {
                "name": "genetic_context_sensitivity",
                "computation": """
                    THIS IS THE KEY PROBE: does the model's prediction 
                    change when you change the GENETIC CONTEXT (cell line
                    mutation profile) vs when you change the MOLECULE?
                    
                    Compute: variance in model embeddings attributable to
                    genetic context vs molecular structure.
                    
                    A non-zombie model should show that embeddings shift
                    with genetic context (BRCA-mut vs BRCA-wt), not just
                    with molecular structure.
                """,
                "biological_rationale": "Synthetic lethality IS context-dependent by definition",
                "expected_direction": "high context sensitivity → understands SL mechanism",
            },
            {
                "name": "selectivity_index",
                "computation": """
                    Ratio of drug sensitivity in genetically vulnerable 
                    cell lines vs resistant cell lines.
                    Compute from GDSC dose-response data.
                    
                    Non-zombie model should encode this INDEPENDENTLY of
                    general cytotoxicity (confound: some molecules kill
                    everything, not just vulnerable lines).
                """,
                "biological_rationale": "Selective killing = genuine SL mechanism",
                "expected_direction": "high selectivity → genuine synthetic lethal interaction",
            },
        ],
        
        confounds=["MW", "LogP", "general_cytotoxicity", "growth_rate_of_cell_line"],
        
        zombie_shortcuts=[
            "General cytotoxicity (kills all cells, not just vulnerable ones)",
            "Cell line growth rate (fast-growing lines appear more sensitive to everything)",
            "Molecular promiscuity (hits many targets, some happen to be in the SL pathway)",
        ],
        
        discovery_strategy="""
        1. Train model on GDSC data (molecule + cell line → sensitivity)
        2. CRITICAL PROBE: genetic_context_sensitivity
           - If model ignores genetic context → ZOMBIE for SL
           - If model uses genetic context → potential non-zombie
        3. Second probe: selectivity_index after confound regression
           (remove general cytotoxicity and growth rate)
        4. Non-zombie model can screen for new SL drug candidates
           by finding molecules where:
           a. Model predicts high sensitivity in target genotype
           b. Model predicts LOW sensitivity in wildtype
           c. The DIFFERENCE is encoded in mechanism features, 
              not confounds
        5. This is the hardest probe — requires multi-modal input
           (molecule structure + cell line genomics)
        """,
    ),
]


# =================================================================
# THE DISCOVERY PIPELINE: FROM VALIDATION TO NEW MOLECULES
# =================================================================

"""
The complete path from DESCARTES-PHARMA validation to new drug discovery:

STEP 1: CHOOSE DISEASE TARGET
  Select from probes above (BACE1, tau, kinase, PD-L1, synthetic lethality)

STEP 2: CURATE TARGET-SPECIFIC DATASET  
  Download from ChEMBL/TDC/GDSC with known activity at target
  Clean: remove PAINS, aggregators, assay artifacts
  Split: scaffold-based (critical — no data leakage)

STEP 3: COMPUTE MECHANISTIC GROUND TRUTH
  For each compound, compute the disease-specific mechanism features
  defined above (pharmacophore patterns, selectivity indices, etc.)

STEP 4: TRAIN MODEL + VALIDATE AS NON-ZOMBIE
  Train GNN (start with GCN baseline, try GAT/SchNet/DimeNet)
  Run full DESCARTES-PHARMA probe stack:
    - Ridge ΔR² + MLP ΔR² for each mechanism feature
    - Scaffold-stratified permutation null
    - Confound regression (remove MW, LogP, size features)
    - FDR correction across all mechanism features
    
  The model MUST show CONFIRMED_ENCODED for at least the primary
  mechanism features (e.g., hinge_binder_type for kinase inhibitors)
  to be trusted for screening.

STEP 5: SCREEN VIRTUAL LIBRARY
  If model is non-zombie for key mechanisms:
  a. Screen large virtual library (ZINC20: 1.4B molecules, 
     Enamine REAL: 6.5B molecules)
  b. Rank by predicted activity
  c. ZOMBIE FILTER (novel contribution of DESCARTES-PHARMA):
     For top hits, extract embeddings and verify they fall in
     the "mechanistically valid" region of embedding space
     (high ΔR² for mechanism features, low ΔR² for confounds)
  d. This catches "accidental hits" — molecules that score high
     through shortcuts the model learned despite being non-zombie overall

STEP 6: VALIDATE COMPUTATIONALLY
  For surviving hits:
  a. AlphaFold 3: predict binding pose with target protein
  b. Probe AF3 binding pose for mechanism-specific features
     (e.g., does pose show hinge binding for kinase target?)
  c. ADMET prediction: check drug-likeness, BBB if CNS target
  d. Selectivity prediction: check against off-target panel

STEP 7: EXPERIMENTAL VALIDATION
  Top 50-100 candidates → synthesize and test in biochemical assay
  This is where DESCARTES-PHARMA's value is measured:
  Hit rate of zombie-filtered candidates vs unfiltered candidates
  
  Expected: 5-10x higher hit rate because zombie filtering removes
  the ~60% of predictions that are confound-driven (as we showed
  in the 3-dataset campaign)
"""


def generate_claude_code_prompt(disease_probe: DiseaseProbeSpec) -> str:
    """
    Generate a Claude Code prompt to run this disease-specific probe.
    """
    
    mechanisms_str = "\n".join([
        f"    - {m['name']}: {m['biological_rationale']}"
        for m in disease_probe.mechanism_features
    ])
    
    confounds_str = ", ".join(disease_probe.confounds)
    
    prompt = f"""Build `scripts/pharma_{disease_probe.target_name}_discovery.py` that runs 
the DESCARTES-PHARMA disease-specific discovery probe for 
{disease_probe.disease} / {disease_probe.target_name}.

1. Load data from {disease_probe.dataset_source} 
   (expect ~{disease_probe.n_compounds_expected} compounds)
   
2. Compute these DISEASE-SPECIFIC mechanism features:
{mechanisms_str}

3. Train GCN (hidden=128, 3 layers). Output gate: AUC >= 0.65.

4. Run full probe stack (Ridge ΔR² + MLP ΔR²) for ALL mechanism features.

5. Run 6-method hardening: scaffold-stratified permutation, y-scramble,
   confound regression (remove {confounds_str}), FDR, TOST, Bayes Factor.

6. Print hardened results + zombie verdict per mechanism.

7. For mechanisms that are CONFIRMED_ENCODED, print:
   "DISCOVERY READY: Model genuinely encodes {{mechanism}} — 
    safe to use for virtual screening on this target."

8. For mechanisms that are ZOMBIE, print:
   "WARNING: Model does NOT encode {{mechanism}} — 
    predictions for this target are not mechanistically grounded."

Reference: DESCARTES_PHARMA_v1.md Section 8 for hardening,
existing scripts for GCN/probe/hardening patterns."""
    
    return prompt


# =================================================================
# PRINT PROBE CATALOG
# =================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DESCARTES-PHARMA: Disease-Specific Discovery Probes")
    print("=" * 70)
    
    all_probes = ALZHEIMER_PROBES + CANCER_PROBES
    
    for probe in all_probes:
        n_mechs = len(probe.mechanism_features)
        n_confounds = len(probe.confounds)
        
        print(f"\n{'─' * 70}")
        print(f"  {probe.disease} / {probe.target_name}")
        print(f"  Data: {probe.dataset_source} (~{probe.n_compounds_expected} compounds)")
        print(f"  Mechanism features to probe: {n_mechs}")
        for m in probe.mechanism_features:
            print(f"    • {m['name']}: {m['biological_rationale']}")
        print(f"  Confounds to remove: {', '.join(probe.confounds)}")
        print(f"  Zombie shortcuts: {probe.zombie_shortcuts[0]}")
    
    print(f"\n{'=' * 70}")
    print(f"Total: {len(all_probes)} disease probes")
    print(f"  Alzheimer's: {len(ALZHEIMER_PROBES)} targets")
    print(f"    (BACE1, tau aggregation, neuroinflammation)")
    print(f"  Cancer: {len(CANCER_PROBES)} targets")
    print(f"    (kinase selectivity, PD-L1, synthetic lethality)")
    print(f"\nTo run any probe, call generate_claude_code_prompt(probe)")
    print(f"and paste the result into Claude Code.")
    print(f"{'=' * 70}")
