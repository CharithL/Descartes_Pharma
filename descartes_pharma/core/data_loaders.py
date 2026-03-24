"""
DESCARTES-PHARMA Data Loaders for Tier 2-4 datasets.

Tier 2: TDC ClinTox, BBBP (pharma benchmarks)
Tier 3: Allen Brain Observatory (neuroscience)
Tier 4: RxRx3-core (phenomics)
"""

import numpy as np


def load_clintox():
    """
    Load ClinTox dataset from Therapeutics Data Commons.
    1,491 drugs with FDA approval status and clinical trial toxicity outcomes.
    """
    from tdc.single_pred import Tox

    data = Tox(name='ClinTox')
    df = data.get_data()

    return {
        'smiles': df['Drug'].values,
        'labels': df['Y'].values,
        'task': 'binary_classification',
        'n_compounds': len(df),
        'mechanism_targets': [
            'reactive_metabolite_alerts',
            'herg_pharmacophore',
            'hepatotoxicity_features',
            'mitochondrial_toxicity',
            'phospholipidosis_risk',
            'DNA_intercalation',
            'oxidative_stress_potential',
        ]
    }


def load_bbbp():
    """
    Load Blood-Brain Barrier Penetration dataset.
    Known mechanistic features: MW < 450, PSA < 90, HBD <= 3, logP 1-3.
    """
    from tdc.single_pred import ADME

    data = ADME(name='BBB_Martins')
    df = data.get_data()

    return {
        'smiles': df['Drug'].values,
        'labels': df['Y'].values,
        'task': 'binary_classification',
        'n_compounds': len(df),
        'mechanism_targets': [
            'molecular_weight',
            'polar_surface_area',
            'hydrogen_bond_donors',
            'hydrogen_bond_acceptors',
            'logP',
            'rotatable_bonds',
            'pgp_substrate_features',
            'tight_junction_permeability',
        ]
    }


def load_tox21():
    """Load MoleculeNet Tox21 -- 12 toxicity endpoints."""
    from tdc.single_pred import Tox

    data = Tox(name='Tox21')
    df = data.get_data()

    return {
        'smiles': df['Drug'].values,
        'labels': df['Y'].values,
        'task': 'binary_classification',
        'n_compounds': len(df),
        'mechanism_targets': [
            'nuclear_receptor_signaling',
            'stress_response_pathways',
            'mitochondrial_membrane_potential',
            'antioxidant_response_element',
        ]
    }


def load_allen_brain_observatory():
    """
    Load visual coding dataset from Allen Brain Observatory.
    Known biological ground truth: orientation selectivity, spatial frequency, etc.
    """
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache

    manifest_path = 'allen_brain_observatory_manifest.json'
    boc = BrainObservatoryCache(manifest_file=manifest_path)

    containers = boc.get_experiment_containers(
        targeted_structures=['VISp'],
        cre_lines=['Cux2-CreERT2']
    )

    return {
        'containers': containers,
        'n_containers': len(containers),
        'mechanism_targets': [
            'orientation_selectivity_index',
            'preferred_orientation',
            'spatial_frequency_preference',
            'temporal_frequency_preference',
            'direction_selectivity_index',
            'on_off_ratio',
            'sustained_transient_ratio',
            'receptive_field_size',
        ],
        'note': 'Use AllenSDK to download actual neural traces per container'
    }


def rxrx3_mechanism_targets():
    """Known biological mechanisms for RxRx3-core genetic knockouts."""
    return {
        'mechanism_targets': [
            'cell_cycle_arrest',
            'apoptosis_markers',
            'er_stress_response',
            'mitotic_defects',
            'cytoskeletal_disruption',
            'lipid_accumulation',
            'dna_damage_response',
            'autophagy_markers',
            'senescence_markers',
            'metabolic_shift',
        ],
        'confound_targets': [
            'plate_position',
            'batch_id',
            'well_edge_distance',
            'cell_density_artifact',
        ]
    }
