"""
DESCARTES-PHARMA v1.2 M12: Full Integrated Meta-Learner Pipeline.

Composes all 7 paradigms into a single orchestrator that sits
ABOVE both C1 and C2 factories.
"""

import torch
from .neural_fast_path import PharmaMetaLearner, PharmaMetaTrainer
from .mechanism_decomposer import MechanismDecomposer
from .verified_zombie_store import VerifiedZombieStore
from .probe_cascade import ProbeCascadeRouter
from .vfe_belief import VFEBeliefSystem
from .multi_timescale import MultiTimescaleProcessor
from .hot_layer import MetaCognitionHOT
from .llm_strategic import LLMStrategicReasoner


class DescartesPharmaMetaLearner:
    """
    The complete hybrid meta-learner integrating all 7 paradigms.

    Neural fast path: PharmaMetaLearner (~2M params, <1ms per decision)
    LLM strategic: LLMStrategicReasoner (triggered by HOT, ~$0.01/call)

    Orchestrates:
    - WHAT to probe (VFE + Global Workspace)
    - WHEN to stop (VFE belief convergence)
    - WHERE to search (Thompson + VZS)
    - WHETHER to escalate (cheap -> expensive probe cascade)
    - WHY it's stuck (HOT layer + LLM strategic reasoning)
    """

    def __init__(self, meta_path=None, vzs_path='verified_zombie_store.json'):
        # Paradigm 1: Neural fast path
        self.neural = PharmaMetaLearner()
        self.trainer = PharmaMetaTrainer(self.neural)

        # Paradigm 2: Mechanism decomposer
        self.decomposer = MechanismDecomposer()

        # Paradigm 3: Verified Zombie Store
        self.vzs = VerifiedZombieStore(store_path=vzs_path)

        # Paradigm 4: Probe cascade router
        self.cascade = ProbeCascadeRouter()

        # Paradigm 5: VFE belief system
        self.vfe = VFEBeliefSystem()

        # Paradigm 6: Multi-timescale processor
        self.timescale = MultiTimescaleProcessor(
            self.neural, self.vfe, self.vzs, self.cascade)

        # Paradigm 7: Meta-cognition HOT layer
        self.hot = MetaCognitionHOT()

        # LLM strategic reasoner
        self.llm = LLMStrategicReasoner()

        # Load saved state if available
        if meta_path:
            self.trainer.load(meta_path)

    def evaluate_model(self, model, mechanisms, dataset,
                       hidden_states, mechanism_features):
        """
        Full meta-learned evaluation of one model.

        Instead of blindly running all 43 probes on all mechanisms,
        the meta-learner decides what to probe, in what order,
        and when to stop -- potentially saving 40-60% compute.
        """
        routing_plan = self.decomposer.decompose(mechanisms)
        all_results = {}

        for mechanism, plan in routing_plan.items():
            # Check VZS first (free!)
            arch_name = getattr(model, 'architecture', 'unknown')
            vzs_result = self.vzs.lookup(arch_name, mechanism, dataset)

            if vzs_result['status'] == 'HIT' and vzs_result.get('tier', 99) <= 2:
                all_results[mechanism] = {
                    'verdict': vzs_result['verdict'],
                    'source': 'VZS_CACHE',
                    'tier': vzs_result.get('tier'),
                }
                continue

            # Run probe cascade with meta-learned routing
            cascade_result = self.cascade.run_cascade(
                model, mechanism, dataset, hidden_states,
                mechanism_features, self.neural, self.vzs)

            all_results[mechanism] = cascade_result

            # Fast tick for each probe result
            for probe_name, probe_result in cascade_result.get('evidence', {}).items():
                if isinstance(probe_result, dict) and 'delta_r2' in probe_result:
                    self.timescale.fast_tick(
                        probe_name, arch_name, mechanism,
                        probe_result['delta_r2'],
                        probe_result.get('p_value', 1.0),
                        probe_result.get('compute_seconds', 0))

        # Medium tick
        arch_name = getattr(model, 'architecture', 'unknown')
        model_verdict = self.timescale.medium_tick(
            arch_name, all_results, dataset)

        return model_verdict

    def end_of_campaign(self, campaign_results):
        """Slow tick: cross-campaign learning."""
        slow_result = self.timescale.slow_tick(campaign_results)

        hot_result = self.hot.assess(campaign_results, {
            'coverage_fraction': len(campaign_results) / 200,
            'rounds_remaining': 0,
        })

        if hot_result['recommended_action'] == 'TRIGGER_LLM_STRATEGIC_REASONING':
            llm_result = self.llm.reason(
                factory_state={'n_models': len(campaign_results)},
                hot_diagnosis=hot_result['diagnosis'],
                vzs=self.vzs,
                vfe_system=self.vfe,
                campaign_history=campaign_results)

            return {
                'slow_tick': slow_result,
                'hot': hot_result,
                'llm_strategy': llm_result,
            }

        return {'slow_tick': slow_result, 'hot': hot_result}

    def save(self, path):
        self.trainer.save(f"{path}_neural.pt")
        self.vzs._save()

    def get_stats(self):
        return {
            'neural_updates': self.trainer.update_count,
            'vzs': self.vzs.get_stats(),
            'vfe_beliefs': len(self.vfe.beliefs),
            'hot_productive_prob': self.hot.meta_belief['productive_prob'],
            'avg_meta_loss': (
                sum(self.trainer.loss_history) /
                max(len(self.trainer.loss_history), 1)
            ) if self.trainer.loss_history else None,
        }
