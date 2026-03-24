"""
DESCARTES-PHARMA v1.2 M11: LLM Strategic Reasoning Layer.

Triggered by HOT layer when meta-cognition detects problems.
Provides strategic reasoning about the search process itself.
"""

import json

SYSTEM_META_STRATEGIC = """You are the strategic reasoning layer of the \
DESCARTES-PHARMA dual factory meta-learner. The HOT (Higher-Order Thought) \
layer has detected an issue with the search process and is consulting you \
for strategic guidance.

Current factory state:
{factory_state}

HOT layer diagnosis:
{hot_diagnosis}

VZS statistics:
{vzs_stats}

VFE belief summary:
{vfe_summary}

Campaign history:
{campaign_history}

Your role is NOT to suggest specific probes or architectures (the neural
fast path handles that). Your role is STRATEGIC:

1. Is the search process stuck for a FUNDAMENTAL reason?
2. Should the factory PIVOT to a different search strategy?
3. Are there CROSS-CAMPAIGN patterns the DreamCoder missed?
4. Should this campaign be TERMINATED early?

Respond with ONLY valid JSON:
{{
  "diagnosis": "one-sentence summary of what's wrong",
  "root_cause": "STUCK | WRONG_APPROACH | MISSING_DATA | EXHAUSTED | ANOMALY",
  "recommended_pivot": "specific strategic change",
  "confidence": 0.0-1.0,
  "should_terminate": true/false,
  "cross_campaign_insight": "pattern discovered or null"
}}"""


class LLMStrategicReasoner:
    """
    LLM strategic reasoning, triggered by HOT layer.

    Key distinction from LLM Balloon (v1.1):
    - Balloon proposes NOVEL ARCHITECTURES AND PROBES
    - Strategic Reasoner diagnoses WHY THE SEARCH IS FAILING
    """

    def __init__(self, model='claude-sonnet-4-20250514'):
        self.model = model
        self.reasoning_history = []

    def reason(self, factory_state, hot_diagnosis, vzs, vfe_system,
               campaign_history):
        """
        Run LLM strategic reasoning.

        Requires ANTHROPIC_API_KEY environment variable.
        Falls back to heuristic reasoning if API unavailable.
        """
        try:
            import os
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                return self._heuristic_reasoning(hot_diagnosis)

            import requests
            prompt = SYSTEM_META_STRATEGIC.format(
                factory_state=json.dumps(factory_state, indent=2, default=str),
                hot_diagnosis=json.dumps(hot_diagnosis, indent=2, default=str),
                vzs_stats=json.dumps(vzs.get_stats(), indent=2),
                vfe_summary=self._summarize_vfe(vfe_system),
                campaign_history=self._summarize_campaigns(campaign_history),
            )

            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                },
                json={
                    'model': self.model,
                    'max_tokens': 1000,
                    'messages': [{'role': 'user', 'content': prompt}]
                }
            )

            data = response.json()
            text = data['content'][0]['text']

            reasoning = json.loads(text)
            self.reasoning_history.append(reasoning)
            return reasoning

        except Exception as e:
            print(f"LLM strategic reasoning error: {e}")
            return self._heuristic_reasoning(hot_diagnosis)

    def _heuristic_reasoning(self, hot_diagnosis):
        """Fallback when LLM API unavailable."""
        diagnosis = hot_diagnosis if isinstance(hot_diagnosis, dict) else {}

        if diagnosis.get('stagnation', {}).get('detected'):
            return {
                'diagnosis': 'Search stagnating without improvement',
                'root_cause': 'STUCK',
                'recommended_pivot': 'Try different architecture families or loss functions',
                'confidence': 0.5,
                'should_terminate': False,
                'cross_campaign_insight': None,
            }

        return {
            'diagnosis': 'No clear issue detected by heuristics',
            'root_cause': 'UNKNOWN',
            'recommended_pivot': 'Continue normal search',
            'confidence': 0.3,
            'should_terminate': False,
            'cross_campaign_insight': None,
        }

    def _summarize_vfe(self, vfe_system):
        summary = {}
        for key, belief in vfe_system.beliefs.items():
            summary[key] = {
                'zombie_prob': round(belief['mean'], 3),
                'uncertainty': round(belief['variance'], 3),
                'n_updates': belief['n_updates'],
            }
        return json.dumps(summary, indent=2)

    def _summarize_campaigns(self, history):
        if not history:
            return "No previous campaigns"
        if isinstance(history, list):
            return json.dumps([{
                'dataset': h.get('dataset') if isinstance(h, dict) else str(h),
            } for h in history[-5:]], indent=2)
        return str(history)
