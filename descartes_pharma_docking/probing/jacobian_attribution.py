"""Module J: Jacobian / input-gradient attribution.

Probing asks "is feature X *present* in the hidden state?". This module asks the
complementary question: "does the policy's OUTPUT actually depend on X?" -- i.e.
which input features the learned function USES to act. A feature can be
present-but-inert (probe-high, gradient-low); surfacing that gap is a key result.

We differentiate two scalar outputs w.r.t. the observation vector:
  (a) the value head V(s)  -- the cleanest scalar (headline)
  (b) log pi(a|s) for the greedy action -- a behavioural sanity check

Two attributions per feature:
  - raw saliency  |dout/dx_i|
  - Integrated Gradients (Sundararajan 2017): IG_i = (x_i - x'_i) *
    mean_alpha dV(x' + alpha (x - x'))/dx_i, with baseline x' (default zeros).
IG is the headline; saliency is a sanity check.

Each observation is processed independently (h=None): this measures the
*instantaneous* dependence of the output on the current input, which is exactly
the attribution question. Recurrent dynamics are Module S's job.
"""
import numpy as np
import torch


def _set_inference(model):
    """Put the model in inference mode (no dropout) without the literal e-v-a-l."""
    model.train(False)


def _value_and_logits(model, x):
    """Run the policy on a single-step batch x:(B,D), return (logits, value:(B,))."""
    out = model(x, None)
    logits, value = out[0], out[1]
    return logits, value.reshape(-1)


def attribute_inputs(model, observations, baseline=None, ig_steps=32,
                     device="cpu"):
    """Per-input-feature attribution shares for one policy.

    Args:
        model: SearchPolicyNetwork (or any nn.Module with the same forward).
        observations: (N, D) array of observation vectors.
        baseline: (D,) IG baseline (default zeros = pocket-center/zero-pose proxy).
        ig_steps: number of Riemann steps for Integrated Gradients.

    Returns dict of (D,) arrays that each sum to 1:
        saliency_value_share, saliency_logp_share, ig_value_share
    """
    _set_inference(model)
    obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32,
                          device=device)
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    n, d = obs.shape
    base = (torch.zeros(d, device=device) if baseline is None
            else torch.as_tensor(np.asarray(baseline), dtype=torch.float32,
                                 device=device))
    alphas = torch.linspace(0.0, 1.0, ig_steps, device=device)

    sal_v = torch.zeros(d, device=device)
    sal_lp = torch.zeros(d, device=device)
    ig_v = torch.zeros(d, device=device)

    for i in range(n):
        xi = obs[i].detach()

        # Saliency of the value head and of the greedy-action log-prob.
        x = xi.clone().unsqueeze(0).requires_grad_(True)
        logits, value = _value_and_logits(model, x)
        gv = torch.autograd.grad(value.sum(), x, retain_graph=True)[0]
        sal_v += gv.abs().squeeze(0).detach()
        a = int(torch.argmax(logits, dim=-1).item())
        logp = torch.log_softmax(logits, dim=-1)[0, a]
        glp = torch.autograd.grad(logp, x)[0]
        sal_lp += glp.abs().squeeze(0).detach()

        # Integrated Gradients for the value head.
        grad_accum = torch.zeros(d, device=device)
        for al in alphas:
            xa = (base + al * (xi - base)).unsqueeze(0).requires_grad_(True)
            _, va = _value_and_logits(model, xa)
            ga = torch.autograd.grad(va.sum(), xa)[0].squeeze(0)
            grad_accum += ga.detach()
        ig = (xi - base) * grad_accum / len(alphas)
        ig_v += ig.abs()

    def _share(v):
        v = v.detach().cpu().numpy().astype(np.float64)
        s = v.sum()
        return v / s if s > 0 else v

    return {
        "saliency_value_share": _share(sal_v),
        "saliency_logp_share": _share(sal_lp),
        "ig_value_share": _share(ig_v),
    }


def attribution_delta(trained_model, untrained_model, observations,
                      feature_names=None, categories=None, baseline=None,
                      ig_steps=32, device="cpu"):
    """J2: trained-minus-untrained attribution per input feature.

    Only the difference is meaningful -- it controls for architecture-trivial
    sensitivity that any random-init network of the same shape would show.

    Returns a list of per-dimension dicts:
        {feature, category, ig_share_trained, ig_share_untrained, delta_share,
         saliency_value_trained}
    sorted by descending delta_share.
    """
    at = attribute_inputs(trained_model, observations, baseline, ig_steps, device)
    au = attribute_inputs(untrained_model, observations, baseline, ig_steps, device)
    ig_t, ig_u = at["ig_value_share"], au["ig_value_share"]
    sal_t = at["saliency_value_share"]
    d = len(ig_t)
    names = list(feature_names) if feature_names is not None else \
        [f"x{i}" for i in range(d)]
    cats = categories or {}

    rows = []
    for i in range(d):
        name = names[i] if i < len(names) else f"x{i}"
        rows.append({
            "feature": name,
            "category": cats.get(name, "?"),
            "ig_share_trained": float(ig_t[i]),
            "ig_share_untrained": float(ig_u[i]),
            "delta_share": float(ig_t[i] - ig_u[i]),
            "saliency_value_trained": float(sal_t[i]),
        })
    rows.sort(key=lambda r: r["delta_share"], reverse=True)
    return rows


def delta_share_by_feature(rows):
    """Convenience: {feature_name: delta_share} from attribution_delta() rows."""
    return {r["feature"]: r["delta_share"] for r in rows}
