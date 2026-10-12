"""FedAvg aggregation with a per-client beta gate (beta=0 excludes a client)."""
import torch


def aggregate(global_state: dict, client_states: list, betas: list, eta: float,
               denominator: str = "survivors") -> dict:
    """denominator:
    - "survivors" (default): mean divides by the number of non-excluded clients --
      a proper renormalized FedAvg over whoever survived filtering this round.
    - "sampled": mean divides by the total number of sampled clients regardless of
      exclusions (reference-code behaviour, CODE_NOTES.md discrepancy #7a) --
      excluded clients' zeroed diffs still count toward the denominator, so
      filtering shrinks the effective step size. Kept as an ablation; not the
      default.
    """
    keys = [k for k in global_state.keys() if global_state[k].dtype != torch.int64]
    active = [i for i, b in enumerate(betas) if b != 0]
    if not active:
        return {k: v.clone() for k, v in global_state.items()}

    if denominator == "sampled":
        divisor = len(betas)
    elif denominator == "survivors":
        divisor = len(active)
    else:
        raise ValueError(f"unknown aggregation denominator mode: {denominator!r}")

    new_state = {k: v.clone() for k, v in global_state.items()}
    with torch.no_grad():
        for k in keys:
            diffs = torch.stack([
                (client_states[i][k] - global_state[k]) * betas[i] for i in active
            ])
            new_state[k] = global_state[k] + eta * (diffs.sum(dim=0) / divisor)
    return new_state
