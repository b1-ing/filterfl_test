"""SiftFL addition #2: persistent per-client trust score.

Best-effort implementation — the SiftFL paper itself wasn't available to read
(IEEE Xplore, ~Aug 2026, gated). Every behavioural choice below is an explicit
assumption; see results/siftfl_assumptions.md for the full list and why each
one was made this way.
"""
from dataclasses import dataclass, field


@dataclass
class TrustState:
    accumulation_rate: float
    decay_rate: float
    exclusion_threshold: float
    initial_trust: float = 0.0
    scores: dict = field(default_factory=dict)

    def get(self, client_id: int) -> float:
        return self.scores.get(client_id, self.initial_trust)

    def update(self, client_id: int, flagged_this_round: bool):
        """ASSUMPTION: trust moves by a fixed additive step per round a client
        is sampled, decaying on a flag and accumulating on a clean round.
        Not clamped (a client can accumulate trust indefinitely) — this is an
        assumption too; a bounded/EMA formulation is an equally plausible
        reading of "trust history" and is flagged as an alternative in
        results/siftfl_assumptions.md.
        """
        cur = self.get(client_id)
        if flagged_this_round:
            cur -= self.decay_rate
        else:
            cur += self.accumulation_rate
        self.scores[client_id] = cur

    def is_excluded_by_trust(self, client_id: int) -> bool:
        return self.get(client_id) < self.exclusion_threshold
