"""Multi-step structural MPC (Phase 14).

Model Predictive Control (MPC) for structural intelligence: at each step,
the runtime plans H steps ahead but only executes the first action
(receding horizon). This enables:

  - planning for delayed rewards (e.g. a sequence of mutations that only
    improves utility after 3 steps)
  - avoiding actions that look good locally but lead to bad states
  - trading off exploration vs exploitation over a horizon

The MPC planner uses bounded branching to keep the search tractable:
  - at each step, consider at most B candidate actions
  - total sequences: B^H (bounded by max_sequences)
  - evaluate each sequence with a utility function
  - select the first action of the best sequence

The planner never mutates authoritative state; it uses shadow transactions
to evaluate counterfactuals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .candidates import CandidateUnion


@dataclass(frozen=True, slots=True)
class MPCPlan:
    """A multi-step plan from MPC."""
    actions: list[str]  # candidate IDs for each step
    expected_utilities: list[float]  # utility at each step
    total_utility: float
    horizon: int

    @property
    def first_action(self) -> str | None:
        return self.actions[0] if self.actions else None

    def to_log(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "expected_utilities": [float(u) for u in self.expected_utilities],
            "total_utility": float(self.total_utility),
            "horizon": int(self.horizon),
            "first_action": self.first_action,
        }


@dataclass(slots=True)
class MPCPlanner:
    """Multi-step structural MPC planner with bounded branching.

    The planner explores action sequences up to ``horizon`` steps, with at
    most ``max_branching`` candidates per step and ``max_sequences`` total
    sequences. It uses a utility function to evaluate each sequence.
    """
    horizon: int = 3
    max_branching: int = 4
    max_sequences: int = 64
    utility_fn: Callable[[Any, str], float] | None = None  # (state, action_id) -> utility

    def plan(
        self,
        *,
        candidates: Sequence[str],  # candidate IDs for the current state
        simulate_fn: Callable[[Any, str], Any] | None = None,  # (state, action_id) -> next_state
        initial_state: Any = None,
        utility_fn: Callable[[Any, str], float] | None = None,
    ) -> MPCPlan:
        """Plan the best H-step sequence using bounded search.

        ``simulate_fn`` applies an action to a shadow state and returns the
        next state. ``utility_fn`` evaluates (state, action) -> utility.
        """
        u_fn = utility_fn or self.utility_fn
        if u_fn is None:
            raise ValueError("no utility function provided")
        sim_fn = simulate_fn or (lambda s, a: s)  # default: no simulation

        # Bounded branching: take top-B candidates.
        top_candidates = list(candidates)[:self.max_branching]
        if not top_candidates:
            return MPCPlan(actions=[], expected_utilities=[], total_utility=0.0, horizon=0)

        # Enumerate sequences up to max_sequences.
        best_plan = MPCPlan(actions=[], expected_utilities=[], total_utility=float("-inf"), horizon=0)
        sequences_explored = 0

        def _search(state: Any, actions_so_far: list[str], utils_so_far: list[float], depth: int) -> None:
            nonlocal best_plan, sequences_explored
            if sequences_explored >= self.max_sequences:
                return
            if depth >= self.horizon:
                total = sum(utils_so_far)
                if total > best_plan.total_utility:
                    best_plan = MPCPlan(
                        actions=list(actions_so_far),
                        expected_utilities=list(utils_so_far),
                        total_utility=total,
                        horizon=len(actions_so_far),
                    )
                sequences_explored += 1
                return
            # Get candidates for the current state.
            if depth == 0:
                cands = top_candidates
            else:
                # In a real implementation, we'd generate candidates for the
                # simulated state. Here we reuse the same candidates for simplicity.
                cands = top_candidates
            for cand_id in cands[:self.max_branching]:
                if sequences_explored >= self.max_sequences:
                    return
                u = float(u_fn(state, cand_id))
                next_state = sim_fn(state, cand_id)
                actions_so_far.append(cand_id)
                utils_so_far.append(u)
                _search(next_state, actions_so_far, utils_so_far, depth + 1)
                actions_so_far.pop()
                utils_so_far.pop()

        _search(initial_state, [], [], 0)
        return best_plan

    def to_log(self) -> dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "max_branching": int(self.max_branching),
            "max_sequences": int(self.max_sequences),
        }


def plan_with_mpc(
    *,
    candidates: Sequence[str],
    horizon: int = 3,
    max_branching: int = 4,
    max_sequences: int = 64,
    utility_fn: Callable[[Any, str], float],
    simulate_fn: Callable[[Any, str], Any] | None = None,
    initial_state: Any = None,
) -> MPCPlan:
    """Convenience function: plan with MPC in one call."""
    planner = MPCPlanner(
        horizon=horizon, max_branching=max_branching,
        max_sequences=max_sequences, utility_fn=utility_fn,
    )
    return planner.plan(
        candidates=candidates, simulate_fn=simulate_fn,
        initial_state=initial_state,
    )
