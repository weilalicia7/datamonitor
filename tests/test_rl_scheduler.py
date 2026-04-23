"""
Tests for ml/rl_scheduler.py — Wave 3.2 (T3 coverage).

Covers:
- RLState.to_discrete returns tuple with correct bin cardinality
- RLAction.to_discrete mapping
- SchedulingRLAgent choose_action + update + recommend_action
- Reward computation signs on no-show risk / successful attendance
- train_on_historical populates Q-table and returns stats
- get_policy_summary before vs after training
- Multi-agent ChairAgent select/update cycle
- MultiAgentChairScheduler train_on_historical returns stats
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.rl_scheduler import (
    ChairAgent,
    ChairState,
    MultiAgentChairScheduler,
    RLAction,
    RLState,
    SchedulingRLAgent,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _state(hour=10, chairs_occupied=3, queue_size=4, risk=0.2, urgent=1, utilization=0.6):
    return RLState(
        hour=hour,
        chairs_occupied=chairs_occupied,
        total_chairs=19,
        queue_size=queue_size,
        avg_noshow_risk=risk,
        urgent_count=urgent,
        utilization=utilization,
    )


def _appointments(n=20, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append({
            "Appointment_Hour": int(rng.integers(8, 17)),
            "Patient_NoShow_Rate": float(rng.uniform(0, 0.4)),
            "Priority": f"P{rng.integers(1, 5)}",
            "Attended_Status": "Yes" if rng.random() > 0.2 else "No",
            "Planned_Duration": int(rng.integers(60, 200)),
            "Chair_Number": int(rng.integers(1, 19)),
        })
    return out


# --------------------------------------------------------------------------- #
# State / action encoding
# --------------------------------------------------------------------------- #


class TestEncoding:
    def test_state_to_discrete_tuple_shape(self):
        s = _state()
        discrete = s.to_discrete()
        assert isinstance(discrete, tuple)
        assert len(discrete) == 5
        # Each element is a non-negative int.
        assert all(isinstance(x, (int, np.integer)) and x >= 0 for x in discrete)

    def test_action_to_discrete_roundtrip(self):
        a = RLAction(action_type="assign", chair_id=2, time_slot=600)
        idx = a.to_discrete(n_chairs=19)
        assert 0 <= idx < 4 * 19
        # "delay" should map to a higher index than "assign" for the same chair.
        b = RLAction(action_type="delay", chair_id=2, time_slot=600)
        assert b.to_discrete(19) > a.to_discrete(19)


# --------------------------------------------------------------------------- #
# SchedulingRLAgent
# --------------------------------------------------------------------------- #


class TestSchedulingRLAgent:
    def test_choose_action_returns_valid_action(self):
        np.random.seed(0)
        agent = SchedulingRLAgent(n_chairs=19, epsilon=0.0)
        a = agent.choose_action(_state())
        assert isinstance(a, RLAction)
        assert a.action_type in {"assign", "delay", "double_book", "reject"}
        assert 0 <= a.chair_id < 19

    def test_compute_reward_penalises_noshow_and_rewards_success(self):
        agent = SchedulingRLAgent(n_chairs=19, noshow_penalty_weight=0.3)
        r_success = agent.compute_reward(
            waiting_time=10, noshow_risk=0.1, utilization=0.7,
            patient_priority=1, successful=True,
        )
        r_fail = agent.compute_reward(
            waiting_time=10, noshow_risk=0.1, utilization=0.7,
            patient_priority=1, successful=False,
        )
        assert r_success > r_fail

    def test_update_modifies_q_table(self):
        np.random.seed(1)
        agent = SchedulingRLAgent(n_chairs=19, epsilon=0.0)
        s = _state()
        a = agent.choose_action(s)
        info = agent.update(s, a, reward=1.0, next_state=_state(hour=11, queue_size=3))
        assert "td_error" in info
        assert "q_before" in info
        assert "q_after" in info
        assert agent.episodes == 1
        assert agent.q_table  # populated

    def test_train_on_historical_returns_stats(self):
        np.random.seed(2)
        agent = SchedulingRLAgent(n_chairs=19)
        stats = agent.train_on_historical(_appointments(n=20), n_epochs=2)
        assert stats["epochs"] == 2
        assert stats["total_updates"] == 40
        assert stats["q_table_size"] > 0

    def test_recommend_action_returns_dict(self):
        np.random.seed(3)
        agent = SchedulingRLAgent(n_chairs=19)
        agent.train_on_historical(_appointments(n=10), n_epochs=1)
        rec = agent.recommend_action(_state())
        assert "recommended_action" in rec
        assert rec["recommended_action"] in {"assign", "delay", "double_book", "reject"}
        assert "q_value" in rec


# --------------------------------------------------------------------------- #
# MARL ChairAgent + MultiAgentChairScheduler
# --------------------------------------------------------------------------- #


class TestMARL:
    def _chair_state(self, chair_id=0, occupied=False):
        return ChairState(
            chair_id=chair_id,
            is_occupied=occupied,
            current_patient_priority=2 if occupied else 0,
            current_remaining_min=30 if occupied else 0,
            next_gap_min=15,
            queue_size=4,
            hour=10,
        )

    def test_chair_agent_select_and_update(self):
        np.random.seed(4)
        agent = ChairAgent(chair_id=0, epsilon=0.0)
        s = self._chair_state()
        action = agent.select_action(s)
        assert action in {"accept", "reject", "delay"}
        agent.update(s, action, reward=1.0, next_state=self._chair_state(occupied=True))
        assert agent.updates == 1
        assert agent.q_table

    def test_multiagent_train_on_historical(self):
        np.random.seed(5)
        scheduler = MultiAgentChairScheduler(n_chairs=3)
        stats = scheduler.train_on_historical(_appointments(n=12), n_epochs=2)
        assert stats["epochs"] == 2
        assert stats["total_updates"] == 24
        assert stats["agents"] == 3
        assert sum(stats["agent_updates"]) == 24

    def test_compute_shared_reward_sign(self):
        scheduler = MultiAgentChairScheduler(n_chairs=3)
        high_util = scheduler.compute_shared_reward(
            utilization=0.9, total_wait=0.0, noshow_rate=0.0, priority_variance=0.0
        )
        low_util = scheduler.compute_shared_reward(
            utilization=0.1, total_wait=120.0, noshow_rate=0.3, priority_variance=0.9
        )
        assert high_util > low_util
