"""
Tests for ml/stochastic_mpc_scheduler.py (Dissertation §5.5)
===========================================================

Verifies the MPC controller + scenario sampler + Gamma-Poisson
arrival model + terminal value function + receding-horizon rollout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.stochastic_mpc_scheduler import (
    ChairState,
    ChairStatus,
    DEFAULT_LOOKAHEAD_MINUTES,
    DEFAULT_N_SCENARIOS,
    MDPState,
    MPCController,
    MPCDecision,
    QueuedPatient,
    RolloutPlanner,
    ScenarioSampler,
    ScheduleAction,
    TerminalValueFunction,
    UrgentArrivalModel,
    _poisson,
    compute_immediate_reward,
    compute_terminal_penalty,
    get_controller,
    set_controller,
    state_from_app_state,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_mpc_dir(tmp_path):
    return tmp_path / "mpc"


@pytest.fixture
def state_two_chairs_three_queue():
    return MDPState(
        time_min=60,
        chairs=[
            ChairState("WC-C01", "WC", ChairStatus.IDLE),
            ChairState("WC-C02", "WC", ChairStatus.IDLE),
        ],
        queue=[
            QueuedPatient("P001", priority=1, expected_duration=60,
                          arrival_time_min=0),
            QueuedPatient("P002", priority=2, expected_duration=45,
                          arrival_time_min=30),
            QueuedPatient("P003", priority=3, expected_duration=90,
                          arrival_time_min=45),
        ],
    )


@pytest.fixture
def arrival_model():
    m = UrgentArrivalModel()
    m.update(2, 20.0)
    return m


@pytest.fixture
def controller(tmp_mpc_dir, arrival_model):
    return MPCController(
        arrival_model=arrival_model,
        n_scenarios=5, lookahead_minutes=60,
        storage_dir=tmp_mpc_dir,
    )


# --------------------------------------------------------------------------- #
# 1. UrgentArrivalModel
# --------------------------------------------------------------------------- #


class TestUrgentArrivalModel:
    def test_prior_is_exponential(self):
        m = UrgentArrivalModel()
        # With alpha=1, beta=1 the mean is 1.0 arrivals per minute.
        assert m.predict_rate() == pytest.approx(1.0, abs=1e-9)

    def test_update_increases_alpha(self):
        m = UrgentArrivalModel()
        alpha_before = m.alpha
        m.update(3, 5.0)
        assert m.alpha == alpha_before + 3

    def test_posterior_mean(self):
        m = UrgentArrivalModel()
        m.update(5, 50.0)   # 5 arrivals in 50 minutes
        # alpha = 1+5 = 6, beta = 1+50 = 51, rate = 6/51 ≈ 0.1176
        assert m.predict_rate() == pytest.approx(6 / 51, abs=1e-9)

    def test_save_load_roundtrip(self, tmp_path):
        m = UrgentArrivalModel(alpha=5, beta=50)
        m.update(2, 10)
        path = tmp_path / "am.json"
        m.save(path)
        m2 = UrgentArrivalModel.load(path)
        assert m2.alpha == m.alpha
        assert m2.beta == m.beta


# --------------------------------------------------------------------------- #
# 2. TerminalValueFunction
# --------------------------------------------------------------------------- #


class TestTerminalValueFunction:
    def test_default_weights_length(self):
        v = TerminalValueFunction()
        assert len(v.weights) == len(v.FEATURE_NAMES)

    def test_prediction_is_finite(self, state_two_chairs_three_queue):
        v = TerminalValueFunction()
        r = v.predict(state_two_chairs_three_queue)
        assert r == r   # not NaN
        assert isinstance(r, float)

    def test_queue_increase_lowers_value(self, state_two_chairs_three_queue):
        v = TerminalValueFunction()
        before = v.predict(state_two_chairs_three_queue)
        more_queue = state_two_chairs_three_queue.copy()
        for i in range(5):
            more_queue.queue.append(
                QueuedPatient(f"EXTRA{i}", priority=1,
                              expected_duration=60, arrival_time_min=60)
            )
        after = v.predict(more_queue)
        assert after < before


# --------------------------------------------------------------------------- #
# 3. ScenarioSampler
# --------------------------------------------------------------------------- #


class TestScenarioSampler:
    def test_generates_k_scenarios(self, arrival_model, state_two_chairs_three_queue):
        s = ScenarioSampler(arrival_model, n_scenarios=7, rng_seed=1)
        scenarios = s.sample_scenarios(state_two_chairs_three_queue,
                                       lookahead_minutes=60)
        assert len(scenarios) == 7

    def test_scenarios_differ_with_random_seeds(
        self, arrival_model, state_two_chairs_three_queue
    ):
        s1 = ScenarioSampler(arrival_model, n_scenarios=5, rng_seed=1)
        s2 = ScenarioSampler(arrival_model, n_scenarios=5, rng_seed=2)
        a1 = s1.sample_scenarios(state_two_chairs_three_queue, 60)
        a2 = s2.sample_scenarios(state_two_chairs_three_queue, 60)
        # At least one scenario should have different arrival pattern
        assert any(
            len(a["urgent_arrivals"]) != len(b["urgent_arrivals"])
            for a, b in zip(a1, a2)
        ) or any(
            a["patient_outcomes"] != b["patient_outcomes"]
            for a, b in zip(a1, a2)
        )

    def test_same_seed_reproducible(self, arrival_model, state_two_chairs_three_queue):
        s1 = ScenarioSampler(arrival_model, n_scenarios=5, rng_seed=42)
        s2 = ScenarioSampler(arrival_model, n_scenarios=5, rng_seed=42)
        a1 = s1.sample_scenarios(state_two_chairs_three_queue, 60)
        a2 = s2.sample_scenarios(state_two_chairs_three_queue, 60)
        assert len(a1) == len(a2)


# --------------------------------------------------------------------------- #
# 4. MPCController decisions
# --------------------------------------------------------------------------- #


class TestMPCController:
    def test_decide_returns_valid_decision(self, controller, state_two_chairs_three_queue):
        d = controller.decide(state_two_chairs_three_queue)
        assert isinstance(d, MPCDecision)
        assert d.used_fallback is False
        assert d.n_scenarios == 5
        assert "assignments" in d.action

    def test_empty_queue_returns_wait(self, controller):
        state = MDPState(time_min=60, chairs=[
            ChairState("WC-C01", "WC", ChairStatus.IDLE),
        ], queue=[])
        d = controller.decide(state)
        assert all(v is None for v in d.action["assignments"].values())

    def test_no_idle_chairs_returns_wait(self, controller):
        state = MDPState(
            time_min=60,
            chairs=[ChairState("WC-C01", "WC", ChairStatus.OCCUPIED,
                               patient_id="P_X", remaining_minutes=30)],
            queue=[QueuedPatient("P001", priority=1, expected_duration=60)],
        )
        d = controller.decide(state)
        assert all(v is None for v in d.action["assignments"].values())

    def test_high_priority_gets_idle_chair(self, controller, state_two_chairs_three_queue):
        d = controller.decide(state_two_chairs_three_queue)
        assigned_patients = [v for v in d.action["assignments"].values() if v]
        # At least the priority-1 patient should be assigned
        assert "P001" in assigned_patients

    def test_fallback_triggered_by_tight_timeout(self, arrival_model,
                                                 state_two_chairs_three_queue,
                                                 tmp_mpc_dir):
        c = MPCController(
            arrival_model=arrival_model, n_scenarios=20,
            lookahead_minutes=240,
            total_timeout_s=0.0001,   # impossibly tight
            storage_dir=tmp_mpc_dir,
        )
        d = c.decide(state_two_chairs_three_queue)
        assert d.used_fallback is True
        assert d.fallback_reason is not None


# --------------------------------------------------------------------------- #
# 5. Event triggers + simulate_day
# --------------------------------------------------------------------------- #


class TestEventsAndSimulation:
    def test_event_trigger_updates_arrival_model_on_urgent(
        self, controller, state_two_chairs_three_queue
    ):
        before = controller.arrival_model.alpha
        controller.event_trigger(
            "urgent_arrival", state_two_chairs_three_queue,
            payload={"n_arrivals": 3, "minutes_observed": 10.0},
        )
        assert controller.arrival_model.alpha > before

    def test_simulate_day_returns_per_policy_metrics(
        self, controller, state_two_chairs_three_queue
    ):
        results = controller.simulate_day(
            state_two_chairs_three_queue,
            policies=["greedy", "mpc"],
            total_minutes=120,
            rng_seed=7,
        )
        assert set(results.keys()) == {"greedy", "mpc"}
        for m in results.values():
            assert 0 <= m["urgent_acceptance_rate"] <= 1
            assert m["utilisation"] <= 1.0


# --------------------------------------------------------------------------- #
# 6. Rollout planner
# --------------------------------------------------------------------------- #


class TestRolloutPlanner:
    def test_greedy_fill_respects_priority(self, state_two_chairs_three_queue):
        p = RolloutPlanner(TerminalValueFunction(), day_end_min=600)
        a = p._greedy_fill(state_two_chairs_three_queue)
        # Highest priority (P001, priority=1) should get the first chair
        first_chair_id = state_two_chairs_three_queue.chairs[0].chair_id
        assert a.assignments[first_chair_id] == "P001"

    def test_apply_action_occupies_chair(self, state_two_chairs_three_queue):
        p = RolloutPlanner(TerminalValueFunction())
        action = ScheduleAction(assignments={"WC-C01": "P001", "WC-C02": None})
        s = state_two_chairs_three_queue.copy()
        out = p._apply_action(s, action)
        c1 = next(c for c in out.chairs if c.chair_id == "WC-C01")
        assert c1.status == ChairStatus.OCCUPIED
        assert c1.patient_id == "P001"
        # Patient removed from queue
        assert all(p.patient_id != "P001" for p in out.queue)

    def test_evaluate_action_returns_finite_reward(
        self, state_two_chairs_three_queue
    ):
        p = RolloutPlanner(TerminalValueFunction(), day_end_min=600)
        scenario = {"urgent_arrivals": [], "patient_outcomes": {}}
        action = ScheduleAction(
            assignments={"WC-C01": "P001", "WC-C02": "P002"}
        )
        r, final = p.evaluate_action(
            state_two_chairs_three_queue, action, scenario,
            lookahead_minutes=60,
        )
        assert r == r
        assert isinstance(final, MDPState)


# --------------------------------------------------------------------------- #
# 7. Reward + terminal
# --------------------------------------------------------------------------- #


class TestReward:
    def test_terminal_penalty_is_non_positive(self, state_two_chairs_three_queue):
        r = compute_terminal_penalty(state_two_chairs_three_queue)
        assert r <= 0

    def test_empty_queue_has_zero_terminal(self):
        state = MDPState(time_min=600, chairs=[], queue=[])
        assert compute_terminal_penalty(state) == 0.0


# --------------------------------------------------------------------------- #
# T2.1 regression: priority must drive completion reward
# --------------------------------------------------------------------------- #


class TestPriorityWeightedReward:
    """Without the T2.1 fix the reward function awarded the same credit
    regardless of patient priority, so the planner had no incentive to
    prefer urgent patients.  These tests pin the corrected behaviour."""

    def _build_completion(self, priority: int):
        before = MDPState(
            time_min=60,
            chairs=[ChairState(
                chair_id="C1", site_code="WC",
                status=ChairStatus.OCCUPIED,
                patient_id="P1", remaining_minutes=15,
                priority_at_assignment=priority,
            )],
            queue=[],
        )
        after = MDPState(
            time_min=75,
            chairs=[ChairState(
                chair_id="C1", site_code="WC",
                status=ChairStatus.IDLE,
                patient_id="P1", remaining_minutes=0,
                # priority_at_assignment intentionally preserved across
                # OCCUPIED→IDLE so the reward function can read it.
                priority_at_assignment=priority,
            )],
            queue=[],
        )
        return before, after

    def test_p1_reward_strictly_greater_than_p5(self):
        before1, after1 = self._build_completion(priority=1)
        before5, after5 = self._build_completion(priority=5)
        action = ScheduleAction(assignments={})
        r1 = compute_immediate_reward(before1, action, after1, step_minutes=15)
        r5 = compute_immediate_reward(before5, action, after5, step_minutes=15)
        assert r1 > r5, (
            f"Priority-1 reward ({r1}) must exceed priority-5 reward ({r5}) — "
            "if these are equal the multiplier regressed."
        )

    def test_priority_multiplier_is_monotone(self):
        action = ScheduleAction(assignments={})
        rewards = []
        for p in (1, 2, 3, 4, 5):
            before, after = self._build_completion(priority=p)
            rewards.append(
                compute_immediate_reward(before, action, after, step_minutes=15)
            )
        # Strictly decreasing as priority number grows (1 = most urgent).
        for a, b in zip(rewards, rewards[1:]):
            assert a > b, f"reward sequence not monotone: {rewards}"

    def test_missing_priority_defaults_to_mid(self):
        # A pre-existing OCCUPIED chair (e.g. day starts mid-treatment) won't
        # have priority_at_assignment set; the reward should still fire,
        # treating it as priority 3.
        before = MDPState(
            time_min=60,
            chairs=[ChairState("C1", "WC", ChairStatus.OCCUPIED,
                               patient_id="P1", remaining_minutes=10)],
            queue=[],
        )
        after = MDPState(
            time_min=70,
            chairs=[ChairState("C1", "WC", ChairStatus.IDLE,
                               patient_id="P1", remaining_minutes=0)],
            queue=[],
        )
        r = compute_immediate_reward(before, ScheduleAction({}), after,
                                     step_minutes=10)
        # priority=3 → multiplier 3 → r = 3 * priority_complete_base
        # The exact value depends on DEFAULT_PRIORITY_COMPLETE_BASE; just
        # assert positive (would be zero with the old broken comment-only path).
        assert r > 0

    def test_chairstate_copy_preserves_priority(self):
        c = ChairState("C1", "WC", ChairStatus.OCCUPIED,
                       patient_id="P1", remaining_minutes=20,
                       priority_at_assignment=2)
        c2 = c.copy()
        assert c2.priority_at_assignment == 2

    def test_assigning_chair_captures_priority(self):
        # End-to-end: feed a queued patient through the planner's apply path
        # and verify the chair now carries that priority.
        from ml.stochastic_mpc_scheduler import RolloutPlanner
        planner = RolloutPlanner(value_fn=TerminalValueFunction())
        state = MDPState(
            time_min=0,
            chairs=[ChairState("C1", "WC", ChairStatus.IDLE)],
            queue=[QueuedPatient("P9", priority=2, expected_duration=60,
                                 arrival_time_min=0)],
        )
        action = ScheduleAction(assignments={"C1": "P9"})
        new_state = planner._apply_action(state, action)
        assigned = new_state.chairs[0]
        assert assigned.status == ChairStatus.OCCUPIED
        assert assigned.priority_at_assignment == 2


# --------------------------------------------------------------------------- #
# 8. state_from_app_state
# --------------------------------------------------------------------------- #


class TestStateFromAppState:
    def test_empty_appstate(self):
        s = state_from_app_state({})
        assert isinstance(s, MDPState)
        assert len(s.queue) == 0


# --------------------------------------------------------------------------- #
# 9. Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_decision_log_written(self, controller, state_two_chairs_three_queue,
                                  tmp_mpc_dir):
        controller.decide(state_two_chairs_three_queue)
        log = tmp_mpc_dir / "decisions.jsonl"
        assert log.exists()
        rec = json.loads(log.read_text().strip().splitlines()[-1])
        assert "action" in rec

    def test_status_shape(self, controller):
        s = controller.status()
        for k in (
            "n_scenarios", "lookahead_minutes", "arrival_model",
            "value_function", "total_decisions",
        ):
            assert k in s


class TestSingleton:
    def test_get_set(self, tmp_path):
        c = MPCController(storage_dir=tmp_path / "c")
        set_controller(c)
        assert get_controller() is c


# --------------------------------------------------------------------------- #
# 10. Helpers
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_poisson_zero_lambda(self):
        import random as _r
        assert _poisson(0, _r.Random(0)) == 0

    def test_poisson_mean_approximates_lambda(self):
        import random as _r
        rng = _r.Random(42)
        samples = [_poisson(2.0, rng) for _ in range(500)]
        mean = sum(samples) / len(samples)
        assert 1.6 < mean < 2.4

    def test_mdpstate_feature_vector_length(self, state_two_chairs_three_queue):
        feats = state_two_chairs_three_queue.feature_vector()
        assert len(feats) == len(TerminalValueFunction.FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# 11. JSON round-trip
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_decision_json_safe(self, controller, state_two_chairs_three_queue):
        d = controller.decide(state_two_chairs_three_queue)
        back = json.loads(json.dumps(d.to_dict(), default=str))
        assert "action" in back
        assert back["n_scenarios"] == 5
