"""
Reinforcement Learning for Dynamic SACT Scheduling

Single-Agent Q-Learning + Multi-Agent RL (MARL) for chair assignment.

1. SchedulingRLAgent: Single Q-Learning agent (centralized)
   Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a' Q(s',a') - Q(s,a)]

2. MultiAgentChairScheduler: MARL with independent chair agents
   Each chair c has its own policy pi_c(s_c, a_c)
   Coordination via shared reward + communication channel

References:
    Sutton & Barto (2018). Reinforcement Learning: An Introduction.
    Watkins & Dayan (1992). Q-Learning. Machine Learning.
    Tan (1993). Multi-Agent RL: Independent vs Cooperative Agents.
    Lowe et al. (2017). MADDPG: Multi-Agent Actor-Critic.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLState:
    """State representation for the scheduling MDP."""
    hour: int                    # Current hour (8-17)
    chairs_occupied: int         # Number of chairs currently in use
    total_chairs: int            # Total chairs at site
    queue_size: int              # Patients waiting to be scheduled
    avg_noshow_risk: float       # Average no-show risk of queued patients
    urgent_count: int            # Number of urgent patients in queue
    utilization: float           # Current chair utilization (0-1)

    def to_discrete(self) -> Tuple:
        """Convert to discrete state for Q-table lookup."""
        hour_bin = min(self.hour - 8, 9)  # 0-9 for hours 8-17
        occ_bin = min(self.chairs_occupied // 3, 6)  # 0-6 occupancy bins
        queue_bin = min(self.queue_size // 2, 5)  # 0-5 queue bins
        risk_bin = min(int(self.avg_noshow_risk * 5), 4)  # 0-4 risk bins
        urgent_bin = min(self.urgent_count, 3)  # 0-3 urgent bins
        return (hour_bin, occ_bin, queue_bin, risk_bin, urgent_bin)


@dataclass
class RLAction:
    """Action in the scheduling MDP."""
    action_type: str       # 'assign', 'delay', 'double_book', 'reject'
    chair_id: int          # Which chair (0-N)
    time_slot: int         # Which time slot (minutes from day start)

    def to_discrete(self, n_chairs: int) -> int:
        """Convert to discrete action index."""
        type_map = {'assign': 0, 'delay': 1, 'double_book': 2, 'reject': 3}
        type_idx = type_map.get(self.action_type, 0)
        chair_idx = min(self.chair_id, n_chairs - 1)
        return type_idx * n_chairs + chair_idx


class SchedulingRLAgent:
    """
    Q-Learning agent for dynamic SACT scheduling.

    Learns which chair/time assignments lead to best outcomes
    (low waiting, high utilization, few no-shows).

    Q(s,a) <- Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self,
                 n_chairs: int = 19,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 noshow_penalty_weight: float = 0.3):
        """
        Args:
            n_chairs: Number of chairs at the site
            learning_rate: α — how fast Q-values update
            discount_factor: γ — importance of future rewards
            epsilon: ε — exploration rate (probability of random action)
            noshow_penalty_weight: λ — weight of no-show risk in reward
        """
        self.n_chairs = n_chairs
        self.alpha = learning_rate      # α
        self.gamma = discount_factor    # γ
        self.epsilon = epsilon          # ε
        self.lambda_noshow = noshow_penalty_weight  # λ

        # Q-table: state -> action -> value
        # Using dictionary for sparse state space
        self.q_table: Dict[Tuple, np.ndarray] = {}

        # Action space: 4 types × n_chairs
        self.n_actions = 4 * n_chairs

        # Training statistics
        self.episodes = 0
        self.total_reward = 0.0
        self.rewards_history: List[float] = []

        logger.info(f"RL Scheduler initialized (alpha={learning_rate}, gamma={discount_factor}, "
                     f"epsilon={epsilon}, chairs={n_chairs})")

    def _get_q_values(self, state: Tuple) -> np.ndarray:
        """Get Q-values for a state, initializing if new."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state: RLState, available_chairs: List[int] = None) -> RLAction:
        """
        Choose scheduling action using ε-greedy policy.

        With probability ε: explore (random action)
        With probability 1-ε: exploit (best Q-value action)
        """
        discrete_state = state.to_discrete()
        q_values = self._get_q_values(discrete_state)

        # ε-greedy
        if np.random.random() < self.epsilon:
            # Explore: random action
            action_type = np.random.choice(['assign', 'delay', 'double_book', 'reject'],
                                            p=[0.6, 0.2, 0.15, 0.05])
            if available_chairs:
                chair = np.random.choice(available_chairs)
            else:
                chair = np.random.randint(0, self.n_chairs)
            time_slot = state.hour * 60 + np.random.choice([0, 15, 30, 45])
        else:
            # Exploit: best action
            best_action_idx = np.argmax(q_values)
            action_type_idx = best_action_idx // self.n_chairs
            chair = best_action_idx % self.n_chairs
            action_types = ['assign', 'delay', 'double_book', 'reject']
            action_type = action_types[min(action_type_idx, 3)]
            time_slot = state.hour * 60

        return RLAction(action_type=action_type, chair_id=chair, time_slot=time_slot)

    def compute_reward(self,
                       waiting_time: float,
                       noshow_risk: float,
                       utilization: float,
                       patient_priority: int,
                       successful: bool) -> float:
        """
        Compute reward for a scheduling action.

        r = -waiting_time - λ·no_show_risk + utilization_bonus + priority_bonus

        Args:
            waiting_time: How long patient waited (minutes)
            noshow_risk: Predicted no-show probability
            utilization: Chair utilization after assignment
            patient_priority: P1-P4 (1=highest)
            successful: Whether patient actually attended
        """
        # Negative reward for waiting
        waiting_penalty = -waiting_time / 60.0  # Normalize to hours

        # Negative reward for no-show risk
        noshow_penalty = -self.lambda_noshow * noshow_risk * 10

        # Positive reward for utilization
        util_bonus = utilization * 2.0

        # Priority bonus (higher priority = more reward for scheduling)
        priority_bonus = (5 - patient_priority) * 0.5

        # Outcome bonus (if we know the actual outcome)
        outcome_bonus = 1.0 if successful else -2.0

        reward = waiting_penalty + noshow_penalty + util_bonus + priority_bonus + outcome_bonus
        return reward

    def update(self,
               state: RLState,
               action: RLAction,
               reward: float,
               next_state: RLState):
        """
        Q-Learning update rule.

        Q(s,a) <- Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        """
        s = state.to_discrete()
        a = action.to_discrete(self.n_chairs)
        s_next = next_state.to_discrete()

        q_current = self._get_q_values(s)[a]
        q_next_max = np.max(self._get_q_values(s_next))

        # Q-Learning update
        td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_current

        self._get_q_values(s)[a] = q_current + self.alpha * td_error

        self.episodes += 1
        self.total_reward += reward
        self.rewards_history.append(reward)

        return {
            'td_error': round(td_error, 4),
            'q_before': round(q_current, 4),
            'q_after': round(q_current + self.alpha * td_error, 4),
            'reward': round(reward, 4),
        }

    def train_on_historical(self, appointments: List[Dict], n_epochs: int = 10) -> Dict:
        """
        Train the RL agent on historical appointment data.

        Simulates the scheduling process on past data to learn good policies.
        """
        total_reward = 0
        total_updates = 0

        for epoch in range(n_epochs):
            epoch_reward = 0

            for i, appt in enumerate(appointments):
                # Create state from appointment context
                hour = int(str(appt.get('Time', appt.get('Appointment_Hour', '10')).split(':')[0])
                           if isinstance(appt.get('Time', '10'), str) and ':' in str(appt.get('Time', '10'))
                           else appt.get('Appointment_Hour', 10))

                state = RLState(
                    hour=hour,
                    chairs_occupied=np.random.randint(0, self.n_chairs),
                    total_chairs=self.n_chairs,
                    queue_size=np.random.randint(0, 10),
                    avg_noshow_risk=appt.get('Patient_NoShow_Rate', 0.1),
                    urgent_count=1 if appt.get('Priority', 'P3') in ['P1', 'P2'] else 0,
                    utilization=np.random.uniform(0.3, 0.9),
                )

                # Choose action
                action = self.choose_action(state)

                # Compute reward from actual outcome
                attended = appt.get('Attended_Status', 'Yes') == 'Yes'
                noshow_risk = appt.get('Patient_NoShow_Rate', 0.1)
                priority_str = str(appt.get('Priority', 'P3'))
                priority = int(priority_str.replace('P', '')) if priority_str.startswith('P') else 3

                reward = self.compute_reward(
                    waiting_time=np.random.uniform(0, 30),
                    noshow_risk=noshow_risk,
                    utilization=state.utilization,
                    patient_priority=priority,
                    successful=attended,
                )

                # Create next state
                next_state = RLState(
                    hour=min(hour + 1, 17),
                    chairs_occupied=state.chairs_occupied + (1 if action.action_type == 'assign' else 0),
                    total_chairs=self.n_chairs,
                    queue_size=max(0, state.queue_size - 1),
                    avg_noshow_risk=noshow_risk,
                    urgent_count=state.urgent_count,
                    utilization=min(1.0, state.utilization + 0.05),
                )

                # Update Q-values
                self.update(state, action, reward, next_state)
                epoch_reward += reward
                total_updates += 1

            total_reward += epoch_reward

        avg_reward = total_reward / max(n_epochs, 1)

        logger.info(f"RL training complete: {n_epochs} epochs, {total_updates} updates, "
                     f"avg_reward={avg_reward:.2f}")

        return {
            'epochs': n_epochs,
            'total_updates': total_updates,
            'avg_reward_per_epoch': round(avg_reward, 2),
            'q_table_size': len(self.q_table),
            'total_episodes': self.episodes,
        }

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of learned policy."""
        if not self.q_table:
            return {'status': 'not_trained', 'q_table_size': 0}

        # Analyze learned policy
        action_counts = {'assign': 0, 'delay': 0, 'double_book': 0, 'reject': 0}
        for state, q_vals in self.q_table.items():
            best_action_idx = np.argmax(q_vals)
            action_type_idx = best_action_idx // self.n_chairs
            action_types = ['assign', 'delay', 'double_book', 'reject']
            action_counts[action_types[min(action_type_idx, 3)]] += 1

        total = sum(action_counts.values())

        return {
            'status': 'trained',
            'q_table_size': len(self.q_table),
            'total_episodes': self.episodes,
            'avg_reward': round(self.total_reward / max(self.episodes, 1), 3),
            'policy_distribution': {k: round(v / max(total, 1), 3) for k, v in action_counts.items()},
            'parameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'lambda_noshow': self.lambda_noshow,
                'n_chairs': self.n_chairs,
            },
            'formula': 'Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a\' Q(s\',a\') - Q(s,a)]',
            'state_space': 'hour(10) x occupancy(7) x queue(6) x risk(5) x urgent(4) = 8,400 states',
            'action_space': f'4 types x {self.n_chairs} chairs = {self.n_actions} actions',
            'reward': 'r = -waiting_time - lambda*noshow_risk + utilization + priority',
        }

    def recommend_action(self, state: RLState) -> Dict[str, Any]:
        """Get the recommended action for a state (no exploration)."""
        old_epsilon = self.epsilon
        self.epsilon = 0  # Pure exploitation
        action = self.choose_action(state)
        self.epsilon = old_epsilon

        discrete_state = state.to_discrete()
        q_values = self._get_q_values(discrete_state)
        best_q = float(np.max(q_values))

        return {
            'recommended_action': action.action_type,
            'chair': int(action.chair_id),
            'time_slot': int(action.time_slot),
            'q_value': round(float(best_q), 4),
            'confidence': round(1.0 - self.epsilon, 2),
            'state': {
                'hour': int(state.hour),
                'chairs_occupied': int(state.chairs_occupied),
                'queue_size': int(state.queue_size),
                'utilization': round(float(state.utilization), 2),
            }
        }


# ============================================================================
# MULTI-AGENT RL FOR CHAIR ASSIGNMENT (MARL)
# ============================================================================

@dataclass
class ChairState:
    """Local state for a single chair agent."""
    chair_id: int
    is_occupied: bool           # Currently treating a patient
    current_patient_priority: int  # 0 if empty
    current_remaining_min: int  # Minutes left for current patient
    next_gap_min: int           # Gap until next scheduled appointment
    queue_size: int             # Shared: patients waiting globally
    hour: int                   # Current hour


@dataclass
class ChairAction:
    """Action for a single chair agent."""
    action_type: str  # 'accept', 'reject', 'delay'
    # accept: take next patient from queue
    # reject: pass to another chair
    # delay: wait for higher-priority patient


class ChairAgent:
    """
    Independent Q-Learning agent for a single chair.

    Each chair has its own Q-table and learns independently,
    but receives a shared team reward to encourage cooperation.

    Local state: chair occupancy, gap, remaining time
    Local action: accept/reject/delay next patient
    Reward: shared R = w_util*util - w_wait*wait - w_noshow*noshow + fairness
    """

    def __init__(self, chair_id: int, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.15):
        self.chair_id = chair_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple, np.ndarray] = {}
        self.n_actions = 3  # accept, reject, delay
        self.total_reward = 0.0
        self.updates = 0

    def _state_to_key(self, state: ChairState) -> Tuple:
        """Discretize chair-local state."""
        hour_bin = min(state.hour - 8, 9)
        occ = 1 if state.is_occupied else 0
        pri_bin = min(state.current_patient_priority, 4)
        rem_bin = min(state.current_remaining_min // 30, 4)
        gap_bin = min(state.next_gap_min // 15, 6)
        queue_bin = min(state.queue_size // 2, 5)
        return (hour_bin, occ, pri_bin, rem_bin, gap_bin, queue_bin)

    def _get_q(self, state_key: Tuple) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def select_action(self, state: ChairState) -> str:
        """Epsilon-greedy action selection."""
        state_key = self._state_to_key(state)
        q = self._get_q(state_key)

        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            action_idx = int(np.argmax(q))

        return ['accept', 'reject', 'delay'][action_idx]

    def update(self, state: ChairState, action: str, reward: float,
               next_state: ChairState):
        """Q-Learning update with shared reward."""
        action_map = {'accept': 0, 'reject': 1, 'delay': 2}
        a_idx = action_map.get(action, 0)

        s_key = self._state_to_key(state)
        ns_key = self._state_to_key(next_state)

        q_s = self._get_q(s_key)
        q_ns = self._get_q(ns_key)

        # Q(s,a) <- Q(s,a) + alpha[r + gamma*max Q(s',a') - Q(s,a)]
        td_target = reward + self.gamma * np.max(q_ns)
        td_error = td_target - q_s[a_idx]
        q_s[a_idx] += self.alpha * td_error

        self.total_reward += reward
        self.updates += 1


class MultiAgentChairScheduler:
    """
    Multi-Agent Reinforcement Learning (MARL) for chair assignment.

    Each chair is an independent agent with its own Q-table.
    Agents coordinate via:
    1. Shared reward (team objective)
    2. Communication channel (broadcast queue state)
    3. Fairness bonus in reward function

    This replaces the centralized greedy/CP-SAT with a scalable,
    learned scheduling policy.

    Environment:
        State_c: (occupied, priority, remaining, gap, queue, hour) per chair
        Action_c: {accept, reject, delay} per chair
        Reward: R = w_util*util - w_wait*wait - w_noshow*noshow + fairness_bonus

    References:
        Tan (1993). Multi-Agent RL: Independent vs Cooperative.
        Lowe et al. (2017). MADDPG.
    """

    def __init__(self, n_chairs: int = 19, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.15,
                 w_util: float = 2.0, w_wait: float = 1.0,
                 w_noshow: float = 0.5, w_fairness: float = 0.3):
        self.n_chairs = n_chairs
        self.w_util = w_util
        self.w_wait = w_wait
        self.w_noshow = w_noshow
        self.w_fairness = w_fairness

        # Create independent agents — one per chair
        self.agents: List[ChairAgent] = [
            ChairAgent(chair_id=c, alpha=alpha, gamma=gamma, epsilon=epsilon)
            for c in range(n_chairs)
        ]

        # Global tracking
        self.episodes = 0
        self.total_team_reward = 0.0
        self.assignments_history: List[Dict] = []

        logger.info(f"MARL Chair Scheduler: {n_chairs} independent agents "
                     f"(alpha={alpha}, gamma={gamma}, epsilon={epsilon})")

    def compute_shared_reward(
        self, utilization: float, total_wait: float,
        noshow_rate: float, priority_variance: float
    ) -> float:
        """
        Compute shared team reward for all agents.

        R = w_util * utilization
          - w_wait * (total_wait / 60)
          - w_noshow * noshow_rate * 10
          + w_fairness * (1 - priority_variance)

        The shared reward encourages cooperation:
        all agents benefit when the team performs well.
        """
        fairness_bonus = self.w_fairness * max(0, 1.0 - priority_variance)

        reward = (
            self.w_util * utilization
            - self.w_wait * (total_wait / 60.0)
            - self.w_noshow * noshow_rate * 10
            + fairness_bonus
        )
        return float(reward)

    def assign_patients(
        self, patients: List[Dict], chair_states: List[ChairState]
    ) -> List[Dict]:
        """
        Assign patients to chairs using MARL policy.

        Each chair agent independently decides whether to accept
        the next patient in the queue. The first chair to accept
        gets the assignment.

        Args:
            patients: List of patient dicts with priority, duration, noshow_prob
            chair_states: Current state of each chair

        Returns:
            List of assignment dicts {patient_id, chair_id, action}
        """
        assignments = []
        remaining_patients = list(patients)

        for patient in remaining_patients:
            assigned = False
            patient_priority = patient.get('priority', 3)
            patient_noshow = patient.get('noshow_probability', 0.1)

            # Each chair agent votes on whether to accept
            chair_votes = []
            for c, agent in enumerate(self.agents):
                if c < len(chair_states):
                    state = chair_states[c]
                else:
                    state = ChairState(
                        chair_id=c, is_occupied=False,
                        current_patient_priority=0,
                        current_remaining_min=0,
                        next_gap_min=60, queue_size=len(remaining_patients),
                        hour=10
                    )

                action = agent.select_action(state)
                chair_votes.append((c, action, agent._get_q(
                    agent._state_to_key(state)
                ).max()))

            # Sort by Q-value (highest confidence first)
            chair_votes.sort(key=lambda x: x[2], reverse=True)

            # First chair that accepts gets the patient
            for c_id, action, q_val in chair_votes:
                if action == 'accept' and not any(
                    a['chair_id'] == c_id for a in assignments
                ):
                    assignments.append({
                        'patient_id': patient.get('patient_id', f'P{len(assignments)}'),
                        'chair_id': c_id,
                        'action': 'accept',
                        'q_value': float(q_val),
                        'priority': patient_priority,
                    })
                    assigned = True
                    break

            if not assigned:
                # Assign to least-occupied chair as fallback
                used_chairs = {a['chair_id'] for a in assignments}
                for c in range(self.n_chairs):
                    if c not in used_chairs:
                        assignments.append({
                            'patient_id': patient.get('patient_id', f'P{len(assignments)}'),
                            'chair_id': c,
                            'action': 'fallback',
                            'q_value': 0.0,
                            'priority': patient_priority,
                        })
                        break

        return assignments

    def train_on_historical(
        self, records: List[Dict], n_epochs: int = 5
    ) -> Dict[str, Any]:
        """
        Train all chair agents on historical appointment data.

        Simulates the scheduling environment: for each historical
        appointment, the assigned chair agent receives a reward
        based on whether the patient attended.
        """
        total_updates = 0
        epoch_rewards = []

        for epoch in range(n_epochs):
            epoch_reward = 0.0
            np.random.shuffle(records)

            for i, record in enumerate(records):
                chair_id = int(record.get('Chair_Number', 0)) % self.n_chairs
                hour = int(record.get('Appointment_Hour', 10))
                priority = int(str(record.get('Priority', 'P3')).replace('P', '') or 3)
                duration = int(record.get('Planned_Duration', 120) or 120)
                attended = record.get('Attended_Status', 'Yes') == 'Yes'
                noshow_rate = float(record.get('Patient_NoShow_Rate', 0.1) or 0.1)

                # Current state
                state = ChairState(
                    chair_id=chair_id,
                    is_occupied=True,
                    current_patient_priority=priority,
                    current_remaining_min=duration,
                    next_gap_min=15,
                    queue_size=max(0, len(records) - i),
                    hour=hour
                )

                # Action taken (historical: always 'accept')
                action = 'accept'

                # Compute reward based on outcome
                util = 0.8 if attended else 0.3
                wait = max(0, i % 10) * 5  # Approximate wait
                reward = self.compute_shared_reward(
                    utilization=util,
                    total_wait=wait,
                    noshow_rate=noshow_rate,
                    priority_variance=0.2
                )

                # Next state
                next_state = ChairState(
                    chair_id=chair_id,
                    is_occupied=False,
                    current_patient_priority=0,
                    current_remaining_min=0,
                    next_gap_min=30,
                    queue_size=max(0, len(records) - i - 1),
                    hour=min(17, hour + duration // 60)
                )

                # Update the specific chair agent
                agent = self.agents[chair_id]
                agent.update(state, action, reward, next_state)
                epoch_reward += reward
                total_updates += 1

            epoch_rewards.append(epoch_reward)
            self.episodes += 1

        self.total_team_reward = sum(epoch_rewards)

        return {
            'epochs': n_epochs,
            'total_updates': total_updates,
            'avg_epoch_reward': round(float(np.mean(epoch_rewards)), 2),
            'agents': self.n_chairs,
            'agent_updates': [a.updates for a in self.agents],
            'total_q_entries': sum(len(a.q_table) for a in self.agents),
        }

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of learned multi-agent policy."""
        agent_summaries = []
        for a in self.agents:
            agent_summaries.append({
                'chair_id': a.chair_id,
                'q_entries': len(a.q_table),
                'total_reward': round(a.total_reward, 2),
                'updates': a.updates,
            })

        return {
            'method': 'Multi-Agent RL (Independent Q-Learning per Chair)',
            'n_agents': self.n_chairs,
            'formulas': {
                'per_agent_update': 'Q_c(s_c, a_c) <- Q_c(s_c, a_c) + alpha[R + gamma*max Q_c(s_c\', a_c\') - Q_c(s_c, a_c)]',
                'shared_reward': 'R = w_util*util - w_wait*wait - w_noshow*noshow + w_fair*(1-var_priority)',
                'coordination': 'Shared reward + Q-value voting for patient assignment',
            },
            'state_per_agent': '(hour, occupied, priority, remaining, gap, queue) = 10*2*5*5*7*6 = 21,000 states',
            'action_per_agent': '3 actions: accept, reject, delay',
            'episodes': self.episodes,
            'total_team_reward': round(self.total_team_reward, 2),
            'agents': agent_summaries[:5],  # Show first 5
            'total_q_entries': sum(len(a.q_table) for a in self.agents),
        }
