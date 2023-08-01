import numpy as np

class Agent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        q_table_size = int,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = np.zeros(q_table_size)

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs, env):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return np.unravel_index(np.argmax(self.q_values, axis=None), self.q_values.shape)[2]

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """Updates the Q-value of an action."""
        #print(next_obs, next_obs)
        #print(self.q_values[next_obs[0]][next_obs[1]])
        future_q_value = (not terminated) * np.max(self.q_values[next_obs[0]][next_obs[1]])
        #print(obs[0], obs[1])
        #print(self.q_values[obs[0]][obs[1]])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs[0]][obs[1]][action]
        )

        self.q_values[obs[0]][obs[1]][action] = (
            self.q_values[obs[0]][obs[1]][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)