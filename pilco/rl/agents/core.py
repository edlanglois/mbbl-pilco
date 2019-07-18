"""Core Markov decision process agents, interfaces, and tools."""
import pickle


def load(file):
    """Load an agent from a file."""
    with open(file, "rb") as f:
        data = pickle.load(f)
        return data["agent"]


class Agent:
    """Agent interface for OpenAI Gym environments."""

    def __init__(
        self,
        env=None,
        *,
        observation_space=None,
        action_space=None,
        reward_range=None,
        discount_factor=1,
        greedy=False,
        trainable=True,
        **kwargs
    ):
        """Initialize an Agent.

        Args:
            env: Optional environment in which the agent will act.
            observation_space: The environment observation space.
                Obtained from `env` if omitted.
            action_space: The environment action space.
                Obtained from `env` if omitted.
            reward_range: The environment reward range.
                Obtained from `env` if omitted.
            discount_factor: Future reward discount factor used by the agent.
            greedy: If `True`, the agent acts the best it can with no attempt
                at exploration.
            trainable: Whether the agent can be trained after initialization
                via `update`.
            **kwargs: Extra arguments are discarded.
        """
        del kwargs  # Allow callers to pass extra arguments.
        if env is not None:
            if observation_space is None:
                observation_space = env.observation_space
            if action_space is None:
                action_space = env.action_space
            if reward_range is None:
                try:
                    reward_range = env.reward_range
                except AttributeError:
                    reward_range = (float("-inf"), float("inf"))

        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range
        self.discount_factor = discount_factor
        self.greedy = greedy
        self.trainable = trainable

    def _is_greedy(self, greedy):
        """Whether to act greedily given own state & optional override."""
        if greedy is None:
            return self.greedy
        return greedy

    def act(self, observation, *, greedy=None):
        """Select an action given an observation.

        Args:
            observation: The observation for the current state.
            greedy: If given, overrides the default `greedy` setting.

        Returns:
            An action from `action_space`.
        """
        if self._is_greedy(greedy):
            return self.act_greedy(observation)
        return self._act_normal(observation)

    def act_greedy(self, observation):
        """Select a greedy action given an observation.

        Args:
            observation: The observation for the current state.

        Returns:
            An action from `action_space`.
        """
        return self._act_normal(observation)

    def _act_normal(self, observation):
        """Select an action given an observation.

        Args:
            observation: The observation for the current state.

        Returns:
            An action from `action_space`.
        """
        raise NotImplementedError

    def update(self, step_info):
        """Update the agent based on the environment step."""
        del step_info
        if self.trainable:
            raise NotImplementedError

    def policy(self, observation, *, greedy=None):
        """"Generate a distribution of actions for a given observation.

        Args:
            observation: The observation for the current state.
            greedy: If given, overrides the default `greedy` setting.
        """
        raise NotImplementedError

    def save(self, file):
        """Save the agent to a file."""
        with open(file, "wb") as f:
            pickle.dump({"agent": self}, f)
