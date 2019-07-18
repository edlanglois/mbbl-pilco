"""Running RL environment."""
import itertools

from . import core


def run_steps(env, agent, update=True, num_steps=None):
    """Generate steps from an environment-agent simulation.

    Args:
        env: An OpenAI Gym environment.
        agent: An instance of `Agent`.
        update: Update the agent with step results.
        num_steps: Number of steps to run. By default runs forever.

    Yields:
        The step info for each environment step.
    """
    observation = env.reset()
    if num_steps is not None:
        counter = range(num_steps)
    else:
        counter = itertools.count()

    for _ in counter:
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)
        step_info = core.StepInfo(
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=done,
            info=info,
        )
        yield step_info
        if update:
            agent.update(step_info)
        if done:
            observation = env.reset()
        else:
            observation = next_observation


def episode_iterator(steps):
    """Yields from steps until the end of the episode."""
    for step_info in steps:
        yield step_info
        if step_info.done:
            break


def n_steps_iterator(n):
    """Returns a generator function that yields n steps."""

    def n_steps_iterator_(steps):
        yield from itertools.islice(steps, n)

    return n_steps_iterator_


def run_step_sequences(
    env, agent, sequence_iterator=episode_iterator, update=True, num_sequences=None
):
    """Generate sequences of steps from an environment-agent simulation.

    Args:
        env: An OpenAI Gym environment.
        agent: An instance of `Agent`.
        sequence_iterator: A generator taking an iterator of steps and
            yielding steps until the end of the sequence.
        update: Update the agent with step results.
        num_sequences: Number of sequences to run. By default runs forever.

    Yields:
        Successive sequences of steps.
    """
    steps = run_steps(env=env, agent=agent, update=update)
    if num_sequences is not None:
        counter = range(num_sequences)
    else:
        counter = itertools.count()
    for _ in counter:
        yield sequence_iterator(steps)


def batch_run_steps(envs, agents, update=True):
    """Generate batched steps from environment-agent simulations.

    Args:
        envs: A list of `batch_size` environments.
        agents: A list of `batch_size` agents.
        update: Update the agents with step results.

    Yields:
        At each step, a list of `batch_size` step_info objects.
    """
    yield from zip(
        *[run_steps(env, agent, update=update) for env, agent in zip(envs, agents)]
    )


def batch_run_step_sequences(
    envs, agents, sequence_iterator=episode_iterator, update=True
):
    """Generate batched step sequences from environment-agent simulations.

    Args:
        envs: A list of `batch_size` environments.
        agents: A list of `batch_size` agents.
        sequence_iterator: A generator taking an iterator of steps and
            yielding steps until the end of the sequence.
        update: Update the agents with step results.

    Yields:
        A list of `batch_size` step sequence iterators.
    """
    yield from zip(
        *[
            run_step_sequences(
                env, agent, update=update, sequence_iterator=sequence_iterator
            )
            for env, agent in zip(envs, agents)
        ]
    )
