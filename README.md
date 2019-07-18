# PILCO
An Implementation of the paper
[PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://dl.acm.org/citation.cfm?id=3104541)
by Mark Deisenroth and Carl Rasmussen

## Author
Eric Langlois

## Install
```sh
pip install -e .[tf_gpu]
# for CPU instead:
# pip install -e .[tf]
```

## Test
These take awhile.
```sh
python -m pytest
```
This only runs the faster tests (takes ~30s). To run all tests (20min) use
```sh
python -m pytest --run-slow
```

## Pre-Commit
The code is automatically formatted with [Black](https://github.com/ambv/black).
This is enforced with pre-commit checks. Install them with
```sh
pip install pre-commit
pre-commit install
```

## Run
Cart-pole environment (defaults for most settings):
```
./scripts/run-pilco.py --log-level debug --gpu --visualize
```

Inverted pendulum with logging (in `~/data/pilco` by default, change with
`--root-logdir`)
```
./scripts/run-pilco.py --gpu --visualize --log \
    --env InvertedPendulumExtra-v2 \
    --random-actions
```

Available environments:
* `ContinuousCartPole-v0`
* `InvertedPendulumExtra-v2`
* `InvertedDoublePendulumExtra-v2`
* `SwimmerExtra-v2`

*Note*: Training does not currently seem to work well on the MuJoCo environments
(all but `ContinuousCartPole`).

See arguments descriptions:
```
./scripts/run-pilco.py --help
```

Use `mbbl-run.py` to run on the Model-Based Baseline environments.

## Environments
The script runs only on Gym environments that define the
following environment keys:
* `reward.moment_map`: The reward function as a moment map.
* `initial_state.mean`: The initial state mean vector.
* `initial_state.covariance`: The initial state covariance matrix.

See `pilco.rl.envs.gym` for examples.

## Licence
This project is distributed under the MIT license (see the `LICENSE` file).

There are additional copyright notices within code in the `pilco/third_party`
directory.

## Troubleshooting
### ImportError: Failed to import any qt binding
```sh
pip install pyqt5
```

### Crash: Matrix Not Invertible
* Sometimes the error is flaky and running again will succeed the next time.
* Try increasing `--min-noise`
* Try setting `--random-actions` (might instead make it worse)
