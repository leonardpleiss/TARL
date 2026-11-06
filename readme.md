# Target-Aligned Reinforcement Learning (TARL) for Stable-Baselines3

This repository contains the implementations accompanying the NeurIPS 2026 paper **[Target-Aligned Reinforcement Learning](https://arxiv.org/abs/2603.29501)** by Leonard S. Pleiss, James Harrison, and Maximilian Schiffer. TARL is provided as drop-in replacements for the DQN and SAC models of [Stable-Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) and requires no tuning of the base algorithm's hyperparameters.

### The stability-recency trade-off

Target networks stabilize bootstrapped learning, but their estimates lag behind the online network and can become stale. Slower target updates improve stability at the cost of recency, and faster updates risk reintroducing instability. TARL leaves the target update schedule untouched. Instead, it uses the recent (online) network to validate the stable (target) network on a per-transition basis, and focuses updates on transitions for which the two are well aligned.

---

## Files

- `dqn_custom.py` — `CustomDQN`: Target-Aligned DQN and DDQN (TA-DQN / TA-DDQN).
- `sac_custom.py` — `CustomSAC`: Target-Aligned Soft Actor-Critic (TA-SAC).

Both modules follow the SB3 reference implementations and are compatible with SB3 training loops, callbacks, vector environments, and logging.

---

## Installation

```bash
git clone https://github.com/leonardpleiss/TARL.git
cd TARL
pip install stable-baselines3
```

---

## Usage

Both models are direct replacements for the corresponding SB3 models. All standard SB3 arguments remain supported, alongside the TARL-specific arguments described below.

```python
import gymnasium as gym
from dqn_custom import CustomDQN
from sac_custom import CustomSAC

# Target-Aligned DDQN
model = CustomDQN(
    policy="MlpPolicy",
    env=gym.make("CartPole-v1"),
    target="DDQN",            # offline target: "DQN" or "DDQN"
    oversampling_ratio=1.0,   # sample 2m candidates, keep the m best-aligned
    td_scale=False,           # standard alignment-based oversampling
    verbose=1,
)
model.learn(total_timesteps=100_000)

# Target-Aligned SAC
model = CustomSAC(
    policy="MlpPolicy",
    env=gym.make("Pendulum-v1"),
    oversampling_ratio=1.0,
    td_scale=False,
    verbose=1,
)
model.learn(total_timesteps=50_000)
```

### Configuration

- `target` *(str, `CustomDQN` only)*: The offline target. `"DQN"` uses the standard DQN target; `"DDQN"` uses the Double DQN target, where the online network selects the next action and the target network evaluates it.
- `oversampling_ratio` *(float)*: The oversampling margin *b*, as a multiple of the batch size *m*. Each update samples *m + b* candidate transitions and keeps the *m* with the highest alignment (*b* is floored to an integer).
  - `1.0` samples *2m* candidates. This is the setting used for all main experiments in the paper.
  - In our ablations, larger values generally yielded stronger gains, at the cost of a wider forward pass. We recommend choosing *b* such that *m + b* matches the largest batch your hardware can evaluate in a single forward pass.
- `td_scale` *(bool)*: **[EXPERIMENTAL]** If `True`, the alignment score scales the per-transition TD loss instead of being used for top-*m* selection. This loss-scaling variant is discussed as an alternative in Appendix A of the paper, but it was not used in the paper's experiments. Keep it `False` to reproduce the paper.

---

## How it works: alignment-based oversampling

TARL modifies only the batch selection step of the standard training loop:

1. **Sample** an oversized minibatch of *m + b* transitions from the replay buffer.
2. **Score** each transition. TARL computes the online TD error $\delta$ (bootstrapping from the online network) and the offline TD error $\bar\delta$ (bootstrapping from the target network), and combines them into the target alignment score

$$
\mathcal{A} =
\begin{cases}
1 & \text{if } \delta \bar\delta > 0 \text{ and } |\delta| \geq |\bar\delta|, \\
\dfrac{|\delta|}{|\delta| + |\delta - \bar\delta| + \varepsilon} & \text{otherwise.}
\end{cases}
$$

   The score is 1 when the online network fully supports the update proposed by the target network: both point in the same direction, and the offline step is no larger than the online one. It decreases when the online network supports the offline update only partially (the offline target overshoots) or contradicts it (the two disagree on direction).

3. **Select** the *m* transitions with the highest alignment.
4. **Update** the network with the base algorithm's standard loss on these *m* transitions.

TARL does **not** modify the TD targets: every selected transition is regressed onto the base algorithm's unchanged offline target. TARL only changes *which* transitions enter each update. Poorly aligned transitions are deferred rather than discarded, and under hard target updates, every target synchronization resets their alignment.

**TA-SAC.** Alignment is computed separately for each of the twin critics, and each transition is assigned the minimum of the two scores. The entropy term $\alpha \log \pi(a' \mid s')$ is included in both the online and the offline targets, so that alignment is evaluated under the soft Bellman equation.

### Computational overhead

TARL requires a wider forward pass over the *m + b* candidates; the backward pass is performed on *m* transitions, as in the base algorithm. In the paper, this increased wall-clock time by 3.29% for TA-DDQN on Atari (*b = m*) and by 17.82% for TA-SAC, since vanilla SAC does not natively compute online next-state values.

---

## Reproducing the paper

All main experiments use `oversampling_ratio=1.0` and the base algorithm's hyperparameters without further tuning. The hyperparameters for all experiments are listed in Appendix C of the paper; SAC hyperparameters follow [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo). This repository contains the algorithm implementations; it does not include experiment scripts.

---

## Citation

If you use TARL in your work, please cite:

```bibtex
@inproceedings{pleiss2026target,
  title         = {Target-Aligned Reinforcement Learning},
  author        = {Pleiss, Leonard S. and Harrison, James and Schiffer, Maximilian},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2603.29501},
  archivePrefix = {arXiv}
}
```

## License

This codebase is licensed under the MIT License, following the license of Stable-Baselines3.
