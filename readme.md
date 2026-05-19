# Target-Aligned RL Policies for Stable-Baselines3 (TARL)

This repository provides **Target-Aligned Reinforcement Learning (TARL)** extensions of standard off-policy RL algorithms. Designed as seamless, drop-in custom policies for the [Stable-Baselines3 (SB3)](https://www.google.com/search?q=https://github.com/DLR-RM/stable-baselines3) ecosystem, these implementations mitigate the fundamental stability-recency tradeoff in deep RL without requiring base hyperparameter tuning.

### The Stability-Recency Tradeoff

Traditional target networks stabilize training but introduce staleness, steering updates using outdated value predictions. TARL approaches this not by changing update frequencies, but by using the recent (online) network to validate the stable (target) network on a per-transition basis. By oversampling transitions and filtering for those where online and offline targets are **highly aligned**, TARL accelerates convergence and improves asymptotic performance.

---

## 📦 Files & Architecture

The release contains two primary Python modules, staying strictly faithful to the SB3 reference implementations:

* `sac_custom.py` — **Target-Aligned Soft Actor-Critic (TA-SAC)**
Implements TARL with aligned twin-critic updates and target-aware regularization.
* `dqn_custom.py` — **Target-Aligned DQN and DDQN (TA-DQN / TA-DDQN)**
Implements Target-Aligned DQN and DDQN with alignment-aware temporal-difference targets.

Both implementations are fully compatible with SB3 training loops, callbacks, vector environments, and logging.

---

## 🚀 Usage

The policies can be used as direct replacements for standard SB3 models. All standard SB3 hyperparameters remain supported, alongside a few custom TARL arguments.

### Quickstart

```python
import gymnasium as gym
from dqn_custom import CustomDQN
from sac_custom import CustomSAC

env = gym.make("CartPole-v1")

# Initialize Target-Aligned DDQN
model = CustomDQN(
    policy="MlpPolicy",
    env=env,
    target="DDQN",             # Computes offline targets using DDQN
    oversampling_ratio=1.0,    # Samples 2x the batch size for alignment filtering
    td_scale=False,            # Standard alignment-based oversampling
    verbose=1,
)

model.learn(total_timesteps=100000)

```

### ⚙️ Configuration

TARL introduces three new arguments that control the target alignment behavior. You can pass these directly into the model initialization:

* `target` *(str, CustomDQN only)*: Selects the algorithm for computing offline targets. Options are `"DQN"` or `"DDQN"`.
* `oversampling_ratio` *(float)*: The margin b by which to enlarge the batch before filtering for alignment.
* Set to `1.0` to sample m + 1.0m = 2m transitions (the default recommended setting used in the paper).
* Higher values (e.g., `3.0`) yield strictly stronger alignment gains up to your hardware's memory limits.


* `td_scale` *(bool)*: **[EXPERIMENTAL]** If `True`, the computed target alignment score is multiplied directly with the Temporal Difference (TD) error (loss scaling) rather than strictly being used as an inclusion filter for oversampling.

---

## 🔬 How it Works: Alignment-Based Oversampling

TARL acts as a dynamic filter. During the standard RL training loop, it modifies the batch sampling process:

1. **Sample:** Instead of sampling a batch of size m, TARL samples an oversized minibatch of m + b (controlled by `oversampling_ratio`).
2. **Score:** It computes a normalized *Target Alignment Score* for each transition, measuring the directional and magnitude agreement between the offline target and the online network.
3. **Filter:** It discards the b least-aligned transitions.
4. **Update:** The network performs standard gradient descent on the final, highly-aligned minibatch of size m.

This naturally defers updates on misaligned transitions until subsequent parameter updates clarify the alignment signal.

---

## 📊 Reproducibility & Intended Use

This code is released **for research purposes**. It is designed for:

* Replicating the TARL baseline experiments on Atari, MinAtar, and MuJoCo.
* Algorithmic ablations and benchmarking.
* Extension into other off-policy methods (e.g., TD3).

---

## 📖 Citation

If you use TARL or these custom policies in your academic work, please cite the accompanying paper:

```bibtex
@article{pleiss2026target,
  title={Target-Aligned Reinforcement Learning},
  author={Pleiss, Leonard S. and Harrison, James and Schiffer, Maximilian},
  journal={arXiv preprint arXiv:2603.06252},
  year={2026}
}

```

## ⚖️ License

This codebase is licensed under the MIT License, following the license structure of Stable-Baselines3.