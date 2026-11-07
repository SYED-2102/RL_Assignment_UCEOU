# Zero-Shot Sim-to-Real Policy Transfer via Domain Randomization

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical evaluation and reference implementation of **Domain Randomization (DR)** for zero-shot sim-to-real reinforcement learning policy transfer, evaluated on a parameterized CartPole dynamics benchmark.

---

## Overview

Model-free RL policies trained in static simulation environments overfit to nominal dynamics parameters. When deployed under unobserved real-world variations (mass, length, gravity), baseline policies suffer sudden performance degradation. 

By sampling physical dynamics parameters uniformly during training, the Domain-Randomized REINFORCE policy trades peak deterministic speed for phase-space trajectory stability, retaining **82.5%** of maximum episode capacity under severe out-of-distribution dynamic shifts where baseline policies collapse.

---

## Dynamic System & Domain Randomization Formulation

The environment is modeled as a parameterized Markov Decision Process:

$$\mathcal{M}_{\xi} = (\mathcal{S}, \mathcal{A}, \mathcal{P}_{\xi}, \mathcal{R}, \gamma, \mathcal{D}_{\xi})$$

Where physical parameter vector $\xi = [m_p, m_c, l, g]^T$ is sampled uniformly per training episode from bounded parameter space $\mathcal{D}_{\xi}$:

- **Pole Mass ($m_p$):** $m_p \sim \mathcal{U}(0.02, 0.50) \text{ kg}$ *(Nominal: $0.10 \text{ kg}$)*
- **Cart Mass ($m_c$):** $m_c \sim \mathcal{U}(0.50, 2.00) \text{ kg}$ *(Nominal: $1.00 \text{ kg}$)*
- **Pole Half-Length ($l$):** $l \sim \mathcal{U}(0.10, 1.50) \text{ m}$ *(Nominal: $0.50 \text{ m}$)*
- **Gravity ($g$):** $g \sim \mathcal{U}(4.90, 14.70) \text{ m/s}^2$ *(Nominal: $9.80 \text{ m/s}^2$)*

---

## Equations of Motion

System transition dynamics $\mathcal{P}_{\xi}(s_{t+1} \mid s_t, a_t)$ for state $s = [x, \dot{x}, \theta, \dot{\theta}]^T$ are governed by non-linear physical equations:

$$\ddot{\theta} = \frac{g \sin\theta - \cos\theta \left( \frac{F + m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p} \right)}{l \left( \frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p} \right)}$$

$$\ddot{x} = \frac{F + m_p l (\dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta)}{m_c + m_p}$$

---

## Algorithm & Optimization

The REINFORCE policy gradient objective maximizes expected return across the domain parameter distribution $\mathcal{D}_{\xi}$:

$$J(\theta) = \mathbb{E}_{\xi \sim \mathcal{D}_{\xi}, \tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

Score function update with baseline return normalization:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\xi, \tau} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \hat{G}_t \right], \quad \hat{G}_t = \frac{G_t - \mu_G}{\sigma_G + \epsilon}$$

- **Network Architecture:** MLP ($4 \to 64 \to 2$), ReLU hidden activation, Softmax output.
- **Optimization:** Adam ($\alpha = 10^{-3}$, $\gamma = 0.99$).

---

## Benchmark Results

### Zero-Shot Out-of-Distribution (OOD) Transfer (Mean Survival Steps / Max 500)

| Evaluation Configuration | Random Policy | Nominal Baseline | Domain Randomized (DR) |
| :--- | :---: | :---: | :---: |
| **Nominal Config** | 18.4 | **500.0** | **500.0** |
| **Heavy Pole ($m_p = 0.45\text{ kg}$)** | 19.2 | 112.4 | **482.1** |
| **Long Pole ($l = 1.40\text{ m}$)** | 20.1 | 84.2 | **465.3** |
| **High Gravity ($g = 14.0\text{ m/s}^2$)** | 16.5 | 143.0 | **491.0** |
| **Extreme Shift ($m_p = 0.40\text{ kg}, l = 1.20\text{ m}$)** | 14.2 | 41.6 | **412.8** |

Under the **Extreme Shift** setting, the Nominal policy suffers dynamic collapse ($91.7\%$ degradation down to $41.6$ steps), while the DR policy maintains angular stability within $\pm 4^\circ$ for $412.8$ steps.

---

## Repository Structure

```text
.
├── sim2real_dr.py        # Environment, network, training loop, and visual benchmarking
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── assets/               # Generated diagnostics (learning curves, trajectory plots)
