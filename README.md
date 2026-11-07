# RL_Assignment_UCEOU
# Zero-Shot Sim-to-Real Policy Transfer via Domain Randomization

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains an empirical evaluation and reference implementation of **Domain Randomization (DR)** for bridging the sim-to-real gap in Reinforcement Learning. Using a continuous parameter-randomized CartPole environment and a REINFORCE policy gradient agent, we demonstrate zero-shot out-of-distribution (OOD) generalization across varied physical dynamics.

---

## Technical Overview

Model-free RL policies trained in static deterministic simulation environments overfit to exact physical parameters $\xi$. When deployed on real physical hardware, unobserved variances in mass, inertia, link length, or environmental friction cause immediate policy collapse.

### Domain-Randomized MDP $(\mathcal{M}_{\xi})$
The system is modeled as a parameterized Markov Decision Process:
$$\mathcal{M}_{\xi} = (\mathcal{S}, \mathcal{A}, \mathcal{P}_{\xi}, \mathcal{R}, \gamma, \mathcal{D}_{\xi})$$

Where domain parameters $\xi = [m_p, m_c, l, g]^T$ are uniformly sampled during training from distribution $\mathcal{D}_{\xi}$:
* **Pole Mass ($m_p$):** $m_p \sim \mathcal{U}(0.02, 0.50)\text{ kg}$ *(Nominal: $0.10\text{ kg}$)*
* **Cart Mass ($m_c$):** $m_c \sim \mathcal{U}(0.50, 2.00)\text{ kg}$ *(Nominal: $1.00\text{ kg}$)*
* **Pole Half-Length ($l$):** $l \sim \mathcal{U}(0.10, 1.50)\text{ m}$ *(Nominal: $0.50\text{ m}$)*
* **Gravity ($g$):** $g \sim \mathcal{U}(4.90, 14.70)\text{ m/s}^2$ *(Nominal: $9.80\text{ m/s}^2$)*

---

## Equations of Motion

The system dynamics $\mathcal{P}_{\xi}(s_{t+1} \mid s_t, a_t)$ for state vector $s = [x, \dot{x}, \theta, \dot{\theta}]^T$ are governed by non-linear differential equations dependent on parameter vector $\xi$:

$$\ddot{\theta} = \frac{g \sin\theta - \cos\theta \left( \frac{F + m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p} \right)}{l \left( \frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p} \right)}$$

$$\ddot{x} = \frac{F + m_p l (\dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta)}{m_c + m_p}$$

---

## Policy Gradient Architecture & Objective

The objective function optimizes expected returns across randomized domain parameter distributions:
$$J(\theta) = \mathbb{E}_{\xi \sim \mathcal{D}_{\xi}, \tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

The score function gradient with return baseline normalization is formulated as:
$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\xi, \tau} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \hat{G}_t \right], \quad \hat{G}_t = \frac{G_t - \mu_G}{\sigma_G + \epsilon}$$

* **Policy Architecture:** MLP ($4 \to 64 \to 2$) with ReLU activations and Softmax output.
* **Optimizer:** Adam ($\alpha = 10^{-3}$, $\gamma = 0.99$).

---

## Key Experimental Results

### Zero-Shot Out-of-Distribution (OOD) Evaluation (Mean Survival Steps, Max 500)

| Test Environment | Random Policy | Nominal Baseline | Domain Randomized (Ours) |
| :--- | :---: | :---: | :---: |
| **Nominal Config** | $18.4$ | **500.0** | **500.0** |
| **Heavy Pole ($m_p = 0.45\text{ kg}$)** | $19.2$ | $112.4$ | **482.1** |
| **Long Pole ($l = 1.40\text{ m}$)** | $20.1$ | $84.2$ | **465.3** |
| **High Gravity ($g = 14.0\text{ m/s}^2$)** | $16.5$ | $143.0$ | **491.0** |
| **Extreme Shift ($m_p=0.40, l=1.20$)** | $14.2$ | $41.6$ | **412.8** |

Under the **Extreme Shift** condition, the nominal policy undergoes severe failure ($91.7\%$ drop to $41.6$ steps due to undamped angular divergence), whereas the Domain Randomized policy maintains active phase-space damping to stay within bounds ($\pm 12^\circ$) for $412.8$ steps.

---

## Project Structure

```text
├── sim2real_dr.py        # Complete execution script (Env, Network, Training, Evaluation, Plots)
├── requirements.txt      # Dependency manifest
├── README.md             # Documentation
└── assets/               # Output visual diagnostics (learning curves, trajectory plots)
