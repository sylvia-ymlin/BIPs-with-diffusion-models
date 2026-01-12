# Generative Diffusion Priors in Bayesian Inverse Problems: A Review of Monte Carlo Sampling Theory and Practice

**Author:** Y. Lin (Draft for KTH PhD Application Context)  
**Date:** January 2026

---

## Abstract

This review explores the convergence of Generative AI and Bayesian Statistics, specifically identifying how Diffusion Models can serve as rigorous priors for solving ill-posed Inverse Problems (IPs). While current mainstream methods like Diffusion Posterior Sampling (DPS) demonstrate impressive visual capabilities, they often rely on heuristic approximations (e.g., Tweedie's formula) that introduce uncontrolled bias and fail to provide reliable uncertainty quantification. This document synthesizes the field's evolution—from classical regularization to score-based generative models—and highlights the "Statistical Turn" led by researchers like Jimmy Olsson. We focus on Sequential Monte Carlo (SMC), Twisting mechanisms, and the Divide-and-Conquer Posterior Sampling (DCPS) framework as the rigorous path forward for high-stakes scientific reconstruction.

---

## 1. Introduction: The Core Scientific Problem

### 1.1 From Observation to Reality
The fundamental challenge addressed here is the probabilistic solution of **Ill-posed Inverse Problems**. In domains ranging from medical imaging (CT, MRI) to astrophysics (Black Hole imaging), we rarely observe the object of interest $\mathbf{x}$ directly. Instead, we observe a noisy, transformed version $\mathbf{y}$:

$$ \mathbf{y} = \mathcal{A}(\mathbf{x}) + \mathbf{n} $$

where $\mathcal{A}$ is a forward operator (often non-invertible) and $\mathbf{n}$ is noise. The problem is "ill-posed" because information is lost; infinitely many $\mathbf{x}$ could explain $\mathbf{y}$. To select the physically plausible solution, we require **Prior Knowledge**.

### 1.2 The Evolution of Priors: A History of Paradigm Shifts
The definition of "Prior" has undergone four distinct paradigm shifts:

1.  **Analytical Regularization (The Era of Smoothness):**
    Mathematical norms like Tikhonov ($\ell_2$) or Total Variation (TV) penalized roughness. While stable, these produced "cartoon-like" reconstructions lacking texture.
2.  **Compressed Sensing (The Era of Sparsity):**
    Donoho and Candès proved that signals sparse in a transform domain (e.g., Wavelets) could be recovered from sub-Nyquist samples. This established that *structure* can substitute for *data*.
3.  **Deep Image Prior & Plug-and-Play (The Era of Implicit Priors):**
    Deep Learning introduced the idea that network architecture itself (DIP) or pre-trained denoisers (PnP) could act as implicit priors, moving beyond handcrafted mathematics to data-driven features.
4.  **Score-Based Diffusion Models (The Current Era):**
    Diffusion models (SDEs) allow us to learn the full distribution gradient (Score Function) $\nabla \log p(\mathbf{x})$. Unlike GANs, they cover the full support of the data distribution, enabling theoretically grounded Bayesian inference without mode collapse.

---

## 2. Theoretical Framework: Diffusion as a Stochastic Prior

### 2.1 The Forward SDE
We model the data distribution $p_{\text{data}}(\mathbf{x})$ by slowly destroying it with noise via a forward Stochastic Differential Equation (SDE):

$$ d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)dt + g(t) d\mathbf{w} $$

As $t \to T$, the distribution converges to a simple noise prior $\mathcal{N}(0, \mathbf{I})$.

### 2.2 The Reverse Generative SDE
Generating data corresponds to reversing time. Anderson (1982) proved the existence of a reverse SDE:

$$ d\mathbf{x}_t = [\mathbf{f}(\mathbf{x}_t, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}} $$

A neural network $s_\theta(\mathbf{x}_t, t)$ is trained via Denoising Score Matching to approximate the score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$.

### 2.3 The Bayesian Inversion Challenge
For inverse problems, we must sample from the **conditional** posterior $p(\mathbf{x}_0 | \mathbf{y})$. This requires the *conditional* score, which decomposes via Bayes' rule:

$$ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{y}) = \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)}_{\text{Prior (Known)}} + \underbrace{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} | \mathbf{x}_t)}_{\text{Likelihood (Intractable)}} $$

The second term—the **Likelihood Score**—is the central difficulty. It represents the gradient of the log-likelihood of the noisy state $\mathbf{x}_t$. Since $\mathbf{y}$ depends on $\mathbf{x}_0$, computing $p(\mathbf{y}|\mathbf{x}_t)$ requires integrating over all possible $\mathbf{x}_0$ paths, which is analytically intractable.

---

## 3. State of the Art: Heuristics vs. Rigor

### 3.1 The Mainstream Approach: DPS and Tweedie's Approximation
The current dominant method, **Diffusion Posterior Sampling (DPS)**, bypasses the intractable integral using **Tweedie's Formula**. It estimates the expected clean image $\hat{\mathbf{x}}_0(\mathbf{x}_t) = \mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t]$ and evaluates the likelihood at this point estimate:

$$ \nabla_{\mathbf{x}_t} \log p(\mathbf{y} | \mathbf{x}_t) \approx \nabla_{\mathbf{x}_t} \| \mathbf{y} - \mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t)) \|^2 $$

**Critique:** While computationally efficient, this is statistically flawed. By Jensen's inequality, likelihood of the mean $\neq$ mean of the likelihood. In high-noise regimes (early diffusion steps), $\hat{\mathbf{x}}_0$ is a blurry average, leading to gradients that are either vanishing or misleading. This results in **uncontrolled bias** and "hallucinations"—features generated to satisfy the gradient but unsupported by the data.

### 3.2 The Rigorous Alternative: Sequential Monte Carlo (SMC)
To solve the bias problem, we turn to **Sequential Monte Carlo (SMC)**, a framework championed by Jimmy Olsson's group.

SMC treats the diffusion inverse problem as a **Feynman-Kac model**. Instead of a single trajectory, we maintain a population of $N$ particles. At each timestep, we perform:
1.  **Mutation:** Propagate particles via the reverse SDE (Prior).
2.  **Reweighting:** Assign weights based on compatibility with $\mathbf{y}$ (Likelihood).
3.  **Resampling:** Multiply high-weight particles and kill low-weight ones to focus computation.

SMC is **asymptotically exact**: as $N \to \infty$, the samples converge to the true posterior, providing valid uncertainty quantification.

---

## 4. Key Methodology: Divide-and-Conquer Posterior Sampling (DCPS)

A major challenge in SMC is **Weight Degeneracy**: in high dimensions, standard particle filters collapse because particles blindly explore space and rarely hit the data manifold.

Olsson et al. (NeurIPS 2024) proposed **DCPS** to address this.

### 4.1 The Core Idea: "Bridging the Gap"
DCPS avoids the impossible jump from Prior to Posterior by introducing a sequence of **Intermediate Posteriors** $\pi_k$. Using a tempering schedule $\lambda_k \in [0, 1]$:
$$ \pi_k(\mathbf{x}) \propto p(\mathbf{x}) \cdot p(\mathbf{y}|\mathbf{x})^{\lambda_k} $$
The algorithm solves a sequence of "short-distance" inverse problems, guiding the particle cloud smoothly from the unconditional prior ($\lambda=0$) to the data-constrained posterior ($\lambda=1$).

### 4.2 The Unified Theory: "Twisting"
Recent theoretical work unifies these methods under the concept of **Twisting**. We modify the transition kernel $M_t$ with a potential function $\psi_t(\mathbf{x})$:
$$ \tilde{K}(\mathbf{x}_{t-1}|\mathbf{x}_t) \propto M(\mathbf{x}_{t-1}|\mathbf{x}_t) \cdot \psi_t(\mathbf{x}_{t-1}) $$
*   In **DPS**, $\psi_t$ is implicitly a greedy Gaussian approximation (Tweedie).
*   In **SMC**, $\psi_t$ can be learned or estimated to be the "Expected Future Likelihood".
This perspective frames the inverse problem as an **Optimal Control** problem: finding the steering policy $\psi_t$ that minimizes the KL divergence between the sampling path and the true posterior path.

---

## 5. Future Directions

### 5.1 Variational Learning of Twisting Functions
Fixed analytical twisting functions (like in DPS) are insufficient for complex, non-linear operators (e.g., Phase Retrieval). A promising direction is **Recursive Variational Learning**, where the twisting potentials $\psi_t$ are parameterized (e.g., as neural networks) and learned online or offline to minimize variance.

### 5.2 Mixture Models for Multi-modal Ambiguity
Gaussian approximations fail when the posterior is multi-modal (e.g., an occluded object could be 'A' or 'B'). Future algorithms should incorporate **Mixture Models** (GMMs) within the twisting mechanism to track multiple hypotheses simultaneously, fully leveraging the population-based nature of SMC.

### 5.3 Towards "Trustworthy AI" in Science
The ultimate goal is to move from "Perceptual Quality" (Image looks good) to "Distributional Correctness" (Statistics are correct). For medical and scientific applications, the rigorous quantification of uncertainty—telling the doctor "there is a 30% chance this spot is a tumor and 70% it's noise"—is as valuable as the image itself.

---

## Bibliography & Key References

1.  **Ho et al. (2020)**. *Denoising Diffusion Probabilistic Models*. (The Prior)
2.  **Song et al. (2021)**. *Score-Based Generative Modeling through Stochastic Differential Equations*. (The SDE Framework)
3.  **Chung et al. (2023)**. *Diffusion Posterior Sampling for General Noisy Inverse Problems*. (The Baseline DPS)
4.  **Janati, Moufad, Olsson (2024)**. *Divide-and-Conquer Posterior Sampling for Denoising Diffusion Priors*. (The Method: DCPS)
5.  **Chopin & Papaspiliopoulos**. *An Introduction to Sequential Monte Carlo*. (The Theory: SMC)
6.  **Olsson et al. (2025)**. *Bridging diffusion posterior sampling and Monte Carlo methods*. (The Unification: Twisting)
