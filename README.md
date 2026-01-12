# Diffusion Bias Analysis Project

## Overview
This project investigates the bias in standard diffusion guidance methods (like DPS) for inverse problems and proposes a **Sequential Twisted SMC** approach (based on DCPS) to achieve asymptotically correct posterior sampling and uncertainty quantification.

## Key Features
- **"Universal Prior + Explicit Likelihood" Paradigm**: Decoupled priors (DDPM) and measurement operators.
- **Twisted SMC Solver**: Implements particle filtering with gradient-based "twisting" for high-efficiency sampling.
- **Benchmarks**: Inpainting, Super-Resolution (4x), Phase Retrieval (Non-linear), and MRI Reconstruction (k-space).
- **Industrial-Grade Validation**: Support for both MNIST (Proof-of-Concept) and CelebA-HQ/Real-World images (Production-Ready).

## Installation
```bash
pip install -r requirements.txt
pip install diffusers transformers accelerate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Running Experiments

### 1. MNIST Benchmark (Fast)
To run the standard validation suite on handwritten digits:
```bash
python experiments/run_benchmark.py
```

### 2. Real-World Validation (High-Res)
To run on high-resolution images (Astronaut, Cat, Coffee) using a pre-trained `google/ddpm-celebahq-256` model:
```bash
python experiments/run_real_world.py
```
*Note: This automatically downloads the model from HuggingFace and requires significant compute.*

## Methodology Clarification
This implementation uses **Sequential Twisted SMC**. While it shares the same theoretical foundation (Twisted Proposal Distributions) as **Divide-and-Conquer Posterior Sampling (DCPS)** [Olsson et al., 2024], we currently utilize a sequential implementations rather than the recursive divide-and-conquer tree. This preserves the statistical rigor (unbiasedness, consistent UQ) while simplifying the architecture for single-GPU workflows.
