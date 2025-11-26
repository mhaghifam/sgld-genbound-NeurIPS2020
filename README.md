# Information-Theoritic Generalization Bounds for SGLD 

This repo contains the code for the numerical results presented in the following papers. 

1- **"Information-Theoretic Generalization Bounds for SGLD via Data-Dependent Estimates"** <br>
**Published at NeurIPS'19**<br>
[https://arxiv.org/abs/1911.02151] <br>
by Jeffrey Negrea*, Mahdi Haghifam*, Gintare Karolina Dziugaite, Ashish Khisti, Daniel M. Roy


2- **"Sharpened Generalization Bounds based on Conditional Mutual Information and an Application to Noisy, Iterative Algorithms"** <br>
**Published at NeurIPS'20**<br>
[https://arxiv.org/abs/2004.12983] <br>
by Mahdi Haghifam, Jeffrey Negrea, Ashish Khisti, Daniel M Roy, Gintare Karolina Dziugaite

We only consider the full-batch SGLD, i.e., LD, for simplicity here. 

# Implementation Details

This repository contains a compact simulation framework for **tracking and comparing generalization bounds during training** of neural networks on standard vision datasets.

Each dataset has:
- a **main script** (`main-*.py`) that runs repeated experiments and plots results, and
- a matching **bounds module** (`*_bounds.py`) that implements the training loop and the bound estimators.



---

## What this code does

For a given dataset/model pair, the code:

1. **Trains a network with noisy SGD** (noise schedule is built in).
2. **Measures empirical generalization error** over training iterations.
3. **Computes three bound estimates** alongside training:
   - **Incoherence-style bound** from [https://arxiv.org/abs/1911.02151]
   - **Gradient-norm bound** from [https://arxiv.org/abs/1902.00621]
   - **Conditional Mutual Information (CMI) bound** from [[https://arxiv.org/abs/2004.129](https://arxiv.org/abs/2004.12983)]
4. Repeats training across multiple random runs.
5. Plots **mean ± std** of the estimated generalization curves.

The result is a clear, iteration-by-iteration view of **how different theoretical predictors track real generalization in practice**.

---


## Contact

For questions and feedback:
- **Mahdi Haghifam** - [haghifam.mahdi@gmail.com](mailto:haghifam.mahdi@gmail.com)



## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

