# Mathematical Modeling

This repository contains laboratory works for the Mathematical Modeling course.

The repository is organized by laboratory work and currently includes the first and second laboratory assignments. Each laboratory focuses on implementing mathematical or computational models, analyzing the obtained results, and visualizing the results where required.

## Repository Contents

Currently, the repository contains:

- Laboratory Work 1 — Random Number Generators and Statistical Testing
- Laboratory Work 2 — Modeling of Discrete Random Variables and Statistical Testing

Additional laboratory works will be added to the repository as they are completed.

## Laboratory Work 1

### Variant 1

Implementation and statistical analysis of two pseudorandom number generators:

- Multiplicative Congruential Generator
- MacLaren-Marsaglia Generator

Parameters:

```text
N = 1000
M = 2^31
A0 = 68921
BETA = 68921
K = 48
EPS = 0.05

```

Statistical testing is performed using:

* Kolmogorov test
* Pearson chi-square test

The results are visualized using density histograms and scatter plots of consecutive values.

## Laboratory Work 2

### Variant 1

Modeling and statistical analysis of discrete random variables:

* **Bernoulli Distribution**: $Bi(1, p)$ with $p = 0.7$
* **Binomial Distribution**: $Bi(m, p)$ with $m = 5$, $p = 0.25$

Parameters & Settings:

```text
N = 1000
ALPHA = 0.05
p_ber = 0.7
m_bin = 5
p_bin = 0.25

```

### Key Implementation Details

* **Custom Pseudorandom Number Generation**: Implemented a Linear Congruential Generator (LCG) from scratch to generate uniformly distributed numbers.
* **Inverse Transform Sampling**: Modeled discrete random variables manually using cumulative distribution functions (CDF).
* **Statistical Moments**: Calculated unbiased estimates of mathematical expectation and variance, comparing them with true theoretical values.
* **Pearson's $\chi^2$ Goodness-of-Fit Test**: Custom implementation featuring automated pooling of low-frequency intervals ($E_i < 5$) and cross-validation between distributions.
* **Type I Error Analysis**: Conducted 10,000 simulations to empirically verify that the probability of Type I error converges to the significance level $\alpha = 0.05$.

### Visualizations

Side-by-side bar charts comparing theoretical probability mass functions (PMF) with empirical relative frequencies derived from generated samples.

## Technologies

* Python
* NumPy
* Matplotlib
* SciPy

## Running

```bash
pip install numpy matplotlib scipy
python <script_file>.py

