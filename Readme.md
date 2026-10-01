# Mathematical Modeling

This repository contains laboratory works for the Mathematical Modeling course.

The repository is organized by laboratory work and currently includes the first laboratory assignment. Each laboratory focuses on implementing mathematical or computational models, analyzing the obtained results, and visualizing the results where required.

## Repository Contents

Currently, the repository contains:

- Laboratory Work 1 — Random Number Generators and Statistical Testing

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
````

Statistical testing is performed using:

* Kolmogorov test
* Pearson chi-square test

The results are visualized using density histograms and scatter plots of consecutive values.

## Technologies

* Python
* NumPy
* Matplotlib

## Running

```bash
pip install numpy matplotlib
python <lab1_file>.py
```


