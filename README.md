# concat-hash
Implementation of hash code concatenation (shared code + specific code)

## Experiments
The experiment code is in `expr_suites`
- **original**: training and testing "shared + specific" concatenated hash code, with different loss coefficients
- **upper_bound**: training on large number of target data, and testing on target data. This only uses specific code
