# Structured Kolmogorov-Arnold Neural ODEs (SKANODE)
 
This repository contains the code and data for the following work:
* Wei Liu, Kiran Bacsa, Loon Ching Tang, and Eleni Chatzi (2025). [Structured Kolmogorov-Arnold Neural ODEs for Interpretable Learning and Symbolic Discovery of Nonlinear Dynamics](https://arxiv.org/abs/2506.18339).

## Repository Overview
 * `saved_models` - Trained models used for comparative analysis.
   * `anode` - Trained models for Augmented Neural ODE.
   * `sonode` - Trained models for Second-Order Neural ODE.
   * `s3node` - Trained models for Structured State-Space Neural ODE.
   * `skanode` -Trained models for Structured Kolmogorov-Arnold Neural ODE.
   * `kan` - Fitted Kolmogorov-Arnold Network within SKANODE.
 * `duffing_data.mat` - Simulated data for the Duffing oscillator example.
 * `duffing_skanode.py` - SKANODE model.
 * `plot_figures.py` - Visualizes results as in the paper.

## Citation
Please cite the following paper if you find the work relevant and useful in your research:
```
@article{liu2025structured,
  title={Structured Kolmogorov-Arnold Neural ODEs for Interpretable Learning and Symbolic Discovery of Nonlinear Dynamics},
  author={Liu, Wei and Bacsa, Kiran and Tang, Loon Ching and Chatzi, Eleni},
  journal={arXiv preprint arXiv:2506.18339},
  year={2025}
}
```
