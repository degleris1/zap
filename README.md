# zap ⚡


## Documentation

See the [documentation page](https://degleris1.github.io/zap/) for installation, basic usage, and examples.

For basic usage and examples on solving network utility maximization problems, please refer to the demo notebook. 



## Developer Setup

For developers, we recommend installing [poetry](https://python-poetry.org/docs/).
Then clone the repo and install depedencies:

```zsh
git clone https://github.com/degleris1/zap.git
cd zap
poetry install --all-extras --with experiment
poetry shell  # Start a shell in a local virtual environment
```

Then run Python from the subshell.




## Reproducing Experiments

To reproduce experiments from our published work, use the [developer setup](#developer-setup).
Some experiments require a [Mosek](https://www.mosek.com/) license.




## Citation

If you use this package in published work, we ask that you cite our paper ([preprint](https://arxiv.org/abs/2410.17203)):

```bibtex
@article{degleris2024gpu,
  title={GPU Accelerated Security Constrained Optimal Power Flow},
  author={Anthony Degleris and Abbas El Gamal and Ram Rajagopal},
  journal={arXiv preprint arXiv:2404.01255},
  year={2024},
}
```

If you also used the gradient-based planning methods, please also cite our paper on planning ([preprint](https://arxiv.org/abs/2404.01255)):

```bibtex
@article{degleris2024gradient,
  title={Gradient Methods for Scalable Multi-value Electricity Network Expansion Planning},
  author={Degleris, Anthony and Gamal, Abbas El and Rajagopal, Ram},
  journal={arXiv preprint arXiv:2404.01255},
  year={2024}
}

```

If you use this package for solving network utility maximization problems, please cite our paper ([preprint](https://arxiv.org/abs/2509.10722)):

```bibtex
@article{sreekumar2025num,
  title={Large-Scale Network Utility Maximization via GPU-Accelerated Proximal Message Passing},
  author={Akshay Sreekumar and Anthony Degleris and Ram Rajagopal},
  journal={arXiv preprint arXiv:2509.10722},
  year={2025},
}
```