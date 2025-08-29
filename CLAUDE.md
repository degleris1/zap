# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zap is a GPU-accelerated security constrained optimal power flow library for differentiable electricity models. It provides tools for power system optimization, network expansion planning, and electricity market simulation using PyTorch for GPU acceleration and ADMM solvers.

## Development Setup and Commands

### Environment Setup
- Uses Poetry for dependency management
- Python 3.11+ required
- Install with: `poetry install --all-extras --with experiment`
- Activate environment: `poetry shell`

### Testing
- Test files located in `zap/tests/`
- Run tests with: `python -m pytest zap/tests/`
- Individual test files: `pytest zap/tests/test_network.py`

### Linting and Code Quality
- Uses Ruff for linting (configured in pyproject.toml with line-length = 100)
- Run linting: `ruff check .`
- Format code: `ruff format .`

### Documentation
- Sphinx documentation in `docs/` directory
- Build docs: `cd docs && make html`
- Documentation includes Jupyter notebooks and API reference

## Architecture Overview

### Core Components

**PowerNetwork (zap/network.py)**
- Central class for power system modeling
- Manages devices, constraints, and optimization problems
- Returns DispatchOutcome objects with power flows, angles, and prices

**Devices (zap/devices/)**
- Abstract device interface in `abstract.py`
- Power system components: Generator, Load, DataCenterLoad, Battery, Ground
- Transport devices: ACLine, DCLine
- Dual formulations in `dual/` subdirectory

**ADMM Solvers (zap/admm/)**
- GPU-accelerated ADMM optimization
- `ADMMSolver` and `WeightedADMMSolver` classes
- Layer abstraction in `ADMMLayer`

**Planning Module (zap/planning/)**
- Multi-period network expansion planning
- Gradient-based optimization using `GradientDescent`
- Investment and operation objectives
- Stochastic planning problems

**Conic Optimization (zap/conic/)**
- Conic programming interface
- Variable and slack device abstractions
- Bridge to standard cone solvers

### Experiment Framework

**experiments/** directory contains three main experiment types:
- `solve/`: Single-period dispatch experiments with performance benchmarking
- `plan/`: Multi-period planning experiments
- `conic_solve/`: Conic solver benchmarking

Each experiment type has:
- `launch.py`: SLURM job launcher for HPC clusters
- `runner.py`: Local experiment execution
- `config/`: YAML configuration files
- `plot/`: Analysis and plotting scripts

### Key Design Patterns

- Devices implement constraint matrices and objective functions
- Automatic differentiation through PyTorch for gradient-based methods  
- Modular solver architecture supporting CPU/GPU execution
- Configuration-driven experiments with YAML files
- Results stored in structured format for analysis

## Common Workflows

### Running Experiments
- Configure experiment in `experiments/{type}/config/{name}.yaml`
- Local execution: `python experiments/{type}/runner.py config/{name}.yaml`
- HPC execution: `python experiments/{type}/launch.py config/{name}.yaml`

### Adding New Devices
- Inherit from `AbstractDevice` in `zap/devices/abstract.py`
- Implement constraint matrices and objective functions
- Add dual formulation in `zap/devices/dual/` if needed
- Register in `zap/__init__.py`

### GPU Acceleration
- Uses PyTorch tensors throughout for automatic GPU support
- ADMM solvers automatically detect and use available GPUs
- Sparse matrix operations optimized for GPU computation