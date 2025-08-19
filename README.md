# Divergence Tree: Auditing Optimized Interventions

A Python package for auditing marketing interventions and identifying subpopulations where firm-side and consumer-side outcomes either align (win-win) or diverge (trade-offs).

## Overview

The Divergence Tree algorithm uses recursive partitioning to segment populations based on treatment effects on two outcomes simultaneously. It's designed to audit optimized marketing policies and identify where firm-side gains may come at the expense of consumer well-being.

### Key Features

- **Joint Outcome Analysis**: Simultaneously analyzes firm-side (e.g., conversion) and consumer-side (e.g., satisfaction) treatment effects
- **Interpretable Segmentation**: Creates rule-based segments that are transparent and actionable
- **Audit-Focused**: Designed for policy auditing rather than optimization
- **Regulatory Compliance**: Helps comply with regulations like the Digital Services Act (DSA)
- **Scalable**: Works with high-dimensional feature spaces

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/divergence-tree.git
cd divergence-tree

# Install with exact dependencies for reproducibility
pip install -r requirements.txt
pip install -e .
```

### Dependencies

The package requires:

- `numpy>=1.23`
- `pandas>=2.0`
- `matplotlib>=3.7`
- `optuna>=3.2`

## Quick Start

### Basic Example

See `examples/basic.py` for a basic example that demonstrates:

- How to generate synthetic data
- How to use hyperparameter tuning
- How to create and visualize the divergence tree

### Free Trial Example

We designed a comprehensive free trial scenario with a data generating process that includes both alignment and misalignment between firm and consumer interests. See `simulations/free_trial/example.py` to run and analyze this data using our algorithm.

## Algorithm Details

### Joint Splitting Criterion

The algorithm uses a novel splitting criterion that jointly considers heterogeneity in both firm-side (τF) and consumer-side (τC) treatment effects:

```
g(τF, τC) = (τF - τ̄F)²/σF² + (τC - τ̄C)²/σC² + λ·φ((τF - τ̄F)(τC - τ̄C)/(σF·σC))
```

Where:

- First two terms ensure variation in both outcomes
- Third term captures whether effects move together or apart
- φ() can be configured for convergence-seeking, divergence-seeking, or both

### Cross-Movement Modes

The algorithm can be configured to focus on different types of segments:

- **Convergence-seeking**: Finds win-win scenarios (τF > 0, τC > 0)
- **Divergence-seeking**: Finds trade-off scenarios (τF > 0, τC < 0 or τF < 0, τC > 0)
- **Both**: Finds all types of heterogeneity

## Visualization

The package provides rich visualization capabilities with color-coded segments:

- **Green**: Win-win scenarios (τF > 0, τC > 0)
- **Red**: Firm wins, consumer loses (τF > 0, τC < 0)
- **Blue**: Firm loses, consumer wins (τF < 0, τC > 0)
- **Gray**: Both lose (τF < 0, τC < 0)
