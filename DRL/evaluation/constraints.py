"""Constraint-satisfaction metrics used by the synthesizers."""

from pathlib import Path

import numpy as np
import torch

from constraints_code.parser import parse_constraints_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _to_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features.detach().cpu()

    if hasattr(features, "to_numpy"):
        features = features.to_numpy()

    return torch.as_tensor(np.asarray(features), dtype=torch.float32)


def evaluate_constraint_satisfaction(features, constraints):
    """Evaluate parsed constraints against a feature matrix.

    Rates are returned in the interval [0, 1]. The function runs on CPU because
    it is intended for post-training evaluation rather than the differentiable
    constraint layer used during training.
    """
    feature_tensor = _to_feature_tensor(features)

    if feature_tensor.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional feature matrix, got shape {tuple(feature_tensor.shape)}."
        )
    if feature_tensor.shape[0] == 0:
        raise ValueError("Cannot evaluate constraints on an empty feature matrix.")
    if not constraints:
        raise ValueError("Cannot evaluate an empty constraint set.")

    satisfaction = torch.stack(
        [
            constraint.disjunctive_inequality.check_satisfaction(feature_tensor)
            for constraint in constraints
        ],
        dim=1,
    )
    individual_scores = satisfaction.float().mean(dim=0)

    return {
        "num_rows": feature_tensor.shape[0],
        "num_constraints": len(constraints),
        "mean_constraint_satisfaction": individual_scores.mean().item(),
        "all_constraints_satisfaction": satisfaction.all(dim=1).float().mean().item(),
        "constraints_violated_at_least_once": (~satisfaction.all(dim=0)).float().mean().item(),
        "individual_scores": individual_scores.numpy(),
    }


def constraint_satisfaction(features, use_case):
    """Return aggregate, all-constraints, and per-constraint satisfaction rates."""
    feature_tensor = _to_feature_tensor(features)

    constraints_file = REPOSITORY_ROOT / "data" / use_case / f"{use_case}_constraints.txt"
    if not constraints_file.exists():
        raise FileNotFoundError(f"Constraint file not found: {constraints_file}")

    ordering, constraints = parse_constraints_file(str(constraints_file))
    required_columns = max(variable.id for variable in ordering) + 1
    if feature_tensor.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional feature matrix, got shape {tuple(feature_tensor.shape)}."
        )
    if feature_tensor.shape[1] < required_columns:
        raise ValueError(
            f"Constraint file requires at least {required_columns} columns, "
            f"but received {feature_tensor.shape[1]}."
        )

    metrics = evaluate_constraint_satisfaction(feature_tensor, constraints)
    return (
        metrics["mean_constraint_satisfaction"],
        metrics["all_constraints_satisfaction"],
        metrics["individual_scores"],
    )
