"""Constraint evaluation for real and generated tabular data."""

from pathlib import Path

import pandas as pd
import wandb

from constraints_code.parser import parse_constraints_file
from evaluation.constraints import evaluate_constraint_satisfaction


PARTITIONS = ("train", "val", "test")


def _as_rounds(data):
    if isinstance(data, (list, tuple)):
        return data
    return [data]


def _evaluate_source(source, partitioned_data, constraints):
    per_round_rows = []
    per_constraint_rows = []

    for partition in PARTITIONS:
        if partition not in partitioned_data:
            continue

        for round_index, data in enumerate(_as_rounds(partitioned_data[partition])):
            metrics = evaluate_constraint_satisfaction(data, constraints)
            per_round_rows.append({
                "source": source,
                "partition": partition,
                "round": round_index,
                "num_rows": metrics["num_rows"],
                "mean_constraint_satisfaction": metrics["mean_constraint_satisfaction"],
                "all_constraints_satisfaction": metrics["all_constraints_satisfaction"],
                "constraints_violated_at_least_once": metrics["constraints_violated_at_least_once"],
            })

            for constraint_index, score in enumerate(metrics["individual_scores"]):
                per_constraint_rows.append({
                    "source": source,
                    "partition": partition,
                    "round": round_index,
                    "constraint": constraint_index,
                    "satisfaction_rate": float(score),
                })

    return per_round_rows, per_constraint_rows


def _aggregate_rounds(per_round):
    aggregate = (
        per_round
        .groupby(["source", "partition"], as_index=False)
        .agg(
            rounds=("round", "count"),
            rows_per_round=("num_rows", "mean"),
            mean_constraint_satisfaction=("mean_constraint_satisfaction", "mean"),
            mean_constraint_satisfaction_std=("mean_constraint_satisfaction", "std"),
            all_constraints_satisfaction=("all_constraints_satisfaction", "mean"),
            all_constraints_satisfaction_std=("all_constraints_satisfaction", "std"),
            constraints_violated_at_least_once=("constraints_violated_at_least_once", "mean"),
        )
    )
    return aggregate.fillna(0.0)


def _log_to_wandb(aggregate, per_round, per_constraint):
    scalar_metrics = {}
    for row in aggregate.to_dict(orient="records"):
        prefix = f'constraints/{row["source"]}/{row["partition"]}'
        scalar_metrics[f"{prefix}/mean_constraint_satisfaction"] = row[
            "mean_constraint_satisfaction"
        ]
        scalar_metrics[f"{prefix}/all_constraints_satisfaction"] = row[
            "all_constraints_satisfaction"
        ]
        scalar_metrics[f"{prefix}/constraints_violated_at_least_once"] = row[
            "constraints_violated_at_least_once"
        ]

    scalar_metrics["constraints/summary"] = wandb.Table(dataframe=aggregate)
    scalar_metrics["constraints/per_round"] = wandb.Table(dataframe=per_round)
    scalar_metrics["constraints/per_constraint"] = wandb.Table(dataframe=per_constraint)
    wandb.log(scalar_metrics)


def _save_results(args, aggregate, per_round, per_constraint):
    output_dir = Path(args.exp_path) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_dir / "constraint_summary.csv", index=False)
    per_round.to_csv(output_dir / "constraint_per_round.csv", index=False)
    per_constraint.to_csv(output_dir / "constraint_per_constraint.csv", index=False)
    return output_dir


def constraints_sat_check(
    args,
    real_data,
    generated_data,
    log_wandb=True,
    generated_label="generated",
    comparison_data=None,
):
    """Evaluate constraints for every available partition and sampling round."""
    _, constraints = parse_constraints_file(args.constraints_file)

    real_rounds, real_constraints = _evaluate_source("real", real_data, constraints)
    all_rounds = list(real_rounds)
    all_constraints = list(real_constraints)

    generated_sources = {generated_label: generated_data}
    if comparison_data:
        generated_sources.update(comparison_data)

    for source, source_data in generated_sources.items():
        source_rounds, source_constraints = _evaluate_source(
            source, source_data, constraints
        )
        all_rounds.extend(source_rounds)
        all_constraints.extend(source_constraints)

    per_round = pd.DataFrame(all_rounds)
    per_constraint = pd.DataFrame(all_constraints)
    if per_round.empty:
        raise ValueError("No real or generated data was provided for constraint evaluation.")

    aggregate = _aggregate_rounds(per_round)
    print("Constraint evaluation summary:")
    print(aggregate.to_string(index=False))
    output_dir = _save_results(args, aggregate, per_round, per_constraint)
    print(f"Constraint evaluation files written to {output_dir}")

    if log_wandb:
        _log_to_wandb(aggregate, per_round, per_constraint)

    return {
        "summary": aggregate,
        "per_round": per_round,
        "per_constraint": per_constraint,
        "output_dir": output_dir,
    }


def gen_sat_check(args, generated_data, constraints, log_wandb=True):
    """Compatibility wrapper for generated-data-only constraint evaluation."""
    per_round_rows, per_constraint_rows = _evaluate_source(
        "generated", generated_data, constraints
    )
    per_round = pd.DataFrame(per_round_rows)
    per_constraint = pd.DataFrame(per_constraint_rows)
    aggregate = _aggregate_rounds(per_round)
    if log_wandb:
        _log_to_wandb(aggregate, per_round, per_constraint)
    return aggregate, per_round, per_constraint


def real_sat_check(args, real_data, constraints, log_wandb=True):
    """Compatibility wrapper for real-data-only constraint evaluation."""
    per_round_rows, per_constraint_rows = _evaluate_source("real", real_data, constraints)
    per_round = pd.DataFrame(per_round_rows)
    per_constraint = pd.DataFrame(per_constraint_rows)
    aggregate = _aggregate_rounds(per_round)
    if log_wandb:
        _log_to_wandb(aggregate, per_round, per_constraint)
    return aggregate, per_round, per_constraint
