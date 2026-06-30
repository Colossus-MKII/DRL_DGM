"""Minimal constraint evaluation shim.

The upstream repository imports ``evaluation.constraints`` from the CTGAN,
TVAE, and WGAN implementations, but this file is missing from the published
snapshot. CTGAN's core training path does not need this helper for the default
``unconstrained`` version, so return placeholder metrics if an optional caller
asks for them.
"""


def constraint_satisfaction(features, use_case):
    return float("nan"), float("nan"), float("nan")
