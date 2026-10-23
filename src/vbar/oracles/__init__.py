"""Certified fuel bounds and exact LQ values on CW transfers (see docs/oracles.md)."""

from .certificate import PrimerCertificate, certify_impulsive, hamiltonian, indirect_optimum
from .datasets import ExpertDataset, expert_dataset
from .expert import OracleAgent
from .fuel import DualBank, FuelOracle, OracleConfig, OracleSolution, bounds_for
from .lq import LQBound, LQOracle
from .minimum_time import MinimumTime, minimum_time, verify_infeasible
from .primer import dual_bound, primer

__all__ = [
    "DualBank",
    "ExpertDataset",
    "FuelOracle",
    "LQBound",
    "LQOracle",
    "MinimumTime",
    "OracleAgent",
    "OracleConfig",
    "OracleSolution",
    "PrimerCertificate",
    "bounds_for",
    "certify_impulsive",
    "dual_bound",
    "expert_dataset",
    "hamiltonian",
    "indirect_optimum",
    "minimum_time",
    "primer",
    "verify_infeasible",
]
