"""exp7.3: Task-Conditioned Topology Learning."""
from .task_features import TaskFeatures, extract_features, features_to_topology_hints
from .topology_controller_v2 import TopologyControllerV2, ConformalAdvantageGate, MutationRecord
from .shadow_transfer import compute_shadow_transfer, ShadowTransferResult, sweep_shadow_batch_sizes
from .conditions import run_lgae_adaptive_v2
from .experiment_runner import run_exp7_3, Exp73Result

__all__ = [
    "TaskFeatures", "extract_features", "features_to_topology_hints",
    "TopologyControllerV2", "ConformalAdvantageGate", "MutationRecord",
    "compute_shadow_transfer", "ShadowTransferResult", "sweep_shadow_batch_sizes",
    "run_lgae_adaptive_v2",
    "run_exp7_3", "Exp73Result",
]
