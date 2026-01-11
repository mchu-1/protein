"""
state_tree.py - NetworkX-based state tree for pipeline run observability.

This module implements a directed graph representation of pipeline run state,
enabling:
- Hierarchical tracking of designs (backbone → sequence → prediction → candidate)
- Cost observability per node and aggregated by subtree
- Time observability per node and aggregated by subtree (billable seconds)
- Generation traces with timing, GPU usage, and resource metrics
- Serialization for post-run analysis and visualization

The tree structure mirrors the pipeline DAG:

    RunRoot (campaign)
    ├── Target
    │   ├── Backbone[0..N] (RFDiffusion)
    │   │   ├── Sequence[0..M] (ProteinMPNN)
    │   │   │   └── Prediction (Boltz-2)
    │   │   │       └── Candidate (if passes validation)
    │   │   └── ...
    │   └── ...
    ├── Decoy[0..K] (FoldSeek)
    └── CrossReactivity[seq_id, decoy_id] (Chai-1)

Usage:
    from state_tree import PipelineStateTree, NodeType

    tree = PipelineStateTree(run_id="01ARZ...")
    
    # Add nodes as pipeline progresses
    tree.add_target(target)
    backbone_node = tree.add_backbone(backbone, parent="target")
    tree.start_timing(backbone_node)
    # ... run RFDiffusion ...
    tree.end_timing(backbone_node, cost_usd=0.50)
    
    # Query state
    tree.get_subtree_cost("target")      # Total cost under target node
    tree.get_subtree_timing("target")    # Total billable seconds under target
    tree.get_generation_trace("seq_abc123")
    
    # Export
    tree.to_json("/path/to/state.json")
    tree.to_graphviz("/path/to/graph.dot")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import networkx as nx


# =============================================================================
# Node Types and Enums
# =============================================================================


class NodeType(str, Enum):
    """Types of nodes in the pipeline state tree."""
    
    ROOT = "root"              # Campaign root node
    TARGET = "target"          # Target protein
    BACKBONE = "backbone"      # RFDiffusion backbone design
    SEQUENCE = "sequence"      # ProteinMPNN sequence design
    PREDICTION = "prediction"  # Boltz-2 structure prediction
    CANDIDATE = "candidate"    # Final binder candidate (passed all filters)
    DECOY = "decoy"            # FoldSeek decoy hit
    CROSS_REACTIVITY = "cross_reactivity"  # Chai-1 result


class NodeStatus(str, Enum):
    """Status of a node in the pipeline."""
    
    PENDING = "pending"        # Not yet processed
    RUNNING = "running"        # Currently being processed
    COMPLETED = "completed"    # Successfully completed
    FAILED = "failed"          # Failed during processing
    FILTERED = "filtered"      # Removed by quality filters
    SKIPPED = "skipped"        # Skipped due to limits


class StageType(str, Enum):
    """Pipeline stages for cost attribution."""
    
    RFDIFFUSION = "rfdiffusion"
    PROTEINMPNN = "proteinmpnn"
    BOLTZ2 = "boltz2"
    FOLDSEEK = "foldseek"
    CHAI1 = "chai1"
    SCORING = "scoring"


# =============================================================================
# Node Data Classes
# =============================================================================


@dataclass
class TimingTrace:
    """Timing information for a node."""
    
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_sec: Optional[float] = None
    
    def start(self) -> None:
        self.start_time = time.time()
    
    def end(self) -> None:
        self.end_time = time.time()
        if self.start_time is not None:
            self.duration_sec = self.end_time - self.start_time


@dataclass
class CostTrace:
    """Cost tracking for a node.
    
    Tracks both actual (estimated_cost_usd, gpu_seconds, etc.) and ceiling values.
    - Actual: Derived from real execution timing
    - Ceiling: Pre-computed worst-case based on timeouts and limits
    """
    
    stage: Optional[StageType] = None
    gpu_type: Optional[str] = None
    gpu_seconds: float = 0.0
    cpu_core_seconds: float = 0.0
    memory_gib_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    
    # Ceiling values (passive observability - worst-case from config)
    ceiling_cost_usd: float = 0.0
    ceiling_timing_sec: float = 0.0
    
    # Modal pricing constants
    _GPU_COST_PER_SEC = {
        "H100": 0.001097,
        "A100-80GB": 0.000694,
        "A100": 0.000583,
        "L40S": 0.000542,
        "A10G": 0.000306,
        "L4": 0.000222,
        "T4": 0.000164,
    }
    _CPU_COST_PER_CORE_SEC = 0.0000131
    _MEMORY_COST_PER_GIB_SEC = 0.00000222
    
    def calculate_cost(self, duration_sec: float, gpu: Optional[str] = None,
                       cpu_cores: int = 2, memory_gib: int = 8) -> float:
        """Calculate cost based on resource usage."""
        self.gpu_seconds = duration_sec if gpu else 0.0
        self.cpu_core_seconds = duration_sec * cpu_cores
        self.memory_gib_seconds = duration_sec * memory_gib
        self.gpu_type = gpu
        
        # GPU cost
        gpu_cost = 0.0
        if gpu and gpu in self._GPU_COST_PER_SEC:
            gpu_cost = self._GPU_COST_PER_SEC[gpu] * duration_sec
        
        # CPU cost
        cpu_cost = self._CPU_COST_PER_CORE_SEC * cpu_cores * duration_sec
        
        # Memory cost
        mem_cost = self._MEMORY_COST_PER_GIB_SEC * memory_gib * duration_sec
        
        self.estimated_cost_usd = gpu_cost + cpu_cost + mem_cost
        return self.estimated_cost_usd


@dataclass
class NodeData:
    """Complete data for a node in the state tree."""
    
    node_id: str
    node_type: NodeType
    status: NodeStatus = NodeStatus.PENDING
    
    # Timing and cost
    timing: TimingTrace = field(default_factory=TimingTrace)
    cost: CostTrace = field(default_factory=CostTrace)
    
    # Stage-specific data
    stage: Optional[StageType] = None
    data: dict[str, Any] = field(default_factory=dict)
    
    # Lineage tracking
    parent_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "timing": asdict(self.timing),
            "cost": {
                "stage": self.cost.stage.value if self.cost.stage else None,
                "gpu_type": self.cost.gpu_type,
                "gpu_seconds": self.cost.gpu_seconds,
                "cpu_core_seconds": self.cost.cpu_core_seconds,
                "memory_gib_seconds": self.cost.memory_gib_seconds,
                "estimated_cost_usd": self.cost.estimated_cost_usd,
                "ceiling_cost_usd": self.cost.ceiling_cost_usd,
                "ceiling_timing_sec": self.cost.ceiling_timing_sec,
            },
            "stage": self.stage.value if self.stage else None,
            "data": self.data,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }


# =============================================================================
# Pipeline State Tree
# =============================================================================


class PipelineStateTree:
    """
    NetworkX-based state tree for tracking pipeline run state.
    
    Provides a hierarchical view of all designs generated during a pipeline run,
    with cost attribution and timing traces for observability.
    """
    
    # Stage resource configurations for cost calculation
    STAGE_RESOURCES = {
        StageType.RFDIFFUSION: {"gpu": "A10G", "cpu_cores": 2, "memory_gib": 16},
        StageType.PROTEINMPNN: {"gpu": "L4", "cpu_cores": 2, "memory_gib": 8},
        StageType.BOLTZ2: {"gpu": "A100-80GB", "cpu_cores": 4, "memory_gib": 32},
        StageType.FOLDSEEK: {"gpu": None, "cpu_cores": 2, "memory_gib": 8},
        StageType.CHAI1: {"gpu": "A100-80GB", "cpu_cores": 4, "memory_gib": 32},
        StageType.SCORING: {"gpu": None, "cpu_cores": 1, "memory_gib": 4},
    }
    
    def __init__(self, run_id: str, config: Optional[dict] = None):
        """
        Initialize a new pipeline state tree.
        
        Args:
            run_id: Unique identifier for this pipeline run (ULID)
            config: Optional pipeline configuration dict
        """
        self.run_id = run_id
        self.config = config or {}
        
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Create root node
        root_data = NodeData(
            node_id=run_id,
            node_type=NodeType.ROOT,
            status=NodeStatus.RUNNING,
            stage=None,
            data={"config": config} if config else {},
        )
        self.graph.add_node(run_id, **{"data": root_data})
        self.root_id = run_id
        
        # Index for fast lookups
        self._nodes_by_type: dict[NodeType, list[str]] = {t: [] for t in NodeType}
        self._nodes_by_type[NodeType.ROOT].append(run_id)
        
        # Timing for the entire run
        self._run_start_time = time.time()
    
    # =========================================================================
    # Node Creation Methods
    # =========================================================================
    
    def _add_node(
        self,
        node_id: str,
        node_type: NodeType,
        parent_id: str,
        stage: Optional[StageType] = None,
        data: Optional[dict] = None,
        status: NodeStatus = NodeStatus.PENDING,
    ) -> str:
        """Internal method to add a node to the graph."""
        node_data = NodeData(
            node_id=node_id,
            node_type=node_type,
            status=status,
            stage=stage,
            data=data or {},
            parent_id=parent_id,
        )
        
        self.graph.add_node(node_id, data=node_data)
        self.graph.add_edge(parent_id, node_id)
        self._nodes_by_type[node_type].append(node_id)
        
        return node_id
    
    def add_target(
        self,
        pdb_id: str,
        entity_id: int,
        name: Optional[str] = None,
        hotspot_residues: Optional[list[int]] = None,
    ) -> str:
        """Add target protein node."""
        node_id = f"target_{pdb_id}_E{entity_id}"
        return self._add_node(
            node_id=node_id,
            node_type=NodeType.TARGET,
            parent_id=self.root_id,
            data={
                "pdb_id": pdb_id,
                "entity_id": entity_id,
                "name": name,
                "hotspot_residues": hotspot_residues,
            },
        )
    
    def add_backbone(
        self,
        design_id: str,
        parent_id: str,
        pdb_path: Optional[str] = None,
        binder_length: Optional[int] = None,
        rfdiffusion_score: Optional[float] = None,
    ) -> str:
        """Add backbone design node from RFDiffusion."""
        return self._add_node(
            node_id=design_id,
            node_type=NodeType.BACKBONE,
            parent_id=parent_id,
            stage=StageType.RFDIFFUSION,
            data={
                "pdb_path": pdb_path,
                "binder_length": binder_length,
                "rfdiffusion_score": rfdiffusion_score,
            },
        )
    
    def add_sequence(
        self,
        sequence_id: str,
        backbone_id: str,
        sequence: Optional[str] = None,
        score: Optional[float] = None,
        fasta_path: Optional[str] = None,
    ) -> str:
        """Add sequence design node from ProteinMPNN."""
        return self._add_node(
            node_id=sequence_id,
            node_type=NodeType.SEQUENCE,
            parent_id=backbone_id,
            stage=StageType.PROTEINMPNN,
            data={
                "sequence": sequence,
                "score": score,
                "fasta_path": fasta_path,
            },
        )
    
    def add_prediction(
        self,
        prediction_id: str,
        sequence_id: str,
        pdb_path: Optional[str] = None,
        plddt_overall: Optional[float] = None,
        plddt_interface: Optional[float] = None,
        pae_interface: Optional[float] = None,
        ptm: Optional[float] = None,
        iptm: Optional[float] = None,
        pae_interaction: Optional[float] = None,
        ptm_binder: Optional[float] = None,
        rmsd_to_design: Optional[float] = None,
    ) -> str:
        """Add structure prediction node from Boltz-2."""
        return self._add_node(
            node_id=prediction_id,
            node_type=NodeType.PREDICTION,
            parent_id=sequence_id,
            stage=StageType.BOLTZ2,
            data={
                "pdb_path": pdb_path,
                "plddt_overall": plddt_overall,
                "plddt_interface": plddt_interface,
                "pae_interface": pae_interface,
                "ptm": ptm,
                "iptm": iptm,
                "pae_interaction": pae_interaction,
                "ptm_binder": ptm_binder,
                "rmsd_to_design": rmsd_to_design,
                "ppi_score": (0.8 * iptm + 0.2 * ptm) if iptm and ptm else None,
            },
        )
    
    def add_candidate(
        self,
        candidate_id: str,
        prediction_id: str,
        specificity_score: Optional[float] = None,
        selectivity_score: Optional[float] = None,
        final_score: Optional[float] = None,
    ) -> str:
        """Add final binder candidate node (passed all validations)."""
        return self._add_node(
            node_id=candidate_id,
            node_type=NodeType.CANDIDATE,
            parent_id=prediction_id,
            stage=StageType.SCORING,
            data={
                "specificity_score": specificity_score,
                "selectivity_score": selectivity_score,
                "final_score": final_score,
            },
            status=NodeStatus.COMPLETED,
        )
    
    def add_decoy(
        self,
        decoy_id: str,
        target_id: str,
        pdb_path: Optional[str] = None,
        evalue: Optional[float] = None,
        tm_score: Optional[float] = None,
        aligned_length: Optional[int] = None,
        sequence_identity: Optional[float] = None,
    ) -> str:
        """Add decoy hit node from FoldSeek."""
        node_id = f"decoy_{decoy_id}"
        return self._add_node(
            node_id=node_id,
            node_type=NodeType.DECOY,
            parent_id=target_id,
            stage=StageType.FOLDSEEK,
            data={
                "decoy_id": decoy_id,
                "pdb_path": pdb_path,
                "evalue": evalue,
                "tm_score": tm_score,
                "aligned_length": aligned_length,
                "sequence_identity": sequence_identity,
            },
        )
    
    def add_cross_reactivity(
        self,
        binder_id: str,
        decoy_id: str,
        predicted_affinity: Optional[float] = None,
        plddt_interface: Optional[float] = None,
        binds_decoy: Optional[bool] = None,
        ptm: Optional[float] = None,
        iptm: Optional[float] = None,
        chain_pair_iptm: Optional[float] = None,
    ) -> str:
        """Add cross-reactivity check result from Chai-1."""
        node_id = f"cr_{binder_id}_{decoy_id}"
        # Link to the prediction/sequence node
        return self._add_node(
            node_id=node_id,
            node_type=NodeType.CROSS_REACTIVITY,
            parent_id=binder_id,  # Parent is the sequence being tested
            stage=StageType.CHAI1,
            data={
                "decoy_id": decoy_id,
                "predicted_affinity": predicted_affinity,
                "plddt_interface": plddt_interface,
                "binds_decoy": binds_decoy,
                "ptm": ptm,
                "iptm": iptm,
                "chain_pair_iptm": chain_pair_iptm,
            },
        )
    
    # =========================================================================
    # Timing and Status Methods
    # =========================================================================
    
    def start_timing(self, node_id: str) -> None:
        """Start timing for a node."""
        if node_id in self.graph:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            node_data.timing.start()
            node_data.status = NodeStatus.RUNNING
    
    def end_timing(
        self,
        node_id: str,
        status: NodeStatus = NodeStatus.COMPLETED,
        cost_usd: Optional[float] = None,
    ) -> float:
        """
        End timing for a node and calculate cost.
        
        Returns:
            Duration in seconds
        """
        if node_id not in self.graph:
            return 0.0
        
        node_data: NodeData = self.graph.nodes[node_id]["data"]
        node_data.timing.end()
        node_data.status = status
        
        duration = node_data.timing.duration_sec or 0.0
        
        # Calculate cost if stage is known
        if node_data.stage and node_data.stage in self.STAGE_RESOURCES:
            resources = self.STAGE_RESOURCES[node_data.stage]
            node_data.cost.stage = node_data.stage
            node_data.cost.calculate_cost(
                duration_sec=duration,
                gpu=resources["gpu"],
                cpu_cores=resources["cpu_cores"],
                memory_gib=resources["memory_gib"],
            )
        
        # Override with explicit cost if provided
        if cost_usd is not None:
            node_data.cost.estimated_cost_usd = cost_usd
        
        return duration
    
    def set_status(self, node_id: str, status: NodeStatus, error: Optional[str] = None) -> None:
        """Set status for a node."""
        if node_id in self.graph:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            node_data.status = status
            if error:
                node_data.error_message = error
    
    def set_metrics(self, node_id: str, metrics: dict[str, Any]) -> None:
        """Add metrics to a node."""
        if node_id in self.graph:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            node_data.metrics.update(metrics)
    
    def set_ceiling_cost(self, node_id: str, ceiling_usd: float) -> None:
        """Set the ceiling (budget) cost for a node (passive observability)."""
        if node_id in self.graph:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            node_data.cost.ceiling_cost_usd = ceiling_usd
    
    def set_ceiling_timing(self, node_id: str, ceiling_sec: float) -> None:
        """Set the ceiling (worst-case) timing for a node (passive observability)."""
        if node_id in self.graph:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            node_data.cost.ceiling_timing_sec = ceiling_sec
    
    def set_ceilings(self, node_id: str, ceiling_cost_usd: float, ceiling_timing_sec: float) -> None:
        """Set both ceiling cost and timing for a node (passive observability)."""
        if node_id in self.graph:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            node_data.cost.ceiling_cost_usd = ceiling_cost_usd
            node_data.cost.ceiling_timing_sec = ceiling_timing_sec
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_node(self, node_id: str) -> Optional[NodeData]:
        """Get node data by ID."""
        if node_id in self.graph:
            return self.graph.nodes[node_id]["data"]
        return None
    
    def get_nodes_by_type(self, node_type: NodeType) -> list[NodeData]:
        """Get all nodes of a specific type."""
        return [
            self.graph.nodes[nid]["data"]
            for nid in self._nodes_by_type.get(node_type, [])
            if nid in self.graph
        ]
    
    def get_children(self, node_id: str) -> list[NodeData]:
        """Get direct children of a node."""
        if node_id not in self.graph:
            return []
        return [
            self.graph.nodes[child]["data"]
            for child in self.graph.successors(node_id)
        ]
    
    def get_descendants(self, node_id: str) -> list[NodeData]:
        """Get all descendants of a node (BFS)."""
        if node_id not in self.graph:
            return []
        descendants = list(nx.descendants(self.graph, node_id))
        return [self.graph.nodes[d]["data"] for d in descendants]
    
    def get_ancestors(self, node_id: str) -> list[NodeData]:
        """Get all ancestors of a node (path to root)."""
        if node_id not in self.graph:
            return []
        ancestors = list(nx.ancestors(self.graph, node_id))
        return [self.graph.nodes[a]["data"] for a in ancestors]
    
    def get_generation_trace(self, node_id: str) -> list[NodeData]:
        """
        Get the full generation trace from root to this node.
        
        Returns nodes in order: root → target → backbone → sequence → prediction → candidate
        """
        if node_id not in self.graph:
            return []
        
        path = nx.shortest_path(self.graph, self.root_id, node_id)
        return [self.graph.nodes[n]["data"] for n in path]
    
    def get_subtree_cost(self, node_id: str) -> float:
        """
        Calculate total cost of a node and all its descendants.
        
        Returns:
            Total estimated cost in USD
        """
        if node_id not in self.graph:
            return 0.0
        
        node_data: NodeData = self.graph.nodes[node_id]["data"]
        total = node_data.cost.estimated_cost_usd
        
        for descendant in nx.descendants(self.graph, node_id):
            desc_data: NodeData = self.graph.nodes[descendant]["data"]
            total += desc_data.cost.estimated_cost_usd
        
        return total
    
    def get_total_cost(self) -> float:
        """Get total estimated cost for the entire run."""
        return self.get_subtree_cost(self.root_id)
    
    def get_subtree_timing(self, node_id: str) -> float:
        """
        Calculate total billable seconds of a node and all its descendants.
        
        This returns the sum of individual node durations (billable compute seconds),
        not wall-clock time. For parallel execution, this represents what Modal bills.
        
        Returns:
            Total duration in seconds
        """
        if node_id not in self.graph:
            return 0.0
        
        node_data: NodeData = self.graph.nodes[node_id]["data"]
        total = node_data.timing.duration_sec or 0.0
        
        for descendant in nx.descendants(self.graph, node_id):
            desc_data: NodeData = self.graph.nodes[descendant]["data"]
            total += desc_data.timing.duration_sec or 0.0
        
        return total
    
    def get_total_timing(self) -> float:
        """Get total billable seconds for the entire run."""
        return self.get_subtree_timing(self.root_id)
    
    def get_cost_by_stage(self) -> dict[str, float]:
        """Get cost breakdown by pipeline stage."""
        costs: dict[str, float] = {s.value: 0.0 for s in StageType}
        
        for node_id in self.graph.nodes:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            if node_data.stage:
                costs[node_data.stage.value] += node_data.cost.estimated_cost_usd
        
        return costs
    
    def get_timing_by_stage(self) -> dict[str, float]:
        """Get total duration by pipeline stage (active observability)."""
        timings: dict[str, float] = {s.value: 0.0 for s in StageType}
        
        for node_id in self.graph.nodes:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            if node_data.stage and node_data.timing.duration_sec:
                timings[node_data.stage.value] += node_data.timing.duration_sec
        
        return timings
    
    # =========================================================================
    # Ceiling (Passive Observability) Query Methods
    # =========================================================================
    
    def get_ceiling_cost(self) -> float:
        """
        Get the pre-computed ceiling cost for the run (passive observability).
        
        This is the worst-case cost based on timeouts × max_designs, set at run start.
        Compare with get_total_cost() (active) to see actual savings.
        """
        if self.root_id in self.graph:
            root_data: NodeData = self.graph.nodes[self.root_id]["data"]
            return root_data.cost.ceiling_cost_usd
        return 0.0
    
    def get_ceiling_timing(self) -> float:
        """
        Get the pre-computed ceiling timing for the run (passive observability).
        
        This is the worst-case billable seconds based on timeouts × max_designs.
        Compare with get_total_timing() (active) to see actual time savings.
        """
        if self.root_id in self.graph:
            root_data: NodeData = self.graph.nodes[self.root_id]["data"]
            return root_data.cost.ceiling_timing_sec
        return 0.0
    
    def get_status_summary(self) -> dict[str, dict[str, int]]:
        """Get count of nodes by type and status."""
        summary: dict[str, dict[str, int]] = {}
        
        for node_type in NodeType:
            summary[node_type.value] = {s.value: 0 for s in NodeStatus}
        
        for node_id in self.graph.nodes:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            summary[node_data.node_type.value][node_data.status.value] += 1
        
        return summary
    
    def get_run_duration(self) -> float:
        """Get total run duration so far."""
        return time.time() - self._run_start_time
    
    # =========================================================================
    # Finalization
    # =========================================================================
    
    def finalize(self, success: bool = True) -> None:
        """Mark the run as complete and update root node."""
        root_data: NodeData = self.graph.nodes[self.root_id]["data"]
        root_data.timing.end_time = time.time()
        root_data.timing.duration_sec = self.get_run_duration()
        root_data.status = NodeStatus.COMPLETED if success else NodeStatus.FAILED
        
        # Aggregate metrics - includes both passive (ceiling) and active (actual)
        root_data.metrics = {
            # Active observability (actual execution)
            "total_cost_usd": self.get_total_cost(),
            "total_timing_sec": self.get_total_timing(),
            "cost_by_stage": self.get_cost_by_stage(),
            "timing_by_stage": self.get_timing_by_stage(),
            # Passive observability (pre-computed ceiling)
            "ceiling_cost_usd": self.get_ceiling_cost(),
            "ceiling_timing_sec": self.get_ceiling_timing(),
            # Savings (ceiling - actual)
            "cost_savings_usd": self.get_ceiling_cost() - self.get_total_cost(),
            "timing_savings_sec": self.get_ceiling_timing() - self.get_total_timing(),
            # Tree statistics
            "status_summary": self.get_status_summary(),
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
        }
    
    # =========================================================================
    # Serialization Methods
    # =========================================================================
    
    def to_dict(self) -> dict:
        """Convert the entire tree to a dictionary."""
        nodes = {}
        edges = []
        
        for node_id in self.graph.nodes:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            nodes[node_id] = node_data.to_dict()
        
        for source, target in self.graph.edges:
            edges.append({"source": source, "target": target})
        
        return {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "run_duration_sec": self.get_run_duration(),
            # Active observability (actual execution)
            "total_timing_sec": self.get_total_timing(),
            "total_cost_usd": self.get_total_cost(),
            # Passive observability (pre-computed ceiling)
            "ceiling_timing_sec": self.get_ceiling_timing(),
            "ceiling_cost_usd": self.get_ceiling_cost(),
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "cost_by_stage": self.get_cost_by_stage(),
                "timing_by_stage": self.get_timing_by_stage(),
                "status_summary": self.get_status_summary(),
            },
        }
    
    def to_json(self, output_path: str, indent: int = 2) -> str:
        """
        Export the tree to JSON file.
        
        Args:
            output_path: Path to output file
            indent: JSON indentation
            
        Returns:
            Path to the written file
        """
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)
        
        return output_path
    
    def to_graphviz(self, output_path: str) -> str:
        """
        Export the tree to Graphviz DOT format for visualization.
        
        Args:
            output_path: Path to output .dot file
            
        Returns:
            Path to the written file
        """
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Color scheme by node type
        colors = {
            NodeType.ROOT: "#2C3E50",
            NodeType.TARGET: "#3498DB",
            NodeType.BACKBONE: "#E74C3C",
            NodeType.SEQUENCE: "#2ECC71",
            NodeType.PREDICTION: "#9B59B6",
            NodeType.CANDIDATE: "#F39C12",
            NodeType.DECOY: "#95A5A6",
            NodeType.CROSS_REACTIVITY: "#1ABC9C",
        }
        
        status_shapes = {
            NodeStatus.COMPLETED: "box",
            NodeStatus.FAILED: "octagon",
            NodeStatus.FILTERED: "diamond",
            NodeStatus.RUNNING: "ellipse",
            NodeStatus.PENDING: "ellipse",
            NodeStatus.SKIPPED: "parallelogram",
        }
        
        lines = ["digraph PipelineState {", "  rankdir=TB;", "  node [style=filled];", ""]
        
        # Add nodes
        for node_id in self.graph.nodes:
            node_data: NodeData = self.graph.nodes[node_id]["data"]
            color = colors.get(node_data.node_type, "#FFFFFF")
            shape = status_shapes.get(node_data.status, "ellipse")
            
            # Build label
            label_parts = [node_id[:20]]
            if node_data.timing.duration_sec:
                label_parts.append(f"{node_data.timing.duration_sec:.1f}s")
            if node_data.cost.estimated_cost_usd > 0:
                label_parts.append(f"${node_data.cost.estimated_cost_usd:.3f}")
            label = "\\n".join(label_parts)
            
            lines.append(
                f'  "{node_id}" [label="{label}", '
                f'fillcolor="{color}", shape={shape}, fontcolor="white"];'
            )
        
        lines.append("")
        
        # Add edges
        for source, target in self.graph.edges:
            lines.append(f'  "{source}" -> "{target}";')
        
        lines.append("}")
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def to_networkx(self) -> nx.DiGraph:
        """Return the underlying NetworkX graph for custom analysis."""
        return self.graph.copy()
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"PipelineStateTree(run_id={self.run_id!r}, "
            f"nodes={len(self.graph.nodes)}, "
            f"cost=${self.get_total_cost():.3f})"
        )
    
    def summary(self) -> str:
        """Generate a human-readable summary of the state tree."""
        # Get passive (ceiling) and active (actual) metrics
        ceiling_cost = self.get_ceiling_cost()
        ceiling_timing = self.get_ceiling_timing()
        actual_cost = self.get_total_cost()
        actual_timing = self.get_total_timing()
        
        lines = [
            f"Pipeline State Tree: {self.run_id}",
            "=" * 60,
            "",
            "Observability Summary:",
            f"  Wall-Clock Duration: {self.get_run_duration():.1f}s",
        ]
        
        # Show passive vs active comparison if ceiling was set
        if ceiling_timing > 0 or ceiling_cost > 0:
            lines.append("")
            lines.append("  PASSIVE (Ceiling)     ACTIVE (Actual)     Savings")
            lines.append("  -----------------     ---------------     -------")
            
            time_savings = ceiling_timing - actual_timing
            time_pct = (time_savings / ceiling_timing * 100) if ceiling_timing > 0 else 0
            lines.append(f"  Time:  {ceiling_timing:>8.1f}s       {actual_timing:>8.1f}s          {time_savings:>6.1f}s ({time_pct:.0f}%)")
            
            cost_savings = ceiling_cost - actual_cost
            cost_pct = (cost_savings / ceiling_cost * 100) if ceiling_cost > 0 else 0
            lines.append(f"  Cost:  ${ceiling_cost:>7.3f}        ${actual_cost:>7.3f}          ${cost_savings:>5.3f} ({cost_pct:.0f}%)")
        else:
            lines.append(f"  Billable Time: {actual_timing:.1f}s")
            lines.append(f"  Total Cost: ${actual_cost:.3f}")
        
        lines.append("")
        lines.append("Nodes by Type:")
        for node_type in NodeType:
            count = len(self._nodes_by_type.get(node_type, []))
            if count > 0:
                lines.append(f"  {node_type.value}: {count}")
        
        lines.append("")
        lines.append("Timing by Stage (Active):")
        for stage, timing in self.get_timing_by_stage().items():
            if timing > 0:
                lines.append(f"  {stage}: {timing:.1f}s")
        
        lines.append("")
        lines.append("Cost by Stage (Active):")
        for stage, cost in self.get_cost_by_stage().items():
            if cost > 0:
                lines.append(f"  {stage}: ${cost:.3f}")
        
        lines.append("")
        lines.append("Status Summary:")
        status_summary = self.get_status_summary()
        for node_type, statuses in status_summary.items():
            non_zero = {k: v for k, v in statuses.items() if v > 0}
            if non_zero:
                status_str = ", ".join(f"{k}={v}" for k, v in non_zero.items())
                lines.append(f"  {node_type}: {status_str}")
        
        return "\n".join(lines)


# =============================================================================
# Factory Functions for Pipeline Integration
# =============================================================================


def create_state_tree(run_id: str, config: Optional[dict] = None) -> PipelineStateTree:
    """
    Create a new pipeline state tree.
    
    This is the main entry point for pipeline integration.
    
    Args:
        run_id: Unique identifier for the pipeline run
        config: Optional pipeline configuration
        
    Returns:
        Initialized PipelineStateTree
    """
    return PipelineStateTree(run_id=run_id, config=config)


def load_state_tree(json_path: str) -> PipelineStateTree:
    """
    Load a state tree from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Reconstructed PipelineStateTree
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    tree = PipelineStateTree(run_id=data["run_id"])
    
    # Reconstruct nodes
    for node_id, node_dict in data["nodes"].items():
        if node_id == tree.root_id:
            # Update root node
            root_data = tree.graph.nodes[node_id]["data"]
            root_data.status = NodeStatus(node_dict["status"])
            root_data.metrics = node_dict.get("metrics", {})
            continue
        
        # Create node data
        node_data = NodeData(
            node_id=node_id,
            node_type=NodeType(node_dict["node_type"]),
            status=NodeStatus(node_dict["status"]),
            stage=StageType(node_dict["stage"]) if node_dict.get("stage") else None,
            data=node_dict.get("data", {}),
            parent_id=node_dict.get("parent_id"),
            created_at=node_dict.get("created_at", ""),
            metrics=node_dict.get("metrics", {}),
            error_message=node_dict.get("error_message"),
        )
        
        # Restore timing
        timing = node_dict.get("timing", {})
        node_data.timing.start_time = timing.get("start_time")
        node_data.timing.end_time = timing.get("end_time")
        node_data.timing.duration_sec = timing.get("duration_sec")
        
        # Restore cost (active and passive observability values)
        cost = node_dict.get("cost", {})
        node_data.cost.stage = StageType(cost["stage"]) if cost.get("stage") else None
        node_data.cost.gpu_type = cost.get("gpu_type")
        node_data.cost.gpu_seconds = cost.get("gpu_seconds", 0.0)
        node_data.cost.cpu_core_seconds = cost.get("cpu_core_seconds", 0.0)
        node_data.cost.memory_gib_seconds = cost.get("memory_gib_seconds", 0.0)
        node_data.cost.estimated_cost_usd = cost.get("estimated_cost_usd", 0.0)
        node_data.cost.ceiling_cost_usd = cost.get("ceiling_cost_usd", 0.0)
        node_data.cost.ceiling_timing_sec = cost.get("ceiling_timing_sec", 0.0)
        
        tree.graph.add_node(node_id, data=node_data)
        tree._nodes_by_type[node_data.node_type].append(node_id)
    
    # Reconstruct edges
    for edge in data.get("edges", []):
        tree.graph.add_edge(edge["source"], edge["target"])
    
    return tree
