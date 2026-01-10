# Protein Binder Design Pipeline

A high-throughput, in-silico protein binder design pipeline built on [Modal](https://modal.com) serverless infrastructure. This pipeline maximizes the probability of wet-lab functionality by optimizing for both **Specificity** (binding to target) and **Selectivity** (avoiding off-targets).

## Overview

The pipeline operates as a Directed Acyclic Graph (DAG) with three main phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 1: Generation (Specificity)                │
├─────────────────────────────────────────────────────────────────────┤
│  Target PDB → [RFDiffusion] → Backbones → [ProteinMPNN] → Sequences │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   Phase 2: Validation (Specificity)                  │
├─────────────────────────────────────────────────────────────────────┤
│      Sequences + Target → [Boltz-2] → Validated Candidates          │
│                     (Filter by i-pLDDT and PAE)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                Phase 3: Negative Selection (Selectivity)             │
├─────────────────────────────────────────────────────────────────────┤
│  Target → [FoldSeek] → Decoys → [Chai-1] → Cross-Reactivity Check   │
│              ↓                                                       │
│  Candidates + Decoys → Filter for selectivity → Ranked Output       │
└─────────────────────────────────────────────────────────────────────┘
```

## Scoring Function

Candidates are ranked using the selection function:

$$S(x) = \alpha \cdot \text{pLDDT}_{\text{interface}}(x) - \beta \cdot \max_{d \in D} (\text{Affinity}(x, d))$$

Where:
- $\text{pLDDT}_{\text{interface}}$ is the confidence score from Boltz-2 on the target interface
- $D$ is the set of structural decoys found by FoldSeek
- $\alpha, \beta$ are weighting coefficients for specificity and selectivity

## Installation

```bash
# Install dependencies using uv
uv sync

# Authenticate with Modal
uv run modal token new
```

> **Note:** If you don't have `uv` installed, you can install it with:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

## Usage

### Command Line

```bash
# Preview deployment parameters and estimated cost (no execution)
uv run modal run pipeline.py --pdb-id 3DI3 --entity-id 2 \
    --hotspot-residues "42,64,123" \
    --num-designs 5 \
    --num-sequences 4 \
    --dry-run

# Run with Modal CLI
uv run modal run pipeline.py --pdb-id 3DI3 --entity-id 2 \
    --hotspot-residues "42,64,123" \
    --num-designs 5 \
    --num-sequences 4 \
    --max-budget 5.0

# Test with mocks (no GPU required)
uv run modal run pipeline.py --pdb-id 3DI3 --entity-id 2 \
    --hotspot-residues "42,64,123" \
    --use-mocks
```

**Input Format:**
- `--pdb-id`: 4-letter PDB code (e.g., `3DI3`)
- `--entity-id`: Polymer entity ID identifying the target chain (e.g., `2` for IL7RA receptor in 3DI3)

Use the [RCSB PDB website](https://www.rcsb.org/) to find entity IDs for your target structure.

### Python API

```python
from common import TargetProtein, PipelineConfig, initialize_target
from pipeline import run_pipeline

# Initialize target from PDB ID and entity ID
target = initialize_target(
    pdb_id="3DI3",
    entity_id=2,  # IL7RA receptor
    hotspot_residues=[42, 64, 123],
    output_dir="/tmp/target",
)

# Run pipeline
config = PipelineConfig(target=target)
result = run_pipeline.remote(config)

# Access results
print(f"Best candidate: {result.best_candidate.candidate_id}")
print(f"Sequence: {result.best_candidate.sequence}")
print(f"Score: {result.best_candidate.final_score}")
```

## Project Structure

```
├── common.py          # Pydantic models and Modal image definitions
├── generators.py      # RFDiffusion and ProteinMPNN functions
├── validators.py      # Boltz-2, FoldSeek, and Chai-1 functions
├── pipeline.py        # Main orchestrator DAG
├── requirements.txt   # Python dependencies
└── SYSTEM.md          # Detailed specification
```

## Infrastructure & Cost Optimization

The pipeline is designed to run within a **$5 USD** compute budget:

| Step | Tool | GPU | Estimated Time |
|------|------|-----|----------------|
| Backbone Generation | RFDiffusion | A10G | ~60s per design |
| Sequence Design | ProteinMPNN | L4 | ~15s per backbone |
| Structure Validation | Boltz-2 | A100 | ~120s per sequence |
| Decoy Search | FoldSeek | CPU | ~30s total |
| Cross-Reactivity | Chai-1 | A100 | ~90s per pair |

### Cost Optimization Strategies

1. **Shared Volumes**: Model weights are pre-downloaded to Modal Volumes, not fetched at runtime
2. **GPU Selection**: Cheaper GPUs (A10G, L4) used for lighter tasks, A100 reserved for memory-intensive steps
3. **Parallel Execution**: ProteinMPNN and Chai-1 use `starmap` for efficient parallelization
4. **Early Filtering**: Boltz-2 filters candidates before expensive cross-reactivity checks

## Pre-downloading Model Weights

Before running the pipeline, populate the Modal volumes with model weights:

```bash
# Create and populate the weights volume
uv run modal volume create binder-weights

# Upload RFDiffusion weights
uv run modal volume put binder-weights /path/to/rfdiffusion/weights /rfdiffusion

# Upload ProteinMPNN weights
uv run modal volume put binder-weights /path/to/proteinmpnn /ProteinMPNN

# Upload Boltz-2 weights
uv run modal volume put binder-weights /path/to/boltz2 /boltz2

# Upload Chai-1 weights
uv run modal volume put binder-weights /path/to/chai1 /chai1
```

## Configuration Options

### RFDiffusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_designs` | 10 | Number of backbones to generate |
| `binder_length_min` | 50 | Minimum binder length |
| `binder_length_max` | 100 | Maximum binder length |
| `noise_scale` | 1.0 | Diffusion noise scale |
| `num_diffusion_steps` | 50 | Number of diffusion steps |

### ProteinMPNN

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_sequences` | 8 | Sequences per backbone |
| `temperature` | 0.2 | Sampling temperature (higher = more diverse) |
| `backbone_noise` | 0.0 | Backbone coordinate noise |

### Boltz-2

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_recycles` | 3 | Recycling iterations |
| `min_iplddt` | 0.7 | Minimum interface pLDDT threshold |
| `max_pae` | 10.0 | Maximum PAE threshold |

### FoldSeek

| Parameter | Default | Description |
|-----------|---------|-------------|
| `database` | pdb100 | Database to search |
| `max_hits` | 10 | Maximum structural homologs |
| `evalue_threshold` | 1e-3 | E-value cutoff |

## Output Format

The pipeline returns a `PipelineResult` containing:

- `run_id`: Unique identifier for the run
- `candidates`: All validated binder candidates
- `top_candidates`: Top-ranked candidates (up to 10)
- `validation_summary`: Results from each pipeline stage
- `compute_cost_usd`: Estimated compute cost
- `runtime_seconds`: Total execution time

Each `BinderCandidate` includes:

- Amino acid sequence
- Backbone and sequence design metadata
- Structure prediction metrics (pLDDT, PAE, pTM)
- Cross-reactivity results for each decoy
- Specificity, selectivity, and final scores

## License

See [LICENSE](LICENSE) for details.
