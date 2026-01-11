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

Based on **AlphaProteo SI 2.2** optimized metrics.

### On-Target Binder Scoring (Boltz-2)

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **min_pae_interaction** | < 1.5 Å | **Anchor Lock.** Demands at least one "perfect" atomic contact with the defined hotspots. |
| **pTM (Binder Only)** | > 0.80 | **Fold Quality.** Ensures the binder folds autonomously into a defined structure. |
| **RMSD** | < 2.5 Å | **Self-Consistency.** Ensures Boltz-2 prediction matches the RFDiffusion design. |

### Post-Processing Steps

| Step | Method | Purpose |
|------|--------|---------|
| **Cluster** | TM-score > 0.7 | **Diversify.** Select best representative per cluster. |
| **Novelty** | pyhmmer vs UniRef50 | **Ensure novelty.** Filter sequences with existing homologs. |

### Off-Target Screening (Chai-1 Single-Sequence Mode)

| Stage | Metric | Threshold | Interpretation |
|-------|--------|-----------|----------------|
| Cross-Reactivity | chain_pair_iptm | > 0.5 | Binder binds off-target (rejected) |

### Candidate Ranking

$$S(x) = \alpha \cdot \text{PPI}_{\text{target}}(x) - \beta \cdot \max_{d \in D} \text{PPI}_{\text{decoy}}(x, d)$$

Where:
- $\text{PPI}(x) = 0.8 \cdot \text{ipTM} + 0.2 \cdot \text{pTM}$
- $D$ is the set of structural decoys found by FoldSeek
- $\alpha, \beta$ are weighting coefficients (default: 1.0, 0.5)

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
    --hotspot-residues "58,80,139" \
    --mode bind \
    --dry-run

# Run with Modal CLI
uv run modal run pipeline.py --pdb-id 3DI3 --entity-id 2 \
    --hotspot-residues "58,80,139" \
    --mode bind

# Test with mocks (no GPU required)
uv run modal run pipeline.py --pdb-id 3DI3 --entity-id 2 \
    --hotspot-residues "58,80,139" \
    --mode bind \
    --use-mocks
```

**Input Format:**
- `--pdb-id`: 4-letter PDB code (e.g., `3DI3`)
- `--entity-id`: Polymer entity ID identifying the target chain (e.g., `2` for IL7RA receptor in 3DI3)
- `--mode`: Generation mode - currently supports `bind` for binder design

**Generated Protein Nomenclature:**
```
<pdb_id>_E<entity_id>_<mode>_<ulid>
```
Example: `3DI3_E2_bind_01ARZ3NDEKTSV4RRFFQ69G5FAV`

Use the [RCSB PDB website](https://www.rcsb.org/) to find entity IDs for your target structure.

### Python API

```python
from common import TargetProtein, PipelineConfig, GenerationMode, initialize_target
from pipeline import run_pipeline

# Initialize target from PDB ID and entity ID
target = initialize_target(
    pdb_id="3DI3",
    entity_id=2,  # IL7RA receptor
    hotspot_residues=[58, 80, 139],
    output_dir="/tmp/target",
)

# Run pipeline with mode
config = PipelineConfig(target=target, mode=GenerationMode.BIND)
result = run_pipeline.remote(config)

# Access results - candidate ID uses nomenclature: <pdb_id>_E<entity_id>_<mode>_<ulid>
print(f"Best candidate: {result.best_candidate.candidate_id}")
# Example: 3DI3_E2_bind_01ARZ3NDEKTSV4RRFFQ69G5FAV
print(f"Sequence: {result.best_candidate.sequence}")
print(f"Score: {result.best_candidate.final_score}")
```

## Project Structure

```
├── common.py          # Pydantic models and Modal image definitions
├── generators.py      # RFDiffusion and ProteinMPNN functions
├── validators.py      # Boltz-2, FoldSeek, and Chai-1 functions
├── pipeline.py        # Main orchestrator DAG
├── pyproject.toml     # Python dependencies and project metadata
└── SYSTEM.md          # Detailed specification
```

## Output Filesystem

Results are organized by PDB ID and entity:

```
data/
└── <PDB_ID>/
    ├── info.json                           # Entity metadata
    └── entity_<N>/
        ├── config.json                     # Hotspots, chain info
        ├── best_candidates/                # Symlinks to top results
        └── <YYYYMMDD>_<mode>_<ulid>/       # Campaign
            ├── 01_backbones/
            ├── 02_sequences/
            ├── 03_validation/{boltz,chai}/
            └── 99_metrics/scores_combined.csv
```

## Infrastructure & Cost Observability

The pipeline provides full cost observability through a state tree that tracks resource usage at every stage.

### Modal GPU Pricing (per second)

| GPU | Cost/sec | Use Case |
|-----|----------|----------|
| T4 | $0.000164 | Light inference |
| L4 | $0.000222 | ProteinMPNN |
| A10G | $0.000306 | RFDiffusion |
| L40S | $0.000542 | — |
| A100 (40GB) | $0.000583 | Boltz-2, Chai-1 |
| A100-80GB | $0.000694 | Large models |
| H100 | $0.001097 | — |

### Step Resource Allocation

| Step | Tool | GPU | Timeout | Cost Driver | Default Runs |
|------|------|-----|---------|-------------|--------------|
| Backbone Generation | RFDiffusion | A10G | 600s | `num_designs` | 2 |
| Sequence Design | ProteinMPNN | L4 | 300s | `num_designs` | 2 |
| Structure Validation | Boltz-2 | A100 | 900s | `num_designs × num_sequences` | 4 |
| Decoy Search | FoldSeek | CPU | 120s | Fixed | 1 |
| Cross-Reactivity | Chai-1 | A100 | 900s | `num_designs × num_sequences × max_decoys` | 12* |

\* Worst case assuming all sequences pass Boltz-2 validation. In practice, 60-80% are filtered.

### Cost Formula

The worst-case cost ceiling is computed as:

```
Total Cost = Σ (GPU_cost/s × timeout × runs) + CPU/memory overhead
```

**Critical insight:** Chai-1 cost scales cubically with configuration parameters. With aggressive settings (5 backbones × 4 sequences × 5 decoys = 100 pairs), worst-case Chai-1 alone exceeds $50! The defaults are tuned for cost efficiency.

### Default Configuration

The defaults use a **Pareto-optimal configuration** that minimizes expensive Chai-1 calls while maintaining design diversity:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `--num-designs` | 2 | Fewer backbones, rely on quality |
| `--num-sequences` | 2 | ProteinMPNN quality over quantity |
| `--max-decoys` | 3 | Top decoys capture most off-target risk |

#### CLI Command

```bash
# 1. Preview cost ceiling (always do this first!)
uv run modal run pipeline.py \
  --pdb-id 3DI3 \
  --entity-id 2 \
  --hotspot-residues "58,80,139" \
  --mode bind \
  --dry-run

# 2. Execute the run
uv run modal run pipeline.py \
  --pdb-id 3DI3 \
  --entity-id 2 \
  --hotspot-residues "58,80,139" \
  --mode bind
```

### Why the Defaults Work

1. **Boltz-2 as gatekeeper**: AlphaProteo thresholds (`max_pae_interaction < 1.5 Å`, `min_ptm_binder > 0.80`, `max_rmsd < 2.5 Å`) filter 60-80% of candidates *before* Chai-1, so actual cost is typically 30-50% of ceiling.

2. **Quality over quantity**: RFDiffusion and ProteinMPNN produce high-quality outputs. 2 well-designed backbones with proper hotspot contacts often outperform 10 random ones.

3. **Top decoys matter most**: FoldSeek results are sorted by TM-score. The top 2-3 structural homologs capture the primary off-target risk.

### Cost Optimization Strategies

The pipeline implements several algorithmic optimizations to minimize compute costs:

#### Built-in Optimizations

1. **Shared Volumes**: Model weights are pre-downloaded to Modal Volumes, not fetched at runtime
2. **GPU Selection**: Cheaper GPUs (A10G, L4) used for lighter tasks, A100 reserved for memory-intensive steps
3. **Parallel Execution**: ProteinMPNN and Chai-1 use `starmap` for efficient parallelization
4. **Early Filtering**: Boltz-2 filters candidates before expensive cross-reactivity checks
5. **Budget Auto-Scaling**: Pipeline automatically reduces design count if cost ceiling exceeds budget

#### Advanced Optimizations (Configurable)

6. **Backbone Quality Filter**: Filters low-quality RFDiffusion backbones before ProteinMPNN, preventing wasted downstream compute. Configure via `backbone_filter.enabled`, `backbone_filter.min_score`.

7. **Adaptive Generation with Micro-Batching**: Generates backbones in GPU-efficient micro-batches (default: 4), processes all sequences in parallel, validates in parallel, then checks if threshold met. Early termination happens *between* batches to preserve GPU throughput. Saves 40-80% of generation costs. Configure via `adaptive.enabled`, `adaptive.batch_size`, `adaptive.min_validated_candidates`.

8. **FoldSeek Caching**: Caches decoy results at the entity level. Same target = same decoys, avoiding redundant FoldSeek calls across campaigns. Configure via `foldseek.cache_results`.

9. **Tiered Decoy Checking**: Prioritizes high-risk decoys (TM > 0.7) and skips lower-risk ones for budget runs. Saves 30-50% of Chai-1 calls. Configure via `chai1.tiered_checking`, `chai1.tier1_min_tm`, `chai1.tier2_min_tm`.

| Optimization | Typical Savings | Configuration |
|--------------|-----------------|---------------|
| Backbone Filter | 10-30% overall | `backbone_filter.min_score=0.4` |
| Adaptive Generation | 40-80% generation | `adaptive.min_validated_candidates=3` |
| FoldSeek Caching | Per-run negligible, cumulative significant | `foldseek.cache_results=true` |
| Tiered Decoy Checking | 30-50% Chai-1 | `chai1.tiered_checking=true` |

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
| `num_designs` | 2 | Number of backbones to generate |
| `binder_length_min` | 50 | Minimum binder length |
| `binder_length_max` | 100 | Maximum binder length |
| `noise_scale` | 1.0 | Diffusion noise scale |
| `num_diffusion_steps` | 50 | Number of diffusion steps |

### ProteinMPNN

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_sequences` | 2 | Sequences per backbone |
| `temperature` | 0.2 | Sampling temperature (higher = more diverse) |
| `backbone_noise` | 0.0 | Backbone coordinate noise |

### Boltz-2 (AlphaProteo SI 2.2)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_recycles` | 3 | Recycling iterations |
| `max_pae_interaction` | 1.5 Å | Max PAE at hotspots (Anchor Lock) |
| `min_ptm_binder` | 0.80 | Min pTM for binder-only (Fold Quality) |
| `max_rmsd` | 2.5 Å | Max RMSD vs RFDiffusion (Self-Consistency) |

### FoldSeek

| Parameter | Default | Description |
|-----------|---------|-------------|
| `database` | pdb100 | Database to search |
| `max_hits` | 3 | Maximum structural homologs (fewer = cheaper Chai-1) |
| `evalue_threshold` | 1e-3 | E-value cutoff |
| `cache_results` | true | Cache decoy results for target reuse |

### Chai-1

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_chain_pair_iptm` | 0.5 | Threshold for cross-reactivity detection |
| `tiered_checking` | true | Use tiered decoy checking for cost savings |
| `tier1_min_tm` | 0.7 | Min TM-score for Tier 1 (highest risk, must pass) |
| `tier2_min_tm` | 0.5 | Min TM-score for Tier 2 |
| `tier2_max_decoys` | 2 | Max decoys to check in Tier 2 |

### Backbone Filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | true | Enable backbone quality filtering |
| `min_score` | 0.4 | Minimum RFDiffusion confidence score |
| `max_keep` | None | Maximum backbones to keep (None = no limit) |

### Adaptive Generation (Micro-Batching)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | true | Enable adaptive generation with early termination |
| `min_validated_candidates` | 3 | Stop when this many pass Boltz-2 |
| `batch_size` | 4 | Backbones per micro-batch (GPU-efficient, 1-16) |
| `max_batches` | 3 | Maximum micro-batches before stopping |

> **Note:** Micro-batching preserves GPU throughput by processing full batches, with early termination occurring *between* batches rather than after each backbone.

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
