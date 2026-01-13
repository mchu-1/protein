import modal
import os
import yaml
import argparse
import subprocess
import shutil
from pathlib import Path
import math

# Modal Pricing constants (USD per second)
PRICING = {
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "A100": 0.000694,  # Default to 80GB
    "L40S": 0.000542,
    "A10G": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
    "CPU_CORE": 0.0000131,  # Per core
    "MEMORY_GB": 0.00000222,  # Per GB
}

# 1. Define Volume for Weights
# Mount to /vol/cache to persist HuggingFace/model weights
vol = modal.Volume.from_name("boltzgen-cache", create_if_missing=True)

# 2. Define Image
# CUDA-enabled environment with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("boltzgen", "torch", "uv")
    .env({"HF_HOME": "/vol/cache/huggingface", "TORCH_HOME": "/vol/cache/torch"})
    .add_local_dir(".", remote_path="/root/project")
)

app = modal.App("boltzgen-nanobody-design")


# 3. Define the Remote Runner as a Class
@app.cls(
    image=image,
    gpu="A100",  # Defaulting to A100 for high performance
    timeout=14400,  # 4 hours
    volumes={"/vol/cache": vol},
    concurrency_limit=100,  # Allow high concurrency for 1K designs
)
class BoltzGenRunner:
    @modal.method()
    def run_batch(
        self,
        config_content: str,
        config_filename: str,
        extra_args: list,
        batch_index: int,
    ):
        """
        Executes a batch of boltzgen designs in a separate workspace.
        """
        # Create a unique workspace for this batch
        workspace = Path(f"/root/workspace_{batch_index}")
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)

        print(f"[{batch_index}] Setting up workspace at {workspace}...")

        # Copy project files to workspace
        subprocess.run(f"cp -r /root/project/* {workspace}/", shell=True, check=True)

        os.chdir(workspace)

        # Write the config file
        working_config_path = workspace / config_filename
        with open(working_config_path, "w") as f:
            f.write(config_content)

        # Construct the boltzgen command
        # We output to a local 'output' directory inside the workspace
        cmd = ["boltzgen", "run", str(working_config_path), "--output", "output"]
        cmd.extend(extra_args)

        print(f"[{batch_index}] Executing: {' '.join(cmd)}")

        try:
            # Run boltzgen
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Print last few lines of output for verification
            print(f"[{batch_index}] BoltzGen Success. Output snippet:")
            print("\n".join(process.stdout.splitlines()[-10:]))

        except subprocess.CalledProcessError as e:
            print(f"[{batch_index}] BoltzGen execution failed!")
            print(f"[{batch_index}] STDOUT:\n{e.stdout}")
            print(f"[{batch_index}] STDERR:\n{e.stderr}")
            raise e

        # Collect outputs
        output_dir = workspace / "output"
        results = {}

        if output_dir.exists():
            print(f"[{batch_index}] Collecting output files...")
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    # We rename files to ensure uniqueness across batches
                    # e.g. output/designs/0.cif -> output/designs/batch_3_0.cif
                    rel_path = file_path.relative_to(workspace)

                    # Construct new filename with batch prefix
                    # We insert 'batch_N_' before the filename
                    new_filename = f"batch_{batch_index}_{file_path.name}"
                    new_rel_path = rel_path.parent / new_filename

                    # Store content
                    results[str(new_rel_path)] = file_path.read_bytes()

            print(f"[{batch_index}] Collected {len(results)} files")
        else:
            print(f"[{batch_index}] Warning: No 'output' directory found.")

        return results


def estimate_cost(
    num_designs, batch_size, gpu_type="A100", est_time_per_design_sec=100
):
    """
    Estimates the cost and time for the run based on Modal pricing.
    """
    # Defaults / Constants
    # Assuming standard Modal instances have some CPU/RAM.
    # Usually GPU price is the dominant factor.
    # CPU/RAM cost is usually additive if using 'serverless' custom sizing,
    # but for standard GPU profiles it might be bundled or negligible compared to GPU.
    # We will use the GPU price as the primary estimator.

    price_per_sec = PRICING.get(gpu_type, PRICING["A100"])

    num_batches = math.ceil(num_designs / batch_size)

    # Total compute time = num_designs * time_per_design
    # (Assuming linear scaling inside the batch)
    total_compute_seconds = num_designs * est_time_per_design_sec

    # Wall clock time = max(time per batch)
    # Assuming perfect parallelization
    designs_per_batch = batch_size  # mostly
    wall_clock_seconds = designs_per_batch * est_time_per_design_sec

    # Add overhead for container startup / model loading (e.g. 60s per batch)
    overhead_per_batch = 60
    total_compute_seconds += num_batches * overhead_per_batch
    wall_clock_seconds += overhead_per_batch

    estimated_cost = total_compute_seconds * price_per_sec

    return {
        "gpu_type": gpu_type,
        "price_per_sec": price_per_sec,
        "total_compute_seconds": total_compute_seconds,
        "wall_clock_seconds": wall_clock_seconds,
        "estimated_cost": estimated_cost,
        "num_batches": num_batches,
    }


# 4. Local Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BoltzGen Nanobody Design on Modal (Parallel)"
    )
    parser.add_argument(
        "--config", default="3di3-config.yaml", help="Path to YAML config file"
    )
    parser.add_argument("--target", help="Override target PDB/CIF path")
    parser.add_argument(
        "--num_designs",
        type=int,
        default=100,
        help="Total number of designs to generate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Number of designs per GPU worker"
    )
    parser.add_argument("--budget", type=int, help="Override filtering budget")
    parser.add_argument(
        "--gpu",
        default="A100",
        help="GPU selection (Default: A100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost and time without running the job",
    )

    args = parser.parse_args()

    # Dry Run Logic
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN: Cost & Time Estimation")
        print("=" * 80)

        est = estimate_cost(args.num_designs, args.batch_size, args.gpu)

        print("Configuration:")
        print(f"  Designs:      {args.num_designs}")
        print(f"  Batch Size:   {args.batch_size}")
        print(f"  Parallel Work:{est['num_batches']} workers")
        print(f"  GPU Type:     {est['gpu_type']} (${est['price_per_sec']:.6f}/sec)")
        print("-" * 40)
        print("Estimates (assuming ~100s/design + 60s overhead):")
        print(f"  Wall Time:    {est['wall_clock_seconds'] / 60:.1f} minutes")
        print(f"  Total GPU Time: {est['total_compute_seconds'] / 60:.1f} minutes")
        print(f"  Total Cost:   ${est['estimated_cost']:.4f}")
        print("-" * 40)
        print(
            "Note: Actual costs may vary based on precise runtime and instance availability."
        )
        print("=" * 80 + "\n")
        exit(0)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.")
        exit(1)

    # Read config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Handle Target Override
    if args.target:
        print(f"Overriding target with: {args.target}")
        if "entities" in config_data and len(config_data["entities"]) > 0:
            if "file" in config_data["entities"][0]:
                config_data["entities"][0]["file"]["path"] = args.target

    config_content = yaml.dump(config_data)

    # Prepare Batches
    total_designs = args.num_designs
    batch_size = args.batch_size
    num_batches = math.ceil(total_designs / batch_size)

    print("\n" + "=" * 80)
    print("BoltzGen Parallel Run Configuration")
    print(f"Total Designs: {total_designs}")
    print(f"Batch Size:    {batch_size}")
    print(f"Num Batches:   {num_batches} (Parallel Workers)")
    print(f"GPU:           {args.gpu}")
    print("=" * 80 + "\n")

    # Base extra args
    base_extra_args = ["--protocol", "nanobody-anything"]
    if args.budget:
        base_extra_args.extend(["--budget", str(args.budget)])

    # Prepare inputs for map
    # We need lists of arguments for each parameter of run_batch
    configs = []
    filenames = []
    batch_args_list = []
    indices = []

    for i in range(num_batches):
        # Calculate designs for this batch (handle remainder)
        current_batch_size = min(batch_size, total_designs - i * batch_size)

        configs.append(config_content)
        filenames.append(config_path.name)

        # Args for this specific batch
        b_args = base_extra_args.copy()
        b_args.extend(["--num_designs", str(current_batch_size)])
        batch_args_list.append(b_args)

        indices.append(i)

    # Run Modal App
    with app.run():
        print("→ Submitting batches to remote cluster...")
        runner = BoltzGenRunner()

        # Use map to run in parallel
        # map returns a generator of results
        results_generator = runner.run_batch.map(
            configs, filenames, batch_args_list, indices
        )

        total_files = 0

        print("\nProcessing results as they arrive...")
        for batch_results in results_generator:
            if batch_results:
                for rel_path, content in batch_results.items():
                    # Local save
                    # We strip the 'output/' prefix if present in the remote rel_path
                    # (it was relative to workspace, so likely starts with output/)
                    # We'll save to local output/ directory
                    local_path = Path(".") / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(content)
                    total_files += 1
                print(f"✓ Saved {len(batch_results)} files from a completed batch")

        print("\n" + "=" * 80)
        print(f"Job Complete. Total files generated: {total_files}")
        print("=" * 80 + "\n")
