import modal
import os
import yaml
import argparse
import subprocess
import shutil
from pathlib import Path

# 1. Define Volume for Weights
# Mount to /root/.cache to persist HuggingFace/model weights
vol = modal.Volume.from_name("boltzgen-cache", create_if_missing=True)

# 2. Define Image
# CUDA-enabled environment with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("boltzgen", "torch", "uv")
)

app = modal.App("boltzgen-nanobody-design")

# 3. Define the Remote Runner
@app.function(
    image=image,
    gpu="A100", # Defaulting to A100 for high performance
    timeout=14400, # 4 hours
    volumes={"/root/.cache": vol},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/project")]
)
def run_boltzgen_remote(config_content: str, config_filename: str, extra_args: list):
    """
    Executes boltzgen in the remote environment.
    This function mounts the current directory, executes boltzgen with the provided config/args,
    and returns the contents of the output directory.
    """
    import yaml
    
    # We operate in a separate writable workspace because the mount at /root/project is read-only
    workspace = Path("/root/workspace")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    
    print("Setting up workspace...")
    # Copy project files to workspace
    # Using subprocess for efficiency with wildcards
    subprocess.run("cp -r /root/project/* /root/workspace/", shell=True, check=True)
    
    os.chdir(workspace)
    
    # Write the (potentially modified) config file
    working_config_path = workspace / config_filename
    with open(working_config_path, "w") as f:
        f.write(config_content)
        
    print(f"Using config file: {working_config_path}")
    
    # Construct the boltzgen command
    cmd = ["boltzgen", "run", str(working_config_path)]
    cmd.extend(extra_args)
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Run boltzgen
        # We capture output to show it in the modal logs
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"BoltzGen execution failed with error: {e}")
        # We raise to signal failure to the client
        raise e

    # Collect outputs
    # BoltzGen typically creates an 'output' directory or whatever is specified in config.
    # We assume 'output' relative to workspace or check where it wrote.
    # If standard boltzgen run, it might use a default or config-specified dir.
    # We will look for 'output' directory.
    output_dir = workspace / "output"
    results = {}
    
    if output_dir.exists():
        print(f"Collecting results from {output_dir}...")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                # We return the relative path from the workspace so local reconstruction matches
                rel_path = file_path.relative_to(workspace)
                with open(file_path, "rb") as f:
                    results[str(rel_path)] = f.read()
    else:
        print("Warning: No 'output' directory found in workspace.")
        
    return results

# 4. Local Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BoltzGen Nanobody Design on Modal")
    parser.add_argument("--config", default="3di3-config.yaml", help="Path to YAML config file")
    parser.add_argument("--target", help="Override target PDB/CIF path")
    parser.add_argument("--num_designs", type=int, help="Override number of designs")
    parser.add_argument("--budget", type=int, help="Override filtering budget")
    parser.add_argument("--gpu", default="A100", help="GPU selection (T4, L4, A10G, A100) - Note: currently fixed to A100 in remote function")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.")
        exit(1)
        
    # Read config to potentially override target
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    # Handle Target Override
    if args.target:
        print(f"Overriding target with: {args.target}")
        if args.target.lower().endswith(".pdb"):
            print("Warning: You are using a PDB file. BoltzGen uses mmCIF label_asym_id (1-based). Ensure chains are correct.")
            
        # Update the first entity's file path in the config
        # This assumes the standard BoltzGen config structure
        if "entities" in config_data and isinstance(config_data["entities"], list) and len(config_data["entities"]) > 0:
            if "file" in config_data["entities"][0]:
                config_data["entities"][0]["file"]["path"] = args.target
            else:
                print("Warning: Could not find 'file' key in first entity to override target.")
        else:
            print("Warning: Config structure does not match expected 'entities' list. Target override may not work.")

    # Serialize config to string to pass to remote function
    config_content = yaml.dump(config_data)
    
    # Construct extra CLI arguments for boltzgen
    extra_args = ["--protocol", "nanobody-anything"]
    
    if args.num_designs:
        extra_args.extend(["--num_designs", str(args.num_designs)])
    
    # Note: --budget might not be a standard flag in all boltzgen versions, verify documentation if needed.
    # Assuming it is based on requirements.
    if args.budget:
        extra_args.extend(["--budget", str(args.budget)])
        
    print(f"Starting Modal app with config: {config_path.name}")
    print(f"Extra args: {extra_args}")
    
    with app.run():
        print("Submitting job to remote GPU...")
        results = run_boltzgen_remote.remote(config_content, config_path.name, extra_args)
        
        if results:
            print(f"Job completed. Received {len(results)} output files.")
            for rel_path, content in results.items():
                local_path = Path(".") / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(content)
            print("Output synced to local directory.")
        else:
            print("Job completed but returned no results.")
