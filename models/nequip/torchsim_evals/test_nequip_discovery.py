"""
Script for testing predictions of a trained NequIP model on the WBM test
dataset (256,963 structures, used as the matbench-discovery test set for
models trained on MPTrj). This torch-sim version uses batched GPU evaluation
for improved performance.
"""

import argparse
import os
from datetime import datetime
from importlib.util import find_spec
from typing import Literal

import pandas as pd
import torch
import torch_sim as ts
from nequip.integrations.torchsim import NequIPTorchSimCalc
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key

from matbench_discovery.data import DataFiles, as_dict_handler, ase_atoms_from_zip

# import for kernels
if find_spec("openequivariance"):
    import openequivariance  # noqa: F401


# %% parse arguments
parser = argparse.ArgumentParser(
    description="Test NequIP model on WBM dataset using torch-sim"
)
parser.add_argument(
    "model_path",
    type=str,
    help="path to compiled NequIP model (.nequip.pt2 file)",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="nequip-0",
    help="name for the model (default: nequip-0)",
)
parser.add_argument(
    "--cell-filter",
    type=str,
    choices=["frechet", "unit"],
    default="frechet",
    help="cell filter to use (default: frechet)",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="run quick test with 128 structures instead of full dataset",
)
# this is the cumulative number of steps for all structures (should be bumped up)
parser.add_argument(
    "--max-steps",
    type=int,
    default=500_000,
    help="maximum optimization steps (default: 500000)",
)
parser.add_argument(
    "--fmax",
    type=float,
    default=0.05,
    help="maximum force tolerance in eV/A (default: 0.05)",
)
parser.add_argument(
    "--steps-between-swaps",
    type=int,
    default=10,
    help="steps before checking convergence and swapping (default: 10)",
)
parser.add_argument(
    "--no-autobatcher",
    dest="use_autobatcher",
    action="store_false",
    default=True,
    help="disable automatic batching (default: enabled)",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default=None,
    help="output directory (default: auto-generated from model name and date)",
)
parser.add_argument(
    "--target-atoms-per-chunk",
    type=int,
    default=40000,
    help="target atoms per chunk for balanced computational load (default: 40000)",
)

args = parser.parse_args()

# config from args
model_path = args.model_path
model_name = args.model_name
cell_filter: Literal["frechet", "unit"] = args.cell_filter
smoke_test = args.test
max_steps = args.max_steps
force_max = args.fmax
steps_between_swaps = args.steps_between_swaps
use_autobatcher = args.use_autobatcher
target_atoms_per_chunk = args.target_atoms_per_chunk

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# determine output directory
module_dir = os.path.dirname(__file__)
job_timestamp = f"{datetime.now().astimezone():%Y-%m-%d-%H-%M-%S}"
if args.out_dir is None:
    suffix = "-test" if smoke_test else ""
    job_name = f"discovery{suffix}-{job_timestamp}"
    out_dir = os.getenv("SBATCH_OUTPUT", f"{module_dir}/{model_name}/{job_name}")
else:
    out_dir = args.out_dir

print(f"\nJob: {model_name}/{job_name if args.out_dir is None else out_dir}")
print(f"Timestamp: {job_timestamp}")

# === load model ===
# model compiled with `nequip-compile --mode aotinductor --target batch ...`
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NequIPTorchSimCalc.from_compiled_model(
    compile_path=model_path,
    device=device,
    chemical_species_to_atom_type_map=True,
)

# === load and prepare data ===
print(f"Loading data from {DataFiles.wbm_initial_atoms.path}")
atoms_list = ase_atoms_from_zip(DataFiles.wbm_initial_atoms.path)
atoms_list = sorted(atoms_list, key=len)  # sort by size for better batching

if smoke_test:
    atoms_list = atoms_list[:128]

mat_ids = [atoms.info[Key.mat_id] for atoms in atoms_list]

# calculate chunk boundaries by total atoms for balanced computational load
chunk_boundaries = []
start_idx = 0
cumulative_atoms = 0

for idx, atoms in enumerate(atoms_list):
    cumulative_atoms += len(atoms)

    # create chunk when we reach target atoms or at end of list
    if cumulative_atoms >= target_atoms_per_chunk or idx == len(atoms_list) - 1:
        chunk_boundaries.append((start_idx, idx + 1))
        start_idx = idx + 1
        cumulative_atoms = 0

n_chunks = len(chunk_boundaries)

# calculate chunk statistics
chunk_sizes = [end - start for start, end in chunk_boundaries]
chunk_atoms = [
    sum(len(atoms_list[i]) for i in range(start, end))
    for start, end in chunk_boundaries
]

print(f"Processing {len(atoms_list)} structures in {n_chunks} adaptive chunks:")
print(
    f"  Structures per chunk: {min(chunk_sizes)}-{max(chunk_sizes)} "
    f"(avg: {sum(chunk_sizes) / len(chunk_sizes):.0f})"
)
print(
    f"  Atoms per chunk: {min(chunk_atoms)}-{max(chunk_atoms)} "
    f"(avg: {sum(chunk_atoms) / len(chunk_atoms):.0f})"
)

# === set up workflow ===
# setup optimization
ts_cell_filter = (
    ts.CellFilter.frechet if cell_filter == "frechet" else ts.CellFilter.unit
)
convergence_fn = ts.runners.generate_force_convergence_fn(
    force_tol=force_max, include_cell_forces=True
)

# configure autobatcher once (reused across chunks for warmup efficiency)
if use_autobatcher:
    max_iterations = max_steps // steps_between_swaps
    autobatcher = ts.InFlightAutoBatcher(
        model=model,
        memory_scales_with=model.memory_scales_with,
        max_memory_scaler=None,
        max_atoms_to_try=10_000,
        max_iterations=max_iterations,
        max_memory_padding=0.9,
        oom_error_message=["CUDA out of memory", "aoti_runner", "API call failed"],
        # account for OOM error messages from AOTI, e.g.
        # RuntimeError: run_func_(...) API call failed at
        # /pytorch/torch/csrc/inductor/aoti_runner/model_container_runner.cpp
    )
else:
    autobatcher = False

# prepare output
os.makedirs(out_dir, exist_ok=True)
save_path = f"{out_dir}/predictions.json.gz"

all_results = {}

# === process in chunks ===
for chunk_idx in range(n_chunks):
    start_idx, end_idx = chunk_boundaries[chunk_idx]
    chunk_atoms = atoms_list[start_idx:end_idx]
    chunk_mat_ids = mat_ids[start_idx:end_idx]
    chunk_total_atoms = sum(len(atoms) for atoms in chunk_atoms)

    print(f"\n{'=' * 60}")
    print(
        f"Chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_atoms)} structures, "
        f"{chunk_total_atoms} atoms"
    )
    print(f"{'=' * 60}")

    # initialize state for this chunk
    batched_state = ts.initialize_state(chunk_atoms, model.device, model.dtype)

    # relax chunk (reusing the same autobatcher)
    print("Running batched relaxation...")
    final_state = ts.optimize(
        system=batched_state,
        model=model,
        optimizer=ts.Optimizer.fire,
        max_steps=max_steps,
        convergence_fn=convergence_fn,
        autobatcher=autobatcher,
        pbar=True,
        init_kwargs=dict(
            cell_filter=ts_cell_filter,
            constant_volume=False,
            hydrostatic_strain=False,
        ),
    )

    # extract results for this chunk
    relaxed_atoms_list = ts.io.state_to_atoms(final_state)
    energies = final_state.energy.detach().cpu().numpy()

    chunk_results = {}
    for mat_id, relaxed_atoms, energy in zip(
        chunk_mat_ids, relaxed_atoms_list, energies, strict=True
    ):
        try:
            relaxed_struct = AseAtomsAdaptor.get_structure(relaxed_atoms)
            chunk_results[mat_id] = {
                "structure": relaxed_struct,
                "energy": float(energy),
            }
        except Exception as exc:
            print(f"Failed to convert {mat_id}: {exc!r}")

    # accumulate results
    all_results.update(chunk_results)

    # write intermediate results after each chunk
    df_out = pd.DataFrame(all_results).T.add_prefix("nequip_")
    df_out.index.name = Key.mat_id
    df_out.reset_index().to_json(save_path, default_handler=as_dict_handler)

    print(f"\n{'=' * 60}")
    print(f"Chunk {chunk_idx + 1}/{n_chunks} complete!")
    print(f"Chunk results: {len(chunk_results)}/{len(chunk_atoms)} structures")
    print(f"Total accumulated: {len(all_results)}/{len(atoms_list)} structures")
    print(f"Saved to: {save_path}")
    print(f"{'=' * 60}")

print(f"\nAll chunks complete! Final results saved to {save_path}")
