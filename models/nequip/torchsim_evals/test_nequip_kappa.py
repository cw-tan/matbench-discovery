"""
Script for calculating thermal conductivity (kappa) for the PhononDB-PBE 103 structures
using a NequIP model with torch-sim batched evaluation.

Uses batched force calculations for FC2/FC3 supercells for efficiency.
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from importlib.metadata import version
from importlib.util import find_spec
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Literal

import ase.io
import pandas as pd
import torch
import torch_sim as ts
from nequip.ase import NequIPCalculator
from nequip.integrations.torchsim import NequIPTorchSimCalc
from phonon_utils import (
    calc_conductivity_from_force_sets,
    calc_force_sets_for_structure,
)
from pymatviz.enums import Key
from tqdm import tqdm

from matbench_discovery.data import DataFiles
from matbench_discovery.metrics import phonons

# import for kernels
if find_spec("openequivariance"):
    import openequivariance  # noqa: F401


# %% parse arguments
parser = argparse.ArgumentParser(
    description=(
        "Calculate thermal conductivity for PhononDB-PBE 103 structures using NequIP"
    )
)
parser.add_argument(
    "ase_model_path",
    type=str,
    help="path to ASE-compiled NequIP model (.nequip.pt2 file)",
)
parser.add_argument(
    "torchsim_model_path",
    type=str,
    help="path to torch-sim-compiled NequIP model (.nequip.pt2 file)",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="nequip-0",
    help="name for the model (default: nequip-0)",
)
parser.add_argument(
    "--ase-filter",
    type=str,
    choices=["frechet", "exp"],
    default="frechet",
    help="ASE filter to use (default: frechet)",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=300,
    help="maximum optimization steps (default: 300)",
)
parser.add_argument(
    "--fmax",
    type=float,
    default=1e-4,
    help="maximum force tolerance in eV/A (default: 1e-4)",
)
parser.add_argument(
    "--symprec",
    type=float,
    default=1e-5,
    help="symmetry precision (default: 1e-5)",
)
parser.add_argument(
    "--no-enforce-relax-symm",
    dest="enforce_relax_symm",
    action="store_false",
    help="disable enforcing symmetry during relaxation (default: enabled)",
)
parser.add_argument(
    "--no-save-forces",
    dest="save_forces",
    action="store_false",
    help="disable saving force sets (default: enabled)",
)
parser.add_argument(
    "--temperatures",
    type=float,
    nargs="+",
    default=[300],
    help="temperatures in Kelvin (default: 300)",
)
parser.add_argument(
    "--displacement-distance",
    type=float,
    default=0.03,
    help="displacement distance for phonopy (default: 0.03)",
)
parser.add_argument(
    "--no-ignore-imaginary-freqs",
    dest="ignore_imaginary_freqs",
    action="store_false",
    help="disable ignoring imaginary frequencies (default: enabled)",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default=None,
    help="output directory (default: auto-generated from model name and date)",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="run smoke test with 3 structures instead of full dataset",
)

args = parser.parse_args()

# config from args
module_dir = os.path.dirname(__file__)
ase_model_path = args.ase_model_path
torchsim_model_path = args.torchsim_model_path
model_name = args.model_name
ase_filter: Literal["frechet", "exp"] = args.ase_filter
max_steps = args.max_steps
fmax = args.fmax
symprec = args.symprec
enforce_relax_symm = args.enforce_relax_symm
save_forces = args.save_forces
temperatures: list[float] = args.temperatures
displacement_distance = args.displacement_distance
ignore_imaginary_freqs = args.ignore_imaginary_freqs
smoke_test = args.test

if not os.path.isfile(ase_model_path):
    raise FileNotFoundError(f"ASE model file not found: {ase_model_path}")
if not os.path.isfile(torchsim_model_path):
    raise FileNotFoundError(f"torch-sim model file not found: {torchsim_model_path}")


# initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading ASE calculator from: {ase_model_path}")
ase_calc = NequIPCalculator.from_compiled_model(
    compile_path=ase_model_path,
    device=device,
    chemical_species_to_atom_type_map=True,
)

print(f"Loading torch-sim model from: {torchsim_model_path}")
torchsim_model = NequIPTorchSimCalc.from_compiled_model(
    compile_path=torchsim_model_path,
    device=device,
    chemical_species_to_atom_type_map=True,
)

# create autobatcher once at start (profiled on first use, reused for all structures)
print("Initializing autobatcher for batched force calculations...")
autobatcher = ts.BinningAutoBatcher(
    model=torchsim_model,
    max_memory_padding=0.9,
    max_atoms_to_try=40_000,  # reasonable bound for NequIP OAM-L
    oom_error_message=["CUDA out of memory", "aoti_runner", "API call failed"],
)

timestamp = f"{datetime.now().astimezone():%Y-%m-%d-%H-%M-%S}"
atoms_list = ase.io.read(DataFiles.phonondb_pbe_103_structures.path, index=":")
atoms_list = sorted(atoms_list, key=len)  # sort by size for consistency

if smoke_test:
    atoms_list = atoms_list[:3]
    job_name = f"kappa-test-{timestamp}"
    print("Running smoke test with 3 structures")
else:
    job_name = f"kappa-{timestamp}"

print(f"\nJob {job_name} with {model_name} started")
print(f"Processing {len(atoms_list)} structures")

# determine output directory
if args.out_dir is None:
    out_dir = os.getenv("SBATCH_OUTPUT", f"{module_dir}/{model_name}/{job_name}")
else:
    out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# save run parameters
run_params = dict(
    ase_filter=ase_filter,
    max_steps=max_steps,
    force_max=fmax,
    symprec=symprec,
    enforce_relax_symm=enforce_relax_symm,
    temperatures=temperatures,
    displacement_distance=displacement_distance,
    save_forces=save_forces,
    smoke_test=smoke_test,
    n_structures=len(atoms_list),
    struct_data_path=DataFiles.phonondb_pbe_103_structures.path,
    versions={dep: version(dep) for dep in ("numpy", "torch", "nequip")},
)

with open(f"{out_dir}/run_params.json", mode="w") as file:
    json.dump(run_params, file, indent=4)

# === stage 1: GPU batched relaxation and force set calculation ===
print("\n" + "=" * 80)
print("STAGE 1: Relaxation and force set calculation (GPU batched)")
print("=" * 80)

stage1_dir = Path(out_dir) / "stage1"
stage1_force_sets_dir = stage1_dir / "force_sets"
stage1_relax_dir = stage1_dir / "relaxations"
stage1_force_sets_dir.mkdir(parents=True, exist_ok=True)
stage1_relax_dir.mkdir(parents=True, exist_ok=True)

for idx, atoms in enumerate(tqdm(atoms_list, desc="Stage 1: Force sets")):
    mat_id, relaxed_atoms, fc2_set, fc3_set, relax_dict, freqs_dict, err_dict = (
        calc_force_sets_for_structure(
            atoms=atoms,
            displacement_distance=displacement_distance,
            ase_optimizer="FIRE",
            max_steps=max_steps,
            force_max=fmax,
            symprec=symprec,
            enforce_relax_symm=enforce_relax_symm,
            relax_log_dir=str(stage1_relax_dir),
            task_id=idx,
            ase_calculator=ase_calc,
            torchsim_model=torchsim_model,
            autobatcher=autobatcher,
            ase_filter=ase_filter,
            ignore_imaginary_freqs=ignore_imaginary_freqs,
            formula_getter=lambda a: a.info.get("name", a.get_chemical_formula()),
        )
    )

    # save intermediate results to pickle
    stage1_data = {
        "atoms": relaxed_atoms,
        "fc2_set": fc2_set,
        "fc3_set": fc3_set,
        "relax_dict": relax_dict,
        "freqs_dict": freqs_dict,
        "err_dict": err_dict,
    }
    with open(stage1_force_sets_dir / f"{mat_id}.pkl", "wb") as f:
        pickle.dump(stage1_data, f)

print(f"\nStage 1 complete. Results saved to {stage1_force_sets_dir}")

# === stage 2: CPU parallel thermal conductivity calculation ===
print("\n" + "=" * 80)
print("STAGE 2: Thermal conductivity calculation (CPU parallel)")
print("=" * 80)


# helper function for multiprocessing
def process_conductivity(pkl_file: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Process a single structure's conductivity calculation."""
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)  # noqa: S301

    atoms = data["atoms"]
    fc2_set = data["fc2_set"]
    fc3_set = data["fc3_set"]
    freqs_dict = data["freqs_dict"]
    relax_dict = data["relax_dict"]

    # skip if no force sets computed (errors in stage 1)
    if len(fc2_set) == 0 or len(fc3_set) == 0:
        mat_id = atoms.info[Key.mat_id]
        return mat_id, {}, data["err_dict"]

    mat_id, kappa_dict, err_dict = calc_conductivity_from_force_sets(
        atoms=atoms,
        fc2_set=fc2_set,
        fc3_set=fc3_set,
        displacement_distance=displacement_distance,
        temperatures=temperatures,
        symprec=symprec,
        has_imaginary_freqs=freqs_dict.get(Key.has_imag_ph_modes, False),
        broken_symmetry=relax_dict.get("broken_symmetry", False),
        conductivity_broken_symm=False,
        ignore_imaginary_freqs=ignore_imaginary_freqs,
    )

    # merge with stage 1 results
    result_dict = relax_dict | freqs_dict | kappa_dict | err_dict

    return mat_id, result_dict, {"fc2_set": fc2_set, "fc3_set": fc3_set}


# load all stage 1 results and process in parallel
pkl_files = sorted(stage1_force_sets_dir.glob("*.pkl"))
n_cpus = int(os.getenv("SLURM_CPUS_ON_NODE", cpu_count()))
print(f"Processing {len(pkl_files)} structures with {n_cpus} CPUs")

with Pool(n_cpus) as pool:
    results = list(
        tqdm(
            pool.imap_unordered(process_conductivity, pkl_files),
            total=len(pkl_files),
            desc="Stage 2: Conductivity",
        )
    )

# collect results
kappa_results: dict[str, dict[str, Any]] = {}
force_results: dict[str, dict[str, Any]] = {}

for mat_id, result_dict, force_dict in results:
    kappa_results[mat_id] = result_dict
    if save_forces and force_dict:
        force_results[mat_id] = force_dict

# save final results
df_kappa = pd.DataFrame(kappa_results).T
df_kappa.index.name = Key.mat_id
df_kappa.reset_index().to_json(f"{out_dir}/kappa.json.gz")

if save_forces:
    df_force = pd.DataFrame(force_results).T
    df_force = pd.concat([df_kappa, df_force], axis=1)
    df_force.index.name = Key.mat_id
    df_force.reset_index().to_json(f"{out_dir}/force-sets.json.gz")

print(f"\nAll results saved to {out_dir}")

# === stage 3: compute metrics against DFT reference ===
print("\n" + "=" * 80)
print("STAGE 3: Compute metrics against PhononDB reference")
print("=" * 80)

df_ml = pd.read_json(f"{out_dir}/kappa.json.gz").set_index(Key.mat_id)
df_dft = pd.read_json(DataFiles.phonondb_pbe_103_kappa_no_nac.path).set_index("mp_id")
df_ml_metrics = phonons.calc_kappa_metrics_from_dfs(df_ml, df_dft)

kappa_srme = df_ml_metrics[Key.srme].mean()
kappa_sre = df_ml_metrics[Key.sre].mean()

print("\nMetrics:")
print(f"  SRME: {kappa_srme:.4f}")
print(f"  SRE:  {kappa_sre:.4f}")

# save metrics to JSON
metrics = {"srme": float(kappa_srme), "sre": float(kappa_sre)}
with open(f"{out_dir}/metrics.json", mode="w") as file:
    json.dump(metrics, file, indent=4)
