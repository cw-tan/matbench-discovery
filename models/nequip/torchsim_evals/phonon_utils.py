"""
Utility functions for phonon and thermal conductivity calculations with torch-sim.

Relaxation code copied from matbench_discovery.phonons.calc_kappa.
Force constant calculations use torch-sim batched inference.
"""

import os
import traceback
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import ase.optimize
import ase.optimize.sciopt
import numpy as np
import torch_sim as ts
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import ExpCellFilter, Filter, FrechetCellFilter
from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter
from nequip.integrations.torchsim import NequIPTorchSimCalc
from phono3py.api_phono3py import Phono3py
from pymatgen.core.structure import Structure
from pymatviz.enums import Key

from matbench_discovery.phonons import check_imaginary_freqs
from matbench_discovery.phonons import thermal_conductivity as ltc

if TYPE_CHECKING:
    from ase.optimize.optimize import Optimizer


def calculate_fc2_batched(
    *,
    ph3: Phono3py,
    model: NequIPTorchSimCalc,
    autobatcher: ts.BinningAutoBatcher,
) -> list:
    """Calculate 2nd order force constants using batched inference.

    Args:
        ph3: phono3py object (with displacements already generated)
        model: torch-sim model instance
        autobatcher: reusable autobatcher instance

    Returns:
        list of FC2 force arrays
    """
    supercells_fc2 = ph3.phonon_supercells_with_displacements
    results = ts.static(
        system=supercells_fc2,
        model=model,
        autobatcher=autobatcher,
        pbar={"desc": "FC2 batched inference"},
    )
    fc2_set = [r["forces"].detach().cpu().numpy() for r in results]
    ph3.phonon_forces = np.array(fc2_set)
    return fc2_set


def calculate_fc3_batched(
    *,
    ph3: Phono3py,
    model: NequIPTorchSimCalc,
    autobatcher: ts.BinningAutoBatcher,
) -> list:
    """Calculate 3rd order force constants using batched inference.

    Args:
        ph3: phono3py object (with displacements already generated)
        model: torch-sim model instance
        autobatcher: reusable autobatcher instance

    Returns:
        list of FC3 force arrays
    """
    supercells_fc3 = ph3.supercells_with_displacements
    results = ts.static(
        system=supercells_fc3,
        model=model,
        autobatcher=autobatcher,
        pbar={"desc": "FC3 batched inference"},
    )
    fc3_set = [r["forces"].detach().cpu().numpy() for r in results]
    ph3.forces = np.array(fc3_set)
    return fc3_set


def calc_force_sets_for_structure(
    *,
    atoms: Atoms,
    displacement_distance: float,
    ase_optimizer: str,
    max_steps: int,
    force_max: float,
    symprec: float,
    enforce_relax_symm: bool,
    relax_log_dir: str,
    task_id: int,
    ase_calculator: Calculator,
    torchsim_model: NequIPTorchSimCalc,
    autobatcher: ts.BinningAutoBatcher,
    ase_filter: str | None = None,
    conductivity_broken_symm: bool = False,
    ignore_imaginary_freqs: bool = False,
    formula_getter: Callable[[Atoms], str] | None = None,
    **_kwargs: Any,
) -> tuple[
    str, Atoms, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Calculate force sets (FC2 and FC3) for a single structure (Stage 1).

    This performs relaxation and batched FC2/FC3 calculations using torch-sim.
    Does NOT calculate thermal conductivity - that's done in Stage 2.

    Args:
        atoms: ASE Atoms object with fc2_supercell, fc3_supercell,
            q_point_mesh keys in its info dict
        displacement_distance: Displacement distance for phono3py (Å)
        ase_optimizer: ASE optimizer name (e.g., 'FIRE', 'BFGS', 'LBFGS')
        max_steps: Maximum relaxation steps
        force_max: Maximum force convergence criterion (eV/Å)
        symprec: Symmetry precision for spglib
        enforce_relax_symm: Whether to enforce symmetry during relaxation
        relax_log_dir: Directory for relaxation log files
        task_id: Task ID for logging
        ase_calculator: ASE calculator for relaxation
        torchsim_model: torch-sim model instance for force calculations
        autobatcher: reusable autobatcher instance (profiled once at start)
        ase_filter: Cell filter for relaxation ('frechet' or 'exp').
            If None, uses FrechetCellFilter. Default None.
        conductivity_broken_symm: Whether to calculate FC3 if symmetry
            breaks. Only used if ignore_imaginary_freqs=False. Default False.
        ignore_imaginary_freqs: Whether to ignore imaginary frequencies
            and calculate FC3 anyway. Default False.
        formula_getter: Custom function to extract formula from atoms.
            If None, uses atoms.get_chemical_formula(). Default None.
        **_kwargs: Additional keywords (unused).

    Returns:
        tuple: (mat_id, relaxed_atoms, fc2_set, fc3_set, relax_dict,
                freqs_dict, err_dict)
    """
    formula = formula_getter(atoms) if formula_getter else atoms.get_chemical_formula()

    print(f"Calculating {Structure.from_ase_atoms(atoms).reduced_formula}")

    # Ensure arrays are writable
    atoms.arrays = {key: val.copy() for key, val in atoms.arrays.items()}

    mat_id = atoms.info[Key.mat_id]
    init_info = deepcopy(atoms.info)
    err_dict: dict[str, Any] = {"errors": [], "error_traceback": []}

    # Select filter class
    if ase_filter in {"frechet", "exp"}:
        filter_cls: type[Filter] = {
            "frechet": FrechetCellFilter,
            "exp": ExpCellFilter,
        }[ase_filter]
    else:
        # Default to FrechetCellFilter if not specified (for MACE compatibility)
        filter_cls = FrechetCellFilter

    # Select optimizer class
    optimizer_dict = {
        "GPMin": ase.optimize.GPMin,
        "GOQN": ase.optimize.GoodOldQuasiNewton,
        "BFGSLineSearch": ase.optimize.BFGSLineSearch,
        "QuasiNewton": ase.optimize.BFGSLineSearch,
        "SciPyFminBFGS": ase.optimize.sciopt.SciPyFminBFGS,
        "BFGS": ase.optimize.BFGS,
        "LBFGSLineSearch": ase.optimize.LBFGSLineSearch,
        "SciPyFminCG": ase.optimize.sciopt.SciPyFminCG,
        "FIRE2": ase.optimize.FIRE2,
        "FIRE": ase.optimize.FIRE,
        "LBFGS": ase.optimize.LBFGS,
    }
    optim_cls: type[Optimizer] = optimizer_dict[ase_optimizer]

    # Initialize variables that might be needed in error handling
    relax_dict: dict[str, Any] = {
        "max_stress": None,
        "reached_max_steps": False,
        "broken_symmetry": False,
    }
    freqs_dict: dict[str, Any] = {}
    fc2_set = np.array([])
    fc3_set = np.array([])
    # initial space group for symmetry breaking detection
    init_spg_num = MoyoDataset(MoyoAdapter.from_atoms(atoms), symprec=symprec).number

    # Relaxation
    try:
        atoms.calc = ase_calculator
        if max_steps > 0:
            if enforce_relax_symm:
                atoms.set_constraint(FixSymmetry(atoms))
                filtered_atoms = filter_cls(atoms, mask=[True] * 3 + [False] * 3)
            else:
                filtered_atoms = filter_cls(atoms)

            os.makedirs(relax_log_dir, exist_ok=True)
            optimizer = optim_cls(
                filtered_atoms, logfile=f"{relax_log_dir}/{task_id}.log"
            )
            optimizer.run(fmax=force_max, steps=max_steps)

            step_count = getattr(optimizer, "nsteps", None)  # Get optimizer step count
            if step_count is None:  # fallback to extract from state_dict if available
                state = getattr(optimizer, "state_dict", dict)()
                step_count = state.get("step", 0)

            reached_max_steps = step_count >= max_steps
            if reached_max_steps:
                print(f"Material {mat_id=} reached {max_steps=} during relaxation")

            # maximum residual stress component in for xx,yy,zz and xy,yz,xz
            # components separately result is a array of 2 elements
            max_stress = atoms.get_stress().reshape((2, 3), order="C").max(axis=1)

            atoms.calc = None
            atoms.constraints = None
            atoms.info = init_info | atoms.info

            # Check if symmetry was broken during relaxation
            moyo_cell = MoyoAdapter.from_atoms(atoms)
            relaxed_spg_num = MoyoDataset(moyo_cell, symprec=symprec).number
            broken_symmetry = init_spg_num != relaxed_spg_num

            relax_dict = {
                "max_stress": max_stress,
                "reached_max_steps": reached_max_steps,
                "broken_symmetry": broken_symmetry,
                "relaxed_space_group_number": relaxed_spg_num,
            }

    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        warnings.warn(f"Failed to relax {formula=}, {mat_id=}: {exc!r}", stacklevel=2)
        traceback.print_exc()
        err_dict["errors"] += [f"RelaxError: {exc!r}"]
        err_dict["error_traceback"] += [traceback.format_exc()]
        return mat_id, atoms, fc2_set, fc3_set, relax_dict, freqs_dict, err_dict

    # Calculation of force sets (using batched torch-sim)
    try:
        ph3 = ltc.init_phono3py(
            atoms,
            fc2_supercell=atoms.info["fc2_supercell"],
            fc3_supercell=atoms.info["fc3_supercell"],
            q_point_mesh=atoms.info["q_point_mesh"],
            displacement_distance=displacement_distance,
            symprec=symprec,
        )

        # Calculate FC2 and get frequencies (batched version of ltc.get_fc2_and_freqs)
        fc2_set = calculate_fc2_batched(
            ph3=ph3, model=torchsim_model, autobatcher=autobatcher
        )

        ph3.produce_fc2(symmetrize_fc2=True)
        ph3.init_phph_interaction(symmetrize_fc3q=False)
        ph3.run_phonon_solver()

        freqs, _eigvecs, _grid = ph3.get_phonon_data()

        has_imaginary_freqs = check_imaginary_freqs(freqs)
        freqs_dict = {Key.has_imag_ph_modes: has_imaginary_freqs, Key.ph_freqs: freqs}

        # Determine if conductivity calculation should proceed
        if ignore_imaginary_freqs:
            # NequIP/Allegro mode: ignore imaginary frequencies
            ltc_condition = True
        else:
            # MACE mode: check both imaginary freqs and broken symmetry
            broken_symmetry = relax_dict.get("broken_symmetry", False)
            ltc_condition = not has_imaginary_freqs and (
                not broken_symmetry or conductivity_broken_symm
            )

        if ltc_condition:
            # Calculate FC3 (batched version of ltc.calculate_fc3_set)
            fc3_set = calculate_fc3_batched(
                ph3=ph3, model=torchsim_model, autobatcher=autobatcher
            )
        else:
            reason = []
            if has_imaginary_freqs:
                reason.append("imaginary frequencies")
            if relax_dict.get("broken_symmetry") and not conductivity_broken_symm:
                reason.append("broken symmetry")
            warnings.warn(
                f"{' and '.join(reason).capitalize()} detected for {mat_id}, "
                f"skipping FC3 calculation!",
                stacklevel=2,
            )
            fc3_set = np.array([])

    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        warnings.warn(f"Failed to calculate force sets {mat_id}: {exc!r}", stacklevel=2)
        traceback.print_exc()
        err_dict["errors"] += [f"ForceConstantError: {exc!r}"]
        err_dict["error_traceback"] += [traceback.format_exc()]

    return mat_id, atoms, fc2_set, fc3_set, relax_dict, freqs_dict, err_dict


def calc_conductivity_from_force_sets(
    *,
    atoms: Atoms,
    fc2_set: np.ndarray,
    fc3_set: np.ndarray,
    displacement_distance: float,
    temperatures: list[float],
    symprec: float,
    has_imaginary_freqs: bool,
    broken_symmetry: bool,
    conductivity_broken_symm: bool = False,
    ignore_imaginary_freqs: bool = False,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Calculate thermal conductivity from pre-computed force sets (Stage 2).

    This reconstructs the phono3py object from saved force sets and calculates
    thermal conductivity. Designed to run on CPU with multiprocessing.

    Args:
        atoms: Relaxed ASE Atoms object with fc2_supercell, fc3_supercell,
            q_point_mesh keys in its info dict
        fc2_set: Pre-computed 2nd order force constants array
        fc3_set: Pre-computed 3rd order force constants array
        displacement_distance: Displacement distance for phono3py (Å)
        temperatures: Temperatures in Kelvin for conductivity calculation
        symprec: Symmetry precision for spglib
        has_imaginary_freqs: Whether structure has imaginary frequencies
        broken_symmetry: Whether symmetry was broken during relaxation
        conductivity_broken_symm: Whether to calculate kappa if symmetry
            breaks. Only used if ignore_imaginary_freqs=False. Default False.
        ignore_imaginary_freqs: Whether to ignore imaginary frequencies
            and calculate kappa anyway. Default False.

    Returns:
        tuple: (mat_id, kappa_dict, err_dict)
    """
    mat_id = atoms.info[Key.mat_id]
    err_dict: dict[str, Any] = {"errors": [], "error_traceback": []}
    kappa_dict: dict[str, Any] = {}

    # determine if conductivity calculation should proceed
    if ignore_imaginary_freqs:
        ltc_condition = True
    else:
        ltc_condition = not has_imaginary_freqs and (
            not broken_symmetry or conductivity_broken_symm
        )

    if not ltc_condition:
        reason = []
        if has_imaginary_freqs:
            reason.append("imaginary frequencies")
        if broken_symmetry and not conductivity_broken_symm:
            reason.append("broken symmetry")
        warnings.warn(
            f"{' and '.join(reason).capitalize()} detected for {mat_id}, "
            f"skipping thermal conductivity calculation!",
            stacklevel=2,
        )
        return mat_id, kappa_dict, err_dict

    # reconstruct phono3py and calculate conductivity
    try:
        # initialize phono3py with relaxed structure
        ph3 = ltc.init_phono3py(
            atoms,
            fc2_supercell=atoms.info["fc2_supercell"],
            fc3_supercell=atoms.info["fc3_supercell"],
            q_point_mesh=atoms.info["q_point_mesh"],
            displacement_distance=displacement_distance,
            symprec=symprec,
        )

        # load pre-computed force sets
        ph3 = ltc.load_force_sets(ph3, fc2_set, fc3_set)

        # calculate thermal conductivity
        ph3, kappa_dict, _cond = ltc.calculate_conductivity(
            ph3, temperatures=temperatures
        )

    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        warnings.warn(
            f"Failed to calculate conductivity {mat_id}: {exc!r}", stacklevel=2
        )
        traceback.print_exc()
        err_dict["errors"] += [f"ConductivityError: {exc!r}"]
        err_dict["error_traceback"] += [traceback.format_exc()]

    return mat_id, kappa_dict, err_dict
