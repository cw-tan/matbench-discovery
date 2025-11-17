# NequIP torch-sim Evaluations

Batched GPU evaluation scripts for NequIP models using torch-sim.

## Installation

```bash
pip install phono3py
```

## Usage

### Discovery Test (WBM dataset)

```bash
# full dataset
python test_nequip_discovery.py model.nequip.pt2

# quick test (128 structures)
python test_nequip_discovery.py model.nequip.pt2 --test
```

### Thermal Conductivity (PhononDB-PBE 103)

Requires two model files: one compiled for ASE (relaxation) and one compiled for torch-sim (batched force calculations).

```bash
python test_nequip_kappa.py ase_model.nequip.pt2 torchsim_model.nequip.pt2
```

## Options

Run with `--help` to see all available options:
```bash
python test_nequip_discovery.py --help
python test_nequip_kappa.py --help
```
