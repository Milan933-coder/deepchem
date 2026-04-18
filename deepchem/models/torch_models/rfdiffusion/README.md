# RFdiffusion Starter for DeepChem

This directory contains a small PR-style starting point for bringing RFdiffusion concepts into DeepChem with a native PyTorch `TorchModel` interface.

## Overview

The goal of this starter is to show the beginning of a real DeepChem integration, not just a placeholder model. The implementation is organized around the same high-level ideas used in upstream RFdiffusion:

- input embeddings for residue, pair, and state features
- track-style iterative updates for MSA, pair, and structure features
- an SE(3)-aware structural update path that can use DeepChem equivariant layers
- a DeepChem `NumpyDataset` and `TorchModel` workflow for training and sampling

## Included in This Starter PR

- `rfdiffusion.py`
  Defines `RFDiffusion` and `RFDiffusionModel`.
- `embeddings.py`
  Adds a compact embedding stack inspired by upstream `Embeddings.py`.
- `track_module.py`
  Adds lightweight DeepChem-side versions of `MSAPairStr2MSA`, `MSA2Pair`, `PairStr2Pair`, `Str2Str`, and `IterativeSimulator`.
- `se3.py`
  Adds a structural update wrapper that uses DeepChem SE(3) layers when `dgl` is available, with a safe fallback path otherwise.
- `data_utils.py`
  Adds helpers for building RFdiffusion-style `NumpyDataset` inputs.
- `tests/test_rfdiffusion.py`
  Adds focused tests for embedding, track, structural update, training, and sampling behavior.

## Current Scope

This is intentionally a starter implementation. It is not yet:

- a full RFdiffusion or RFdiffusion2 port
- a full RoseTTAFold trunk inside DeepChem
- a production inference stack with motif conditioning, contigs, or symmetry controls

## Why This Is Useful

This PR establishes the module layout, DeepChem integration points, and an initial architecture that future work can extend toward closer upstream parity. It gives a clean place to incrementally add:

- richer template and pair conditioning
- closer `Track_module.py` parity
- stronger SE(3) structural updates
- diffusion schedules from `diffusion.py` and `igso3.py`
- inference utilities modeled after upstream RFdiffusion runners

## Next Steps

- extend `embeddings.py` with template and recycling-style features
- deepen the track blocks to better match upstream RFdiffusion
- improve the SE(3) path to cover more of the structural trunk
- add benchmark scripts and evaluation tasks for protein generation
