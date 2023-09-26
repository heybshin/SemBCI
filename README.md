# SemAlign-BMI: Class-Semantics Aligned Learning for BMIs
Brain-Machine Interfaces (BMIs) are paving the way for revolutionized human-machine interactions. Our semantics-aligned learning framework for EEG representation learning presents a novel method to improve scalability and adaptability of BMIs, particularly focusing on modeling the semantic relationships among the classes.

## Introduction
Most BMIs today rely on a limited set of commands, which poses scalability issues and potential system failures with slight alterations in the command set. Our class-semantics aligned learning framework:

- Models the semantic relationships among the classes.
- Disentangles the class-relevant neural latents from class-irrelevant neural noise.
- Implements semantics-aligned deep representation learning or probabilistic modeling.
Through rigorous tests, our hierarchy-based methods have shown superior performance against traditional hierarchy-agnostic methods, both in terms of accuracy and semantic error hierarchy.

## Requirements
Python >= 3.x
CUDA >= 10.x (If using GPU)
Other dependencies
- `cuda`
- `pytorch_lightning`
- `mne`
- `networkx`
- `nltk`

```bash
pip install -r requirements.txt
```

## Usage

To run the experiments, navigate to the root directory and execute:

```bash
python main.py --gpus 0 --kfold 5 -ds Neuroskin -dt MNE HIER
```

## Main Components
Configurations (configs): Houses the YAML configuration files for different experiments and settings.

Datasets (datasets): Contains utilities for preprocessing and augmenting the NeuroSkin dataset, alongside necessary transformations.

Hierarchy (hierarchy): Manages hierarchical relations and trees for the touch gestures.

Models (models): Contains the neural network architectures and modules.

Loss Functions (losses): Consists of custom loss functions used for training, such as ARPL, Contrastive, Disentangle, and Hierarchical.

Hyptorch (hyptorch): Modules related to hyperbolic neural networks.
