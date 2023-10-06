# SemAlign-BMI: Class-Semantics Aligned Learning for BMIs
Most Brain-Machine Interfaces (BMIs) today rely on a limited set of commands, which poses scalability issues and potential system failures with slight alterations in the command set. Our semantics-aligned learning framework for EEG representation learning presents a novel method to improve scalability and adaptability of BMIs, particularly focusing on modeling the semantic relationships among the classes.
Specifically, our framework:

- Models the semantic relationships among the classes using a tree or graph structure.
- Disentangles the class-relevant neural latents from class-irrelevant neural noise based on mutual information.
- Leverages the semantics information to design loss functions and batch sampling strategy.

We test our semantics-aware framework on both closed-set and open-set recognition settings, demonstrating its superior performance over traditional hierarchy-agnostic methods, both in terms of accuracy and hierarchical distance of error, i.e., severity of the error in terms of underlying semantics.

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
$ pip install -r requirements.txt
```

## Usage

To run the experiments, navigate to the root directory and execute:

```bash
$ python main.py --gpus <devices_to_use> --kfold <num_folds>
```
