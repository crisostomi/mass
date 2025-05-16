# Task Vectors Playground

<p align="center">
    <a href="https://github.com/crisostomi/task-vectors-playground/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/crisostomi/task-vectors-playground/Test%20Suite/main?label=main%20checks></a>
    <a href="https://crisostomi.github.io/task-vectors-playground"><img alt="Docs" src=https://img.shields.io/github/deployments/crisostomi/task-vectors-playground/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Playing with those task vectors

## Development installation

Setup the development environment:

```bash
git clone git@github.com:crisostomi/task-vectors-playground.git
cd task-vectors-playground
conda env create -f env.yaml
conda activate tvp
```

## Fine-tuning
First, choose a vision encoder in `{"ViT-B-16", "ViT-B-32", "ViT-L-14"}`. Check that the corresponding WandB artifact is present online (at the time of writing, only `ViT-B-16` is available).

Now choose one or more datasets for which you want to compute the task vectors, and set it in `conf/nn/data/default.yaml` under the `dataset` voice in the defaults. Currently available datasets are `{'svhn', 'mnist', 'cifar100', 'resisc45'}`, but any dataset in `tvp/data/datasets/*` can be used. It is enough to create the corresponding configuration file in `conf/nn/data/dataset/`.

Before fine-tuning the model on the dataset, check if the fine-tuned version is already among the artifacts in WandB. It should be named `<model_name>_<dataset_name>_<seed_index>`. If it is not, use `src/scripts/finetune.py` to fine-tune the pretrained model over the chosen dataset. The corresponding configuration is `conf/finetune.yaml`.

## Applying task-vectors
Now the script is `src/scripts/use_task_vectors.py` and the configuration is `conf/task_vectors.yaml`. Task vectors to apply can be chosen in the `task_vectors.to_apply` voice in the config, which expects a list of dataset names.


## Notation

`task_vector`s should represent flattened vectors, while `task_dict`s should be the deltas structured as `state_dict`s.


## No internet installation 
Launch a python shell and do the following to load FER2013 dataset:
```
    from datasets import load_dataset
    fer2013 = load_dataset("Jeneral/fer-2013", split="train")
```