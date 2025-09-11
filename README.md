# MASS: MoErging through Adaptive Subspace Selection

<p align="center">
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

> [!WARNING] 
> Codebase undergoing a refactoring. Use with caution.

## 👨🏻‍💻 Development installation

Setup the development environment:

```bash
    git clone git@github.com:crisostomi/mass.git
    cd mass
    uv sync
```

and you are ready to go! Fast, right?

<!-- Create a .env file in the root folder with your chosen paths. This is currently due to legacy reasons and these will be removed in the future.
```sh
    DATA_PATH=/path/to/store/datasets
    MODELS_PATH=/path/to/store/models
    HYDRA_FULL_ERROR=1
``` -->

## 🚀 Getting Started
We provide script to replicate all the experiments in the paper and the baselines.

> [!WARNING]  
> We ran our experiments using an **NVIDIA A100 64 GB**. This hardware **is not necessary** to replicate all the experiments, but is highly recommended especially for the ViT-L-14 encoder based experiments. Exception made for WeMoE, to the best of our knowledge, an NVIDIA GeForce RTX 2080 Ti with 11 GB of RAM is sufficient for ViT-B-32 based experiments for all supported methods.


### 🥤 Scripts

Below is the folder structure for the scripts provided in this repository, along with a brief description of their purpose:

```plaintext
src/mass/scripts/
├── evaluate_pipeline.py       # MASS evaluation script
├── train_router.py            # Train an MLP router for task classification
├── evaluate_smile.py          # SMILE evaluation script
├── evaluate_we_moe.py         # WeMoE evaluation script
├── finetune.py                # Fine-tuning checkpoints script
├── interpret.py               # Interpretability analysis of task vectors
├── evaluate_static_merging.py # Evaluate traditional merging methods
└── evaluate_task_classification.py # Task classification evaluation
```

To run the scripts, use `uv run src/scripts/<script_name>.py` from the root folder. However, you might want to check the current configuration first in the [Configuration Structure](#configuration-strcture) section and adapt it to your needs.

**💡 Examples of commands for experiments in the paper:**

To run and test out method, MASS, on the 8 tasks benchmark with a ViT-B-32:
```bash
    uv run evaluate benchmark=n8 nn/module=mass nn/encoder=b32
```

To try out a static merging method, e.g. TSV-M, using the ViT-B-16 encoder on the 20 tasks benchmark:
```bash
    uv run static_merge merger=tsv nn/encoder=b16 benchmark=n20
```

### 🤗 Models and data

You don't have to worry about anything! Models will be downloaded automatically from [Donato Crisostomi's](https://huggingface.co/crisostomi) Hugging Face page, while datasets will be downloaded from [fusion bench's page](https://huggingface.co/tanganke). For more details, please refer to [their paper](#📚-references) at the end of this document.

We note that we produced new checkpoints for all the models used in the paper (CLIP architectures). We did so to solve an annoying bug that was coming with the DTD checkpoint that has been probably trained on an unknown split. For coherence with previous literature, we kept the specif parameters per dataset suggested in the original paper. 

## ⭐️ Contribute

We welcome contributions to this project! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

### Configuration Structure
```plaintext
conf/
├── benchmark/                # 8-14-20 Tasks Benchmark
│   ├── n8.yaml               
│   ├── n14.yaml              
│   └── n20.yaml              
├── hydra/                    # Hydra configurations
├── merger/                   # Static merging options
├── dataset/                  
├── nn/                       
│   ├── encoder/              # b32, b16, l14 Encoders 
│   │   ├── b32.yaml          
│   │   ├── b16.yaml          
│   │   └── l14.yaml          
│   └── module/               # Available methods
│       ├── aggregator/       # Deprecated to be removed
│       ├── router/           # Routers options
│       ├── smile.yaml
│       ├── mass.yaml         # MASS model config
│       ├── task.yaml 
│       └── we_moe.yaml       
├── train/                    # Training configurations
├── static_merging.yaml       # Static merging config
├── task_vectors.yaml         # Main Config
└── train_router.yaml         # Train MLP router config
```

## 📚 References

```bibtex
@misc{tang2024fusionbenchcomprehensivebenchmarkdeep,
      title={FusionBench: A Comprehensive Benchmark of Deep Model Fusion}, 
      author={Anke Tang and Li Shen and Yong Luo and Han Hu and Bo Du and Dacheng Tao},
      year={2024},
      eprint={2406.03280},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.03280}
}
```

### Additional References
- **Models**: Pre-trained and fine-tuned models are available on [Donato Crisostomi's HuggingFace page](https://huggingface.co/crisostomi)
- **Datasets**: Evaluation datasets are sourced from [FusionBench's HuggingFace page](https://huggingface.co/tanganke)