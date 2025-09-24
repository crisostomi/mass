# MASS: MoErging through Adaptive Subspace Selection

<p align="center">
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## 👨🏻‍💻 Development installation

Setup the development environment:

```bash
    git clone https://github.com/crisostomi/mass.git
    cd mass
    uv sync
```

and you are ready to go!

### Optional
If you want to set specific paths for datasets and models, you can create a `.env` file in the root folder with the following environment variables:

```sh
    HYDRA_FULL_ERROR=1
    HF_HOME="~/.cache/huggingface"
    TOKENIZERS_PARALLELISM=false
```

## 🚀 Getting Started
We provide script to replicate all the experiments in the paper and the baselines.

> [!WARNING]  
> While reproducing the results for the vision tasks is no particularly hardware demanding (excpet for WeMoE 20 tasks), for the NLP tasks a GPU with more than 24GB of RAM is reqruired. Using a smaller subsets fot the Glue benchmark can be a solution to this problem (but provides different results).


### 🥤 Scripts

Below is the folder structure for the scripts provided in this repository, along with a brief description of their purpose:

```plaintext
src/mass/scripts/
├── evaluate_vision.py         # MoE vision evaluation script
├── evaluate_llms.py           # MoE LLMs evaluation script
├── finetune.py                # Fine-tuning checkpoints script
└── evaluate_static_merging.py # Evaluate static merging methods
```

To run the scripts, use `uv run src/scripts/<script_name>.py` from the root folder. However, you might want to check the current configuration first in the [Configuration Structure](#configuration-strcture) section and adapt it to your needs.

**💡 Examples of commands for experiments in the paper:**

To run and test out method, MASS, on the 8 vision tasks benchmark with a ViT-B-32:
```bash
    uv run evaluate_vision benchmark=n8 nn/module=mass nn/encoder=b32
```

To run and test out method, MASS, on the Glue benchmark with a ViT-B-32:
```bash
    uv run evaluate_language benchmark=glue nn/module=mass nn/encoder=t5
```

To try out a static merging method, e.g. TSV-M, using the ViT-B-16 encoder on the 20 tasks benchmark:
```bash
    uv run static_merge merger=tsv nn/encoder=b16 benchmark=n20
```

To finetune your own checkpoints on the 8 tasks benchmark use:
```bash
    uv run finetune -m dataset='SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD' 
```



### 🤗 Models and data

You don't have to worry about anything! Models will be downloaded automatically from [Donato Crisostomi's](https://huggingface.co/crisostomi) Hugging Face page, while datasets will be downloaded from [fusion bench's page](https://huggingface.co/tanganke). 

We note that we produced new checkpoints for all the models used in the paper (CLIP architectures). We did so to solve an annoying bug that was coming with the DTD checkpoint that has been probably trained on an unknown split. For coherence with previous literature, we kept the specif parameters per dataset suggested in the original paper. 


### Configuration Structure
```plaintext
conf/
├── benchmark/                # 8-14-20 Tasks Benchmark
│   ├── glue.yaml
│   ├── n8.yaml               
│   ├── n14.yaml              
│   └── n20.yaml   
│           
├── hydra/                    # Hydra configurations
├── merger/                   # Static merging options
├── dataset/                  # individual datasets configurations
├── nn/                       
│   ├── encoder/              # b32, b16, l14 Encoders 
│   │   ├── b32.yaml          
│   │   ├── b16.yaml          
│   │   └── l14.yaml          
│   ├── module/               # Available methods
│   │   ├── router/           # Routers options
│   │   ├── smile.yaml
│   │   ├── mass.yaml         # MASS model config
│   │   ├── task.yaml 
│   │   └── we_moe.yaml 
│   ├── task/                # Task specific inference wrappers
│   │   ├── image_classification.yaml
│   │   ├── lang_classification.yaml 
│   │   └── lang_regression.yaml
│   └── tokenizer/            # Tokenizers options
│
├── train/                    # Training configurations
├── eval_language.yaml        # LLMs config
├── eval_vision.yaml          # Vision config
└── finetune.yaml             # Fine-tuning config
```

## 📚 Cite


### References
- **ViTs**: Pre-trained and fine-tuned vision models are available on [Donato Crisostomi's HuggingFace page](https://huggingface.co/crisostomi)
- **LLMs**: Pre-trained and fine-tuned language models are available on [FusionBench's HuggingFace page](https://huggingface.co/tanganke)
- **Datasets**: Evaluation datasets are sourced from [FusionBench's HuggingFace page](https://huggingface.co/tanganke)