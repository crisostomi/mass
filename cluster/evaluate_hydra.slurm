#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=IscrC_MGNTC
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=2G

cd ..

./shell_scripts/flush.sh

source ./.venv/bin/activate

. /leonardo/prod/spack/5.2/spack_sources/0.21.0_preprod_base/share/spack/setup-env.sh

nvidia-smi

wandb offline

HYDRA_FULL_ERROR=1 HF_HUB_OFFLINE=1 uv run evaluate -m 'nn/benchmark=twenty' 'nn/module/encoder=b16,b32,l14' 'nn/module/aggregator=tsvm' 'nn.module.router.threshold=0.2'

# 'eval_on_train=true' 'number_of_train_batches=10'