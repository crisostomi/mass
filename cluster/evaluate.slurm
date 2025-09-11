#!/bin/bash
#SBATCH --job-name=evaluate_wemoe_b32_20
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=250G

export http_proxy='http://login07:3140'
export https_proxy='http://login07:3140'

source ./.venv/bin/activate

. /leonardo/prod/spack/5.2/spack_sources/0.21.0_preprod_base/share/spack/setup-env.sh

nvidia-smi

HYDRA_FULL_ERROR=1 uv run wemoe -m 'benchmark=N20' 'nn/module=wemoe' 'nn/encoder=b32' 'data_batch_size=64' 'nn.module.init_lambda=0.1' 'nn.module.batch_size=8'