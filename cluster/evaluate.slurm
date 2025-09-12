#!/bin/bash
#SBATCH --job-name=evaluate_b32_20
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=150G

export http_proxy='http://login01:3140'
export https_proxy='http://login01:3140'

source ./.venv/bin/activate

. /leonardo/prod/spack/5.2/spack_sources/0.21.0_preprod_base/share/spack/setup-env.sh

nvidia-smi

HYDRA_FULL_ERROR=1 uv run evaluate -m 'hydra/launcher=basic' 'benchmark=n8' 'nn/module=mass' 'nn/module/router=batch-proj' 'nn/encoder=b32'