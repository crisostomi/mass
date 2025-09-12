#!/bin/bash
#SBATCH --job-name=evaluate_batched
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=2G 


pwd

export http_proxy='http://login01:3140'
export https_proxy='http://login01:3140'

source ./.venv/bin/activate

. /leonardo/prod/spack/5.2/spack_sources/0.21.0_preprod_base/share/spack/setup-env.sh

nvidia-smi

HYDRA_FULL_ERROR=1 uv run evaluate -m 'hydra/launcher=submitit_slurm' 'benchmark=n8,n14,n20' 'nn/module=mass' 'nn/module/router=batch-proj' 'nn/encoder=b32,b16,l14'
