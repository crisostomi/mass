#!/bin/bash
#SBATCH --job-name=evaluate_smile_l14_14_20
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=2G


pwd

export http_proxy='http://login05:3140'
export https_proxy='http://login05:3140'

source ./.venv/bin/activate

. /leonardo/prod/spack/5.2/spack_sources/0.21.0_preprod_base/share/spack/setup-env.sh

nvidia-smi

HYDRA_FULL_ERROR=1 uv run smile -m 'hydra/launcher=submitit_slurm' 'benchmark=N14,N20' 'nn/module=smile' 'nn.module.k=32,128' 'nn/encoder=l14' 'data_batch_size=4'

