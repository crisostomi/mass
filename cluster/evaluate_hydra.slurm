#!/bin/bash
#SBATCH --job-name=evaluate_batched
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=2G 


export http_proxy='http://login01:3140'
export https_proxy='http://login01:3140'

source ./.venv/bin/activate

. /leonardo/prod/spack/5.2/spack_sources/0.21.0_preprod_base/share/spack/setup-env.sh


HYDRA_FULL_ERROR=1 uv run evaluate -m 'hydra/launcher=submitit_slurm' 'benchmark=n86' 'nn/module=mass' 'nn/module/router=nn' 'nn/encoder=b32'
