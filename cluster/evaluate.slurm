#!/bin/bash
#SBATCH -A IscrC_MGNTC                      # account, shared by all the project members
#SBATCH --time 24:00:00                     # max time limit after which the job is killed, format: HH:MM:SS
#SBATCH -N 1                                # ??? 1 node, should match the number of nodes in Pl.Trainer
#SBATCH --ntasks-per-node=1                 # ??? 4 tasks out of 32, should match the number of devices in Pl.Trainer
#SBATCH --gres=gpu:1                        # 1 gpus per node out of 4
#SBATCH --mem=100000                        # memory per node out of 494000MB (481GB)
#SBATCH --job-name=evaluate
#SBATCH --error=logs/%j.err                    # standard error file
#SBATCH --output=logs/%j.out                   # standard output file
#SBATCH --partition=boost_usr_prod

cd ..

source ./.venv/bin/activate

. /leonardo/prod/spack/5.2/spack_sources/0.21.0_preprod_base/share/spack/setup-env.sh

wandb offline

HYDRA_FULL_ERROR=1 HF_HUB_OFFLINE=1 srun uv run evaluate -m 'nn/benchmark=eight,fourteen,twenty' 'nn/module/encoder=b32,b16,l14'