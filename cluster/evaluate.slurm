#!/bin/bash

##############################################################
# SCRIPT MUST BE RUN FROM THE PROJECT DIRECTORY, NOT 'Cluster/'
##############################################################
#SBATCH -A IscrC_RDM                      # account, shared by all the project members
#SBATCH --time 24:00:00                     # max time limit after which the job is killed, format: HH:MM:SS
#SBATCH -N 1                                # ??? 1 node, should match the number of nodes in Pl.Trainer
#SBATCH --ntasks-per-node=1                 # ??? 4 tasks out of 32, should match the number of devices in Pl.Trainer
#SBATCH --gres=gpu:1                        # 1 gpus per node out of 4
#SBATCH --mem=100g                        # memory per node out of 494000MB (481GB)
#SBATCH --job-name=evaluate
#SBATCH --error=logs/%j.err                    # standard error file
#SBATCH --output=logs/%j.out                   # standard output file
#SBATCH --partition=boost_usr_prod

echo $PWD

source cluster/cluster.env

srun uv run python src/mass/scripts/finetune.py -m dataset='SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD,Flowers102,PCAM,FER2013,OxfordIIITPet,STL10,CIFAR100,CIFAR10,Food101,FashionMNIST,EMNIST,KMNIST,RenderedSST2' 'nn/encoder=l14'