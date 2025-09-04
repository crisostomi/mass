HYDRA_FULL_ERROR=1 HF_HUB_OFFLINE=1 uv run evaluate -m 'nn/benchmark=twenty' 'nn/module/encoder=b32' 'nn.module.router.threshold=0.05,0.1,0.15,0.2,0.3' 'nn.module.router.max_num_tasks_to_select=1,2,3,4,5'


HYDRA_FULL_ERROR=1 HF_HUB_OFFLINE=1 uv run evaluate -m 'nn/benchmark=twenty' 'nn/module/encoder=b16' 'similarity_threshold=-0.4,-0.2,0.0,0.1,0.2,0.4,0.6,0.8' 


# to finetune all the datasets
# first 8 
uv run src/mass/scripts/finetune.py -m dataset='SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD' 

# second 6
uv run src/mass/scripts/finetune.py -m dataset='Flowers102,PCAM,FER2013,OxfordIIITPet,STL10,CIFAR100'

# last 6
uv run src/mass/scripts/finetune.py -m dataset='CIFAR10,Food101,FashionMNIST,EMNIST,KMNIST,RenderedSST2'