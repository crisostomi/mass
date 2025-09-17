import torch.nn as nn

class LanguageInference(nn.Module):
    def __init__(self, moe_model):
        super().__init__()
        self.moe_model = moe_model

    def forward(self, x):
        return self.moe_model(x)