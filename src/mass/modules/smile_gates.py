import logging
from typing import Dict, List, Optional, Tuple, Union  # noqa: F401

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mass.utils.smile_utils import mha_shape_check, svd
from mass.utils.utils import is_all_zeros

scaled_dot_product_attention = torch._C._nn.scaled_dot_product_attention

pylogger = logging.getLogger(__name__)


class ExpertNotTrainedError(Exception):
    pass


class SmileGate(nn.Module):
    __constants__ = ["in_features", "num_experts", "k"]
    in_features: int
    num_experts: int
    k: int
    weight: nn.Parameter

    def __init__(
        self,
        input_features: int,
        w_diff_list: List[Tensor],
        k: int,
        svd_cache: List[
            Tuple[Tensor, Tensor, Tensor]
        ] = None,  # cached `svd_cache`, pass it to avoid recomputing
        upscaling_accelerator=None,
    ):
        R"""
        This constructs weights through SVD decomposition.

        Args:
            input_features: The dimension of input features.
            w_diff_list: The list of weight matrices to be decomposed.
            k: The number of singular values to keep.
            svd_cache: The cached SVD decomposition results. If not provided, the SVD decomposition will be computed on the fly.
            upscaling_accelerator: The accelerator to use for SVD decomposition.
        """
        super().__init__()
        self.input_features = input_features
        self.num_experts = len(w_diff_list)
        weights = []
        for i, w_diff in enumerate(w_diff_list):
            if svd_cache is None:
                u, s, v = svd(w_diff, accelerator=upscaling_accelerator)
            else:
                u, s, v = svd_cache[i]
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

            # weights.append((s * v).T)
            weights.append(v.T)
        self.k = s.size(0)  # k is the actual k after truncation

        weights = (
            torch.stack(weights, dim=0)
            .reshape(self.num_experts * self.k, -1)
            .contiguous()
        )
        self.weights = nn.Parameter(
            weights
        )  # weights should be a tensor of shape (num_experts * k, n)

    def forward(self, x: Tensor):
        batch_size = x.size(0)
        if self.num_experts == 1:
            return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        routing_weights = F.linear(x, self.weights).view(
            batch_size, self.num_experts, self.k
        )
        routing_weights = routing_weights.norm(p=2, dim=2)
        return routing_weights


class SmileCompressedLinear(nn.Module):
    """
    This module is used to compress a linear layer using SVD decomposition.
    """

    __constants__ = ["in_features", "out_features", "k"]
    in_features: int
    out_features: int
    k: int

    u: nn.Parameter
    svh: nn.Parameter
    bias: Optional[nn.Parameter]

    def __init__(
        self,
        model: nn.Linear,
        k: int,
        svd_cache: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ):
        """
        Initialize the SmileCompressedLinear module.

        Args:
            model (nn.Linear): The linear model to compress.
            k (int): The number of singular values to keep.
            svd_cache (Tuple[Tensor, Tensor, Tensor]): Cached SVD results.
        """
        super().__init__()
        self.in_features = model.in_features
        self.out_features = model.out_features
        self.k = k

        if svd_cache is None:
            u, s, v = svd(model.weight)
        else:
            u, s, v = svd_cache
        if k > 0:
            u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

        self.u = nn.Parameter(u)
        self.svh = nn.Parameter((s * v).T)

        if model.bias is not None:
            self.bias = nn.Parameter(model.bias.data, requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Forward pass of the SmileCompressedLinear module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = F.linear(x, self.svh)
        x = F.linear(x, self.u, self.bias)
        return x
    
class SmileMultiHeadAttention(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.MultiheadAttention,
        finetuned_models: List[nn.MultiheadAttention],
        gate_k: int,
        k: int,
        top_k: int = 1,
        full_matrices=True,
        upscaling_accelerator=None,
        routing_use_diff=True,
        name: str = "",
    ) -> None:
        super().__init__()
        self.num_experts = len(finetuned_models)
        self.top_k = top_k
        self.k = k
        self.gate_k = gate_k
        self.name = name
        self.embed_dim = pretrained_model.embed_dim
        
        self._qkv_same_embed_dim = pretrained_model._qkv_same_embed_dim
        
        if not self._qkv_same_embed_dim:
            raise NotImplementedError("Not implemented for different qkv embed dim")
        
        in_proj_weight = pretrained_model.in_proj_weight  
        in_proj_bias = pretrained_model.in_proj_bias      
        
        q_weight = in_proj_weight[:self.embed_dim, :]
        k_weight = in_proj_weight[self.embed_dim:2*self.embed_dim, :]
        v_weight = in_proj_weight[2*self.embed_dim:, :]
        
        if in_proj_bias is not None:
            q_bias = in_proj_bias[:self.embed_dim]
            k_bias = in_proj_bias[self.embed_dim:2*self.embed_dim]
            v_bias = in_proj_bias[2*self.embed_dim:]
        else:
            q_bias = k_bias = v_bias = None
        
        pretrained_q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        pretrained_k_linear = nn.Linear(self.embed_dim, self.embed_dim)
        pretrained_v_linear = nn.Linear(self.embed_dim, self.embed_dim)
        
        pretrained_q_linear.weight = nn.Parameter(q_weight)
        pretrained_k_linear.weight = nn.Parameter(k_weight)
        pretrained_v_linear.weight = nn.Parameter(v_weight)
        
        if q_bias is not None:
            pretrained_q_linear.bias = nn.Parameter(q_bias)
            pretrained_k_linear.bias = nn.Parameter(k_bias)
            pretrained_v_linear.bias = nn.Parameter(v_bias)
        else:
            pretrained_q_linear.bias = None
            pretrained_k_linear.bias = None
            pretrained_v_linear.bias = None
        
        finetuned_q_linears = []
        finetuned_k_linears = []
        finetuned_v_linears = []
        
        for m in finetuned_models:
            ft_in_proj_weight = m.in_proj_weight
            ft_in_proj_bias = m.in_proj_bias
            
            ft_q_weight = ft_in_proj_weight[:self.embed_dim, :]
            ft_k_weight = ft_in_proj_weight[self.embed_dim:2*self.embed_dim, :]
            ft_v_weight = ft_in_proj_weight[2*self.embed_dim:, :]
            
            if ft_in_proj_bias is not None:
                ft_q_bias = ft_in_proj_bias[:self.embed_dim]
                ft_k_bias = ft_in_proj_bias[self.embed_dim:2*self.embed_dim]
                ft_v_bias = ft_in_proj_bias[2*self.embed_dim:]
            else:
                ft_q_bias = ft_k_bias = ft_v_bias = None
            
            q_linear = nn.Linear(self.embed_dim, self.embed_dim)
            k_linear = nn.Linear(self.embed_dim, self.embed_dim)
            v_linear = nn.Linear(self.embed_dim, self.embed_dim)
            
            q_linear.weight = nn.Parameter(ft_q_weight)
            k_linear.weight = nn.Parameter(ft_k_weight)
            v_linear.weight = nn.Parameter(ft_v_weight)
            
            if ft_q_bias is not None:
                q_linear.bias = nn.Parameter(ft_q_bias)
                k_linear.bias = nn.Parameter(ft_k_bias)
                v_linear.bias = nn.Parameter(ft_v_bias)
            else:
                q_linear.bias = None
                k_linear.bias = None
                v_linear.bias = None
            
            finetuned_q_linears.append(q_linear)
            finetuned_k_linears.append(k_linear)
            finetuned_v_linears.append(v_linear)
        
        self.q_proj_smile = SmileMoELinear(
            pretrained_model=pretrained_q_linear,
            finetuned_models=finetuned_q_linears,
            gate_k=gate_k,
            k=k,
            top_k=top_k,
            full_matrices=full_matrices,
            upscaling_accelerator=upscaling_accelerator,
            routing_use_diff=routing_use_diff,
            name=f"{name}_q_proj"
        )
        
        self.k_proj_smile = SmileMoELinear(
            pretrained_model=pretrained_k_linear,
            finetuned_models=finetuned_k_linears,
            gate_k=gate_k,
            k=k,
            top_k=top_k,
            full_matrices=full_matrices,
            upscaling_accelerator=upscaling_accelerator,
            routing_use_diff=routing_use_diff,
            name=f"{name}_k_proj"
        )
        
        self.v_proj_smile = SmileMoELinear(
            pretrained_model=pretrained_v_linear,
            finetuned_models=finetuned_v_linears,
            gate_k=gate_k,
            k=k,
            top_k=top_k,
            full_matrices=full_matrices,
            upscaling_accelerator=upscaling_accelerator,
            routing_use_diff=routing_use_diff,
            name=f"{name}_v_proj"
        )

        self.out_proj_smile = SmileMoELinear(
            pretrained_model=pretrained_model.out_proj,
            finetuned_models=[m.out_proj for m in finetuned_models],
            gate_k=gate_k,
            k=k,
            top_k=top_k,
            full_matrices=full_matrices,
            upscaling_accelerator=upscaling_accelerator,
            routing_use_diff=routing_use_diff,
            name=f"{name}_out_proj"
        )

        self.pretrained_model = pretrained_model

    def forward(self, 
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,):
    
        
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        num_heads = self.pretrained_model.num_heads
        
        is_batched = mha_shape_check(
            query, key, value, key_padding_mask, attn_mask, num_heads
        )
        
        head_dim = embed_dim // num_heads

        q_output, k_output, v_output = self.q_proj_smile(query), self.k_proj_smile(key), self.v_proj_smile(value)
        
        q = q_output.view(bsz, num_heads, tgt_len, head_dim)
        k = k_output.view(bsz, num_heads, src_len, head_dim)
        v = v_output.view(bsz, num_heads, src_len, head_dim)

        out_proj_input = scaled_dot_product_attention(
            q, k, v, attn_mask, 0.0, is_causal
        )
        
        out_proj_input = (
            out_proj_input.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = self.out_proj_smile(out_proj_input, batch_size=bsz)

        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            attn_output = attn_output.squeeze(1)
        
        if self.pretrained_model.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        else:
            return attn_output, None

    @property
    def weight(self):
        """
        Mimic linear layer. Bacause in some cases, user might indicate the device (or dtype of parameters) of the linear layer using `linear_layer.weight.device`
        """
        return self.pretrained_model.weight

    @property
    def bias(self):
        return self.pretrained_model.bias

    def __repr__(self):
        return (
            f"SingularMoELinear("
            f"in_features={self.pretrained_model.in_features}, "
            f"out_features={self.pretrained_model.out_features}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"gate_k={self.gate_k}, "
            f"k={self.k}"
            f")"
        )
        
        
    
    
class SmileMoELinear(nn.Module):
    __constants__ = [
        "in_features",
        "out_features",
        "num_experts",
        "top_k",
        "gate_k",
        "k",
    ]
    in_features: int
    out_features: int
    num_experts: int
    top_k: int
    gate_k: int
    k: int

    @torch.no_grad()
    def __init__(
        self,
        pretrained_model: nn.Linear,
        finetuned_models: List[nn.Linear],
        gate_k: int,
        k: int,
        top_k: int = 1,
        full_matrices=True,
        upscaling_accelerator=None,
        routing_use_diff=True,
        name: str = "",
    ):
        """
        Initialize the SmileMoELinear module.

        Args:
            pretrained_model (nn.Linear): The pretrained linear model.
            finetuned_models (List[nn.Linear]): A list of fine-tuned linear models.
            gate_k (int): The number of singular values to keep for the gate.
            k (int): The number of singular values to keep for the experts.
            top_k (int): The number of top experts to select.
            full_matrices (bool): Whether to compute the full-sized U and V matrices.
            upscaling_accelerator (str): The device to perform the computation on.
            routing_use_diff (bool): Whether to use weight differences for routing.
        """
        super().__init__()
        self.num_experts = len(finetuned_models)
        self.top_k = top_k
        self.k = k
        self.gate_k = gate_k
        self.in_features = pretrained_model.in_features
        self.out_features = pretrained_model.out_features
        self.name = name

        w_diff_list = [m.weight - pretrained_model.weight for m in finetuned_models]
        if is_all_zeros(w_diff_list):
            # All fine-tuned models are identical to the pretrained model
            raise ExpertNotTrainedError()

        if routing_use_diff or k > 0:
            svd_cache_list = [
                torch.linalg.svd(w, full_matrices=full_matrices) for w in w_diff_list
            ]  # the svd cache list to avoid recomputing

        # construct the gate network
        if routing_use_diff:
            self.gate = SmileGate(
                input_features=self.in_features,
                w_diff_list=w_diff_list,
                k=gate_k,
                svd_cache=svd_cache_list,
                upscaling_accelerator=upscaling_accelerator,
            )
        else:
            self.gate = SmileGate(
                input_features=self.in_features,
                w_diff_list=[m.weight for m in finetuned_models],
                k=gate_k,
                svd_cache=None,
                upscaling_accelerator=upscaling_accelerator,
            )

        # construct experts
        for m, w_diff in zip(finetuned_models, w_diff_list):
            m.weight.data = w_diff
        if k > 0:
            experts = [
                SmileCompressedLinear(m, k, svd_cache=svd_cache)
                for m, svd_cache in zip(finetuned_models, svd_cache_list)
            ]
        else:
            # if k is not set (<0), we use the full fine-tuned model
            experts = finetuned_models
        self.experts = nn.ModuleList(experts)

        if pretrained_model.bias is not None:
            for m in experts:
                m.bias.data = m.bias.data - pretrained_model.bias
        # assign the pretrained model (the shared part)
        self.pretrained_model = pretrained_model

    def forward(self, hidden_states: Tensor, batch_size=None):
        """
        Forward pass of the SmileMoELinear module.

        Args:
            hidden_states (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # pylogger.info(f"Layer {self.name}")
        pretrained_out = self.pretrained_model(hidden_states)

        input_shape = hidden_states.size()
        # pylogger.info(f"input shape: {input_shape}")
        hidden_states = hidden_states.view(-1, self.in_features)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1)
        # sample the expert according to the routing weights
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (hidden_states.size(0), self.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, self.in_features)
            if current_state.numel() == 0:
                continue
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            *input_shape[:-1], self.out_features
        )
        final_hidden_states = pretrained_out + final_hidden_states

        if len(input_shape) == 3:
            self.last_selected_experts = (
                selected_experts.view(input_shape[0], input_shape[1])
                .cpu()
                .mode(dim=0)
                .values
            )
        elif len(input_shape) == 2 and batch_size:
            patches = input_shape[0] // batch_size
            self.last_selected_experts = (
                selected_experts.view(patches, batch_size)
                .cpu()
                .mode(dim=0)
                .values
            )

        return final_hidden_states

    @property
    def weight(self):
        """
        Mimic linear layer. Bacause in some cases, user might indicate the device (or dtype of parameters) of the linear layer using `linear_layer.weight.device`
        """
        return self.pretrained_model.weight

    @property
    def bias(self):
        return self.pretrained_model.bias

    def __repr__(self):
        return (
            f"SingularMoELinear("
            f"in_features={self.pretrained_model.in_features}, "
            f"out_features={self.pretrained_model.out_features}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"gate_k={self.gate_k}, "
            f"k={self.k}"
            f")"
        )

