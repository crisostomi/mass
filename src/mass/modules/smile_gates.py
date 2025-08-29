import logging
import math
from typing import List, Optional, Tuple  # noqa: F401
from typing import Optional, TYPE_CHECKING
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from mass.utils.smile_utils import svd
from mass.utils.utils import is_all_zeros
from torch.nn.functional import softmax, dropout, pad

from torch.overrides import (
    handle_torch_function,
    has_torch_function
)


if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

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
        name="",
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
        self.last_selected_experts = None
        self.name = name

        w_diff_list = [m.weight - pretrained_model.weight for m in finetuned_models]
        if is_all_zeros(w_diff_list):
            # All fine-tuned models are identical to the pretrained model
            raise ExpertNotTrainedError()

        if routing_use_diff or k > 0:
            svd_cache_list = [
                svd(w, full_matrices=full_matrices, accelerator=upscaling_accelerator)
                for w in w_diff_list
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

        # handles different inputs shapes. E.g. out_proj gets bsz*patches, embed_dim inputs
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

# Needed to reimplement Attention separately
def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in [
            "cpu",
            "cuda",
            torch.utils.backend_registration._privateuse1_backend_name,
        ]
    return True


def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.requires_grad
    return False


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = (
            torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        )
        return any(
            type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
            for x in torch_dispatch_mode_stack
        )
    else:
        return False



class SmileMultiheadAttention(nn.Module):

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]
    

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
        name="",
    
    ) -> None:
        self.num_experts = len(finetuned_models)
        self.top_k = top_k
        self.k = k
        self.gate_k = gate_k
        self.name = name
        self.embed_dim = pretrained_model.embed_dim
        self.num_heads = pretrained_model.num_heads
        self.dropout = pretrained_model.dropout
        self.add_bias_kv = pretrained_model.bias_k is not None
        self.add_zero_attn = pretrained_model.add_zero_attn
        
        self._qkv_same_embed_dim = pretrained_model._qkv_same_embed_dim
        if not self._qkv_same_embed_dim:
            raise NotImplementedError("Not implemented for different qkv embed dim")
        
        self.kdim = pretrained_model.kdim
        self.vdim = pretrained_model.vdim
        self.batch_first = pretrained_model.batch_first
        
        factory_kwargs = {"device": pretrained_model.in_proj_weight.device, "dtype": pretrained_model.in_proj_weight.dtype}
        super().__init__()
        
        
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        q_weight, k_weight, v_weight = pretrained_model.in_proj_weight.chunk(3)

        if pretrained_model.in_proj_bias is not None:
            q_bias, k_bias, v_bias = pretrained_model.in_proj_bias.chunk(3)
        else:
            q_bias, k_bias, v_bias = None, None, None

        # annoying but allows us to use the existing SMILE MoE layer
        pretrained_q = nn.Linear(self.embed_dim, self.embed_dim, bias=q_bias is not None, **factory_kwargs)
        pretrained_k = nn.Linear(self.embed_dim, self.embed_dim, bias=k_bias is not None, **factory_kwargs)
        pretrained_v = nn.Linear(self.embed_dim, self.embed_dim, bias=v_bias is not None, **factory_kwargs)

        pretrained_q.weight.data, pretrained_k.weight.data, pretrained_v.weight.data = q_weight, k_weight, v_weight

        if q_bias is not None:
            pretrained_q.bias.data, pretrained_k.bias.data, pretrained_v.bias.data = q_bias, k_bias, v_bias
        else:
            pretrained_q.bias.data, pretrained_k.bias.data, pretrained_v.bias.data = None, None, None

        # Extract Q, K, V weights and biases from finetuned models
        finetuned_q_linears = []
        finetuned_k_linears = []
        finetuned_v_linears = []
        
        for ft_model in finetuned_models:
            ft_in_proj_weight = ft_model.in_proj_weight
            ft_in_proj_bias = ft_model.in_proj_bias

            ft_q_weight, ft_k_weight,ft_v_weight = ft_in_proj_weight.chunk(3)
            
            if ft_in_proj_bias is not None:
                ft_q_bias, ft_k_bias, ft_v_bias = ft_in_proj_bias.chunk(3)
            else:
                ft_q_bias, ft_k_bias, ft_v_bias = None, None, None

            ft_q = nn.Linear(self.embed_dim, self.embed_dim, bias=ft_q_bias is not None, **factory_kwargs)
            ft_k = nn.Linear(self.embed_dim, self.embed_dim, bias=ft_k_bias is not None, **factory_kwargs)
            ft_v = nn.Linear(self.embed_dim, self.embed_dim, bias=ft_v_bias is not None, **factory_kwargs)

            ft_q.weight.data, ft_k.weight.data, ft_v.weight.data = ft_q_weight, ft_k_weight, ft_v_weight

            if ft_q_bias is not None:
                ft_q.bias.data, ft_k.bias.data, ft_v.bias.data = ft_q_bias, ft_k_bias, ft_v_bias
            else:
                ft_q.bias.data, ft_k.bias.data, ft_v.bias.data = None, None, None

            finetuned_q_linears.append(ft_q)
            finetuned_k_linears.append(ft_k)
            finetuned_v_linears.append(ft_v)
        
        # Create SmileMoELinear layers for Q, K, V projections
        self.in_proj_q = SmileMoELinear(
            pretrained_model=pretrained_q,
            finetuned_models=finetuned_q_linears,
            gate_k=gate_k,
            k=k,
            top_k=top_k,
            full_matrices=full_matrices,
            upscaling_accelerator=upscaling_accelerator,
            routing_use_diff=routing_use_diff,
            name=f"{name}_q_proj"
        )
        
        self.in_proj_k = SmileMoELinear(
            pretrained_model=pretrained_k,
            finetuned_models=finetuned_k_linears,
            gate_k=gate_k,
            k=k,
            top_k=top_k,
            full_matrices=full_matrices,
            upscaling_accelerator=upscaling_accelerator,
            routing_use_diff=routing_use_diff,
            name=f"{name}_k_proj"
        )
        
        self.in_proj_v = SmileMoELinear(
            pretrained_model=pretrained_v,
            finetuned_models=finetuned_v_linears,
            gate_k=gate_k,
            k=k,
            top_k=top_k,
            full_matrices=full_matrices,
            upscaling_accelerator=upscaling_accelerator,
            routing_use_diff=routing_use_diff,
            name=f"{name}_v_proj"
        )

        self.out_proj = SmileMoELinear(
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

        if self.add_bias_kv:
            self.bias_k, self.bias_v = pretrained_model.bias_k, pretrained_model.bias_v
        else:
            self.bias_k, self.bias_v = None, None

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        why_not_fast_path = ""
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))



        attn_output, attn_output_weights = multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_q,
            self.in_proj_k,
            self.in_proj_v,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> tuple[Optional[Tensor], Optional[int]]:
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len
                ).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type
    
    
def _mha_shape_check(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    num_heads: int,
):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, (
            "For batched (3-D) `query`, expected `key` and `value` to be 3-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, (
            "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, (
                "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, (
                    f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}"
                )
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor"
        )

    return is_batched


def _canonical_mask(
    mask: Optional[Tensor],
    mask_name: str,
    other_type: Optional[DType],
    other_name: str,
    target_type: DType,
    check_other: bool = True,
) -> Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(
                mask, float("-inf")
            )
    return mask


def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


def _check_key_padding_mask(
    key_padding_mask: torch.Tensor, src_len: int, bsz: int
) -> None:
    torch._check_with(
        AssertionError,
        key_padding_mask.shape[0] == bsz,
        lambda: f"Expected key_padded_mask.shape[0] to be {bsz}, but got {key_padding_mask.shape[0]}",
    )
    torch._check_with(
        AssertionError,
        key_padding_mask.shape[1] == src_len,
        lambda: f"Expected key_padded_mask.shape[1] to be {src_len}, but got {key_padding_mask.shape[1]}",
    )
    
def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_q: SmileMoELinear,
    in_proj_k: SmileMoELinear,
    in_proj_v: SmileMoELinear,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj: SmileMoELinear,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:

    tens_ops = (
        query,
        key,
        value,
    )
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_q,
            in_proj_k,
            in_proj_v,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, (
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    )
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, (
        f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    )
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], (
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        )
    else:
        assert key.shape == value.shape, (
            f"key shape {key.shape} does not match value shape {value.shape}"
        )

    assert in_proj_q.weight is not None, (
        "use_separate_proj_weight is True but q_proj_weight is None"
    )
    assert in_proj_k.weight is not None, (
        "use_separate_proj_weight is True but k_proj_weight is None"
    )
    assert in_proj_v.weight is not None, (
        "use_separate_proj_weight is True but v_proj_weight is None"
    )

    q, k, v = in_proj_q(query), in_proj_k(key), in_proj_v(value)

    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, (
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        )
        assert static_k.size(2) == head_dim, (
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        )
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, (
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        )
        assert static_v.size(2) == head_dim, (
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        )
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _check_key_padding_mask(key_padding_mask, src_len, bsz)

        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    if need_weights:
        _B, _Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))

        assert not (is_causal and attn_mask is None), (
            "FIXME: is_causal not implemented for need_weights"
        )

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k.transpose(-2, -1)
            )
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        
        attn_output = out_proj(attn_output, bsz)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = out_proj(attn_output, bsz)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
