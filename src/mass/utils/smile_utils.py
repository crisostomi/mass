from copy import deepcopy
import math
from typing import Dict, List, Mapping, Optional, OrderedDict, Tuple, TypeAlias, Union, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

scaled_dot_product_attention = torch._C._nn.scaled_dot_product_attention

StateDictType: TypeAlias = Dict[str, Tensor]


def state_dict_avg(state_dicts: List[StateDictType]):
    """
    Returns the average of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to average.

    Returns:
        Dict: The average of the state dicts.
    """
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all(
        [len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]
    ), "All state_dicts must have the same number of keys"

    num_state_dicts = len(state_dicts)
    avg_state_dict = OrderedDict()
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict in state_dicts:
            avg_state_dict[key] += state_dict[key]
        avg_state_dict[key] /= num_state_dicts
    return avg_state_dict


def simple_average(
    modules: List[Union[nn.Module, StateDictType]],
    base_module: Optional[nn.Module] = None,
):
    R"""
    Averages the parameters of a list of PyTorch modules or state dictionaries.

    This function takes a list of PyTorch modules or state dictionaries and returns a new module with the averaged parameters, or a new state dictionary with the averaged parameters.

    Args:
        modules (List[Union[nn.Module, StateDictType]]): A list of PyTorch modules or state dictionaries.
        base_module (Optional[nn.Module]): A base module to use for the new module. If provided, the averaged parameters will be loaded into this module. If not provided, a new module will be created by copying the first module in the list.

    Returns:
        module_or_state_dict (Union[nn.Module, StateDictType]): A new PyTorch module with the averaged parameters, or a new state dictionary with the averaged parameters.

    Examples:
        >>> import torch.nn as nn
        >>> model1 = nn.Linear(10, 10)
        >>> model2 = nn.Linear(10, 10)
        >>> averaged_model = simple_averageing([model1, model2])

        >>> state_dict1 = model1.state_dict()
        >>> state_dict2 = model2.state_dict()
        >>> averaged_state_dict = simple_averageing([state_dict1, state_dict2])
    """
    if isinstance(modules[0], nn.Module):
        if base_module is None:
            new_module = deepcopy(modules[0])
        else:
            new_module = base_module
        state_dict = state_dict_avg([module.state_dict() for module in modules])
        new_module.load_state_dict(state_dict)
        return new_module
    elif isinstance(modules[0], Mapping):
        return state_dict_avg(modules)


def get_device(obj) -> torch.device:
    """
    Get the device of a given object.

    Args:
        obj: The object whose device is to be determined.

    Returns:
        torch.device: The device of the given object.

    Raises:
        ValueError: If the object type is not supported.
    """
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, torch.nn.Module):
        if hasattr(obj, "device"):
            return obj.device
        else:
            return next(iter(obj.parameters())).device
    elif isinstance(obj, torch.device):
        return obj
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def get_attr(obj, names: List[str]):
    """
    Gets an attribute of an object recursively.

    Args:
        obj (object): Object to get attribute of.
        names (list): List of attribute names to get recursively.

    Returns:
        object: The attribute of the object.
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


        q, k, v, attn_mask, dropout_p, is_causal





def mha_shape_check(
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



def _svd(w: Tensor, full_matrices: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Singular Value Decomposition (SVD) on a tensor.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def svd(
    w: Tensor,
    full_matrices: bool = True,
    accelerator: Optional[Union[torch.device, str]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform SVD on a tensor, optionally using a specified accelerator.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.
        accelerator (Optional[Union[torch.device, str]]): The device to perform the computation on.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)





    