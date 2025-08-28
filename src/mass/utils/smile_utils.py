from copy import deepcopy
from typing import Dict, List, Mapping, Optional, OrderedDict, TypeAlias, Union

import torch
from torch import Tensor, nn

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