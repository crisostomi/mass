import torch
from tqdm import tqdm


@torch.no_grad()
def isotropic_sum(cumulative_dict, datasets, device="cuda"):
    aggregated_model_dict = {}
    for key in cumulative_dict.keys():
        cumulative_dict[key] = cumulative_dict[key].to(device)
        if len(cumulative_dict[key].shape) == 2 and "text_projection" not in key:
            u, s, v = torch.linalg.svd(cumulative_dict[key], full_matrices=False)
            iso_factor = torch.ones_like(s) * s.mean()

            aggregated_model_dict[key] = torch.linalg.multi_dot(
                (
                    u,
                    torch.diag(iso_factor),
                    v,
                )
            )
        else:
            aggregated_model_dict[key] = cumulative_dict[key] / len(datasets)
        aggregated_model_dict[key] = aggregated_model_dict[key].to(device)

    return aggregated_model_dict
