import plotly.graph_objects as go
import json
import matplotlib.pyplot as plt
import numpy as np
import re


class Palette(dict):
    def __init__(self, palette_path, map_path):
        with open(palette_path, "r") as f:
            color_palette = json.load(f)

        with open(map_path, "r") as f:
            mapping = json.load(f)

        for color, hex_code in color_palette.items():
            self[mapping[color]] = hex_code

    def get_colors(self, n):
        return list(self.values())[:n]


def plot_radar_chart(data_list, model_labels, color_map, title="Radar Chart"):
    """
    Plots a radar chart for multiple dictionaries of data.

    Parameters:
    -----------
    data_list : list of dict
        Each dict contains {category_name: value} pairs.
        All dicts should have the same keys (categories).
    model_labels : list of str
        Labels for each dict in data_list (e.g. "Model A", "Model B", ...).
        Must match the length of data_list.
    color_map : dict
        Mapping from model label (str) to color (e.g. {'Model A': 'red', ...}).
    title : str, optional
        Title of the radar chart.
    """

    # -- 1) Gather categories from the first dictionary --
    #    (Assumes all dictionaries share the same keys in the same order)
    categories = list(data_list[0].keys())
    N = len(categories)

    # -- 2) Compute the angle for each category --
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Repeat the first angle to close the circle
    angles += angles[:1]

    # -- 3) Create polar subplot --
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Move the 0° axis to the top
    ax.set_theta_offset(np.pi / 2)
    # Flip direction so that angles increase clockwise
    ax.set_theta_direction(-1)

    # -- 4) Set the category labels around the circle --
    plt.xticks(angles[:-1], categories)

    # Optionally, set radial limits. For typical 0-1 data, use [0, 1.1]
    ax.set_ylim(0, 1.1)

    # -- 5) Plot each dictionary's data --
    for data_dict, label in zip(data_list, model_labels):
        # Extract values in the same category order
        values = [data_dict[cat] for cat in categories]
        # Close the data loop
        values += values[:1]

        # Fetch color from the color map
        color = color_map.get(label, "black")  # default to black if not found

        # Plot the outline
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        # Fill the area
        ax.fill(angles, values, color=color, alpha=0.1)

    # -- 6) Final touches: legend, title, etc. --
    plt.title(title, y=1.08)

    # Add a consistent legend for all subplots
    plt.legend(
        loc="lower center",
        # mode="expand",
        ncol=len(data_list),
        bbox_to_anchor=[0.5, -0.18],
        markerscale=2.5,
        frameon=False,
        labelspacing=2,
    )

    plt.tight_layout()
    plt.show()


def plot_interactive_radar_chart(data_dict, title="Radar Chart"):
    """
    Plots a single-model radar (polar) chart using Plotly.

    Parameters
    ----------
    data_dict : dict
        A dictionary with {category_name: value} pairs (e.g., {"A": 0.8, "B": 0.9, ...}).
    color : str, optional
        Color for the polygon (line + fill).
    title : str, optional
        Title of the radar chart.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly Figure containing the radar chart.
    """
    # 1) Get categories and their values
    categories = list(data_dict.keys())
    accs = list([val[0][f"acc/test/{key}"] for key, val in data_dict.items()])
    normalized_accs = list(
        [val[0][f"normalized_acc/test/{key}"] for key, val in data_dict.items()]
    )

    # 2) Create the Plotly figure
    fig = go.Figure()

    # 3) Add a single scatterpolar trace
    fig.add_trace(
        go.Scatterpolar(
            r=accs,
            theta=categories,
            fill="toself",
            name="Accuracy",
            line=dict(color="blue", width=2),
            fillcolor="blue",
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=normalized_accs,
            theta=categories,
            fill="toself",
            name="Normalized accuracy",
            line=dict(color="red", width=2),
            fillcolor="red",
            opacity=0.6,
        )
    )

    # 4) Update layout to style the chart
    #    - radialaxis.range can be adjusted based on data (0 to 1, 0 to 10, etc.)
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(range=[0.4, 1.1], showticklabels=True, ticks=""),
            angularaxis=dict(direction="clockwise", rotation=90),
        ),
        showlegend=False,
        width=800,
        height=800,
    )

    return fig


def plot_interactive_coefficients_barchart(
    coefficients,
    dataset_names,
    x_label="Dataset",
    y_label="Coefficient Value",
    title="Coefficients by Dataset",
):
    """
    Plots an interactive bar chart for coefficients across multiple datasets.

    Parameters
    ----------
    coefficients : dict
        A dictionary mapping dataset_name -> numeric coefficient value.
        Example: {'Dataset A': 0.5, 'Dataset B': -0.3, ...}
    dataset_names : list
        A list of dataset names in the order they should appear on the x-axis.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly Figure containing the bar chart.
    """

    # 2) Create a Plotly Figure and add a single Bar trace
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dataset_names,
            y=coefficients,
            text=[f"{val:.2f}" for val in coefficients],  # optional numeric label
            textposition="auto",
        )
    )

    # 3) Update layout (size, titles, etc.)
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=800,  # increase or decrease for your preference
        height=500,  # increase or decrease for your preference
        template="plotly_white",  # optional styling
        showlegend=False,
    )

    return fig


def plot_interactive_norms_barcharts(
    parameters,
    values,
    labels=["L2 projection norms", "L1 projection norms", "Layer Norms"],
):
    """
    Plots grouped bar charts comparing two sets of norms.

    Parameters:
    - parameters: List of dataset names or categories
    - values1: First set of norm values (e.g., projection norms)
    - values2: Second set of norm values (e.g., layer norms)
    - label1: Label for the first set of bars
    - label2: Label for the second set of bars
    """

    fig = go.Figure()

    for value, label in zip(values, labels):
        # First set of bars
        fig.add_trace(
            go.Bar(
                x=parameters,
                y=value,
                name=label,
                text=[f"{val:.2f}" for val in value],
                textposition="auto",
            )
        )

    # Update layout for grouped bars
    fig.update_layout(
        title="Norms Comparison by Dataset",
        xaxis_title="Norms",
        yaxis_title="Coefficient Value",
        width=800,
        height=500,
        template="plotly_white",
        showlegend=True,
        barmode="group",  # Display bars side-by-side
    )

    return fig


def plot_interactive_curve(
    x_values, y_values, title="Curve Plot", x_label="X", y_label="Y"
):
    """
    Plots an interactive curve given x and y points.

    Parameters:
    ----------
    x_values : list or array-like
        The x-coordinates of the points.
    y_values : list or array-like
        The y-coordinates of the points.
    title : str, optional
        Title of the plot (default: "Curve Plot").
    x_label : str, optional
        Label for the x-axis (default: "X").
    y_label : str, optional
        Label for the y-axis (default: "Y").

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        A Plotly Figure containing the curve.
    """
    fig = go.Figure()

    # Add a trace for the curve
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",  # Line with points
            name="Activated tasks",
            marker=dict(size=6),
            line=dict(width=2),
            text=[f"({x:.2f}, {y:.2f})" for x, y in zip(x_values, y_values)],
            hoverinfo="text",
        )
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=800,
        height=500,
        template="plotly_white",
        showlegend=True,
    )

    return fig


def plot_interactive_coefficients_std(coeff_means, coeff_stds, dataset_names):
    """
    Plots an interactive bar chart with error bars representing standard deviation.

    Parameters
    ----------
    coeff_means : list or np.array
        Mean coefficient values for each dataset.
    coeff_stds : list or np.array
        Standard deviation values for each dataset.
    dataset_names : list
        List of dataset names in the order they should appear on the x-axis.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly Figure containing the bar chart with error bars.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=dataset_names,
            y=coeff_means,
            error_y=dict(
                type="data", array=coeff_stds, visible=True
            ),  # Adds error bars
            text=[
                f"{mean:.2f} ± {std:.2f}" for mean, std in zip(coeff_means, coeff_stds)
            ],
            textposition="auto",
            marker=dict(color="blue"),
        )
    )

    fig.update_layout(
        title="Coefficient Means with Standard Deviation",
        xaxis_title="Dataset",
        yaxis_title="Coefficient Value",
        width=800,
        height=500,
        template="plotly_white",
        showlegend=False,
    )

    return fig


def parse_resblock(layer_name):
    pattern = r"resblocks\.(\d+)\.(attn|mlp)"
    match = re.search(pattern, layer_name)
    if match:
        idx = int(match.group(1))
        block_type = match.group(2)
        return (idx, block_type)
    return (9999, "unknown")


# Sorting key: first by index, then by block type (attn before mlp)
def sort_key(layer_name):
    idx, btype = parse_resblock(layer_name)
    btype_order = 0 if btype == "attn" else 1
    return (idx, btype_order)


def create_interactive_layer_task_residual_plot(
    layer_residuals: dict, datasets, title="Average Residuals per Layer per Task"
):

    layers_sorted = sorted(layer_residuals.keys(), key=sort_key)

    avg_residuals_by_layer = {}
    std_residuals_by_layer = {}
    n_tasks = None

    for layer in layers_sorted:
        # shape: (n_batches, n_tasks)
        vectors = np.array(layer_residuals[layer])
        avg_vector = vectors.mean(axis=0)  # shape: (n_tasks,)
        std_vector = vectors.std(axis=0)  # shape: (n_tasks,)

        avg_residuals_by_layer[layer] = avg_vector
        std_residuals_by_layer[layer] = std_vector

        if n_tasks is None:
            n_tasks = avg_vector.shape[0]
        else:
            assert (
                n_tasks == avg_vector.shape[0]
            ), "Inconsistent number of tasks across layers."

    fig = go.Figure()
    for task_idx in range(n_tasks):

        y_values = [avg_residuals_by_layer[layer][task_idx] for layer in layers_sorted]

        y_std_values = [
            std_residuals_by_layer[layer][task_idx] for layer in layers_sorted
        ]

        fig.add_trace(
            go.Scatter(
                x=layers_sorted,
                y=y_values,
                mode="lines+markers",
                name=f"{datasets[task_idx]}",
                error_y=dict(type="data", array=y_std_values), 
            )
        )

    fig.update_layout(title=title, xaxis_title="Layer", yaxis_title="Average Residual")
    return fig


def create_interactive_layer_task_accuracy_plot(
    layer_accuracy: dict,
    right_task_index: int,
    datasets,
    title="Task Accuracy per Layer",
):
    """
    Create an interactive Plotly bar chart where the x-axis represents layers and each bar shows the average accuracy
    for the correct task (right_task_index) computed from logged predictions stored in layer_accuracy.

    Args:
        layer_accuracy (dict): Dictionary mapping layer names (e.g., "resblocks.11.attn") to a list of tensors.
                               Each tensor (torch.Tensor) should contain predicted task indices for a batch.
        right_task_index (int): The correct task index that predictions should match.
        datasets (list): List of dataset (or task) names.
        title (str): Title of the plot.

    Returns:
        fig: A Plotly figure with one bar trace showing accuracy per layer for the correct task.
    """
    # Sort layers in a consistent order
    layers_sorted = sorted(layer_accuracy.keys(), key=sort_key)

    accuracies = []

    # Compute accuracy per layer by comparing predictions to the correct task index.
    for layer in layers_sorted:
        predictions_all = []
        for tensor in layer_accuracy[layer]:
            # Ensure the tensor is on CPU and convert it to a NumPy array.
            predictions_all.append(tensor.cpu().numpy())

        predictions_all = np.concatenate(predictions_all)  # shape: (total_samples,)

        # Compute binary accuracy for the correct task.
        binary_accuracy = (predictions_all == right_task_index).astype(np.float32)

        avg_acc = binary_accuracy.mean()

        accuracies.append(avg_acc)

    # Create a bar chart for the correct task's accuracy.
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=layers_sorted,
            y=accuracies,
            text=[f"{acc:.2f}" for acc in accuracies],
            textposition="auto",
            name=f"Task Accuracy {datasets[right_task_index]}",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Task Accuracy",
        yaxis=dict(range=[0, 1]),
    )

    return fig


def create_interactive_layer_impact_bar_chart(
    layer_impact: dict, title="Average Layer Impact per Layer"
):
    """
    Create an interactive Plotly bar chart for average layer impact.

    For each layer in the provided layer_impact dictionary, this function concatenates all recorded
    impact values and computes the average impact. It then produces a bar chart with one column per layer.
    """
    layers_sorted = sorted(layer_impact.keys(), key=sort_key)

    avg_impacts = []
    for layer in layers_sorted:
        all_diffs = np.concatenate([d for d in layer_impact[layer]])
        avg = all_diffs.mean() if all_diffs.size > 0 else 0.0
        avg_impacts.append(avg)

    fig = go.Figure(
        data=[
            go.Bar(
                x=layers_sorted,
                y=avg_impacts,
                marker=dict(color="rgba(100, 150, 200, 0.7)"),
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Average L2 Norm Difference",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig
