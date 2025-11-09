import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats


def plot_matrix_heatmap(matrix_data, title="Matrix Heatmap"):
    fig = px.imshow(matrix_data, title=title, color_continuous_scale="Viridis")

    fig.update_layout(
        title_x=0.5,
        width=800,
        height=800,
        xaxis_title="Column Index",
        yaxis_title="Row Index",
    )

    return fig


import numpy as np
import plotly.graph_objects as go


def plot_distribution_individual_bins_histogram(data, title="Distribution Plot"):
    mean_errors = np.mean(data)
    median_errors = np.median(data)
    std_errors = np.std(data)

    
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data,
            xbins=dict(
                size=1
            ),
            name="Distribution",
            histnorm="percent",
            marker_color="skyblue",
            opacity=0.6,
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=20),
        },
        xaxis_title="Number of Node Errors",
        yaxis_title="Percentage (%)",
        showlegend=True,
        template="plotly_white",
        width=1000,
        height=600,
        bargap=0.1,
    )

    stats_text = (
        f"Statistics:<br>"
        f"Mean: {mean_errors:.2f}<br>"
        f"Median: {median_errors:.2f}<br>"
        f"Std: {std_errors:.2f}"
    )

    fig.add_annotation(
        x=0.95,
        y=0.95,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )

    return fig


def plot_distribution(data, title="Distribution Plot"):
    mean_errors = np.mean(data)
    median_errors = np.median(data)
    std_errors = np.std(data)

    hist_values, bin_edges = np.histogram(data, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    percentages = (hist_values / len(data)) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=30,
            name="Distribution",
            histnorm="percent",
            marker_color="skyblue",
            opacity=0.6,
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=20),
        },
        xaxis_title="Number of Node Errors",
        yaxis_title="Percentage (%)",
        showlegend=True,
        template="plotly_white",
        width=1000,
        height=600,
    )

    stats_text = (
        f"Statistics:<br>"
        f"Mean: {mean_errors:.2f}<br>"
        f"Median: {median_errors:.2f}<br>"
        f"Std: {std_errors:.2f}"
    )

    fig.add_annotation(
        x=0.95,
        y=0.95,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )

    return fig
