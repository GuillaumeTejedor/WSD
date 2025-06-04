from sklearn.metrics import silhouette_samples, silhouette_score
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot_all_sequences(values_per_seq, timestamps_per_seq, id_patient_per_seq=None, suptitle="", title="", x_title="", y_title=""):
    """
    Plot all sequences without considering cluster labels.
    :param values_per_seq:      2D array with a list of values per sequence
    :param timestamps_per_seq:  2D array with a list of deltas per sequence
    :param id_patient_per_seq:  1D array with a list of patient IDs per sequence
    :param suptitle:            Figure suptitle
    :param title:               Plot title
    :param x_title:             X-axis label
    :param y_title:             Y-axis label
    :return:                    The figure object
    """

    all_timestamps = np.concatenate(timestamps_per_seq)
    min_delta, max_delta = np.min(all_timestamps), np.max(all_timestamps)

    all_values = np.concatenate(values_per_seq)
    min_value = min(all_values)

    # Plotting parameters
    figsize = (20, 7)
    s_plot = 5
    s_label = 20
    s_suptitle = 40
    s_title = 25
    s_ticks = 15
    offset_axis = 5
    s_id_patient_text = 13

    fig, ax = plt.subplots(figsize=figsize)
    plt.suptitle(suptitle, fontsize=s_suptitle)

    # Plot all sequences
    for i, (values, deltas) in enumerate(zip(values_per_seq, timestamps_per_seq)):
        ax.plot(deltas, values, marker='o', linestyle='--', markersize=s_plot)
        if id_patient_per_seq is not None:
            id_patient = id_patient_per_seq[i]
            ax.text(deltas[-1], values[-1], id_patient, fontsize=s_id_patient_text,
                    verticalalignment='top', horizontalalignment='left')

    ax.set_title(title, size=s_title)
    ax.set_xlabel(x_title, size=s_label)
    ax.set_ylabel(y_title, size=s_label)
    ax.set_xlim(min_delta - offset_axis, max_delta + offset_axis)
    ax.set_ylim(min_value - offset_axis, 48)
    ax.tick_params(axis='both', labelsize=s_ticks)
    ax.grid(True)

    return fig

def plot_clustered_sequences(values_per_seq, timestamps_per_seq, label_per_seq=None, id_patient_per_seq=None, suptitle="", title="", x_title="", y_title="", label_title="", show_legend=True, axis_fontsize_label=20, axis_fontsize_tick=20, markersize=5):
    """
    Plot sequences with their cluster_label.
    :param values_per_seq:  2D array with a list of values per sequence
    :param timestamps_per_seq:  2D array with a list of deltas per sequence
    :param label_per_seq:   1D array with a list of labels per sequence
    :param title:           Figure title
    :param x_title:         X-axis title
    :param y_title:         Y-axis title
    :param label_title:     Common title to give for each cluster_label from legend
    :param show_legend:     Is figure show legend
    :param axis_fontsize_label: 
    :param axis_fontsize_tick:
    :param markersize:
    :return: final figure, axis and legend content
    """

    all_timestamps = np.concatenate(timestamps_per_seq)
    min_delta, max_delta = np.min(all_timestamps), np.max(all_timestamps)

    all_values = np.concatenate(values_per_seq)
    min_value, max_value = min(all_values), max(all_values)
    unique_cluster_labels = np.unique(label_per_seq)
    nb_unique_cluster_labels = len(unique_cluster_labels)
    figsize, s_suptitle, s_title, s_legend, offset_axis, s_id_patient_text = (35, (nb_unique_cluster_labels+1)*15), 40, 25, 35, 5, 13
    fig, ax = plt.subplots(nrows=nb_unique_cluster_labels+1, ncols=1, figsize=figsize, squeeze=False)
    plt.suptitle(suptitle, fontsize=s_suptitle)

    unique_cluster_colors = get_label_colors(unique_cluster_labels)

    ## ---------------- PLOT ALL SEQUENCES ---------------- ##
    legend_content = [Line2D([0], [0], marker='o', color='w', label=label_title + " " + str(cluster_label) + " (n=" + str(list(label_per_seq).count(cluster_label)) + ")", markerfacecolor=unique_cluster_colors[idx], markersize=20) for idx, cluster_label in enumerate(unique_cluster_labels)]

    for i, (values, deltas) in enumerate(zip(values_per_seq, timestamps_per_seq)):
        cluster_label = label_per_seq[i]
        idx = np.where(unique_cluster_labels == cluster_label)[0][0]
        ax[0][0].plot(deltas, values, marker='o', ls='--', c=unique_cluster_colors[idx], markersize=markersize, linewidth=markersize/3)
        if id_patient_per_seq is not None:
            id_patient = id_patient_per_seq[i]
            ax[0][0].text(deltas[-1], values[-1], id_patient, fontsize=s_id_patient_text, verticalalignment='top', horizontalalignment='left')

    ax[0][0].set_title(title, size=s_title)
    ax[0][0].set_xlabel(x_title, size=axis_fontsize_label)
    ax[0][0].set_ylabel(y_title, size=axis_fontsize_label)
    ax[0][0].set_ylabel(y_title, size=axis_fontsize_label)
    ax[0][0].set_xlim(0, max_delta + 100)
    ax[0][0].set_ylim(0, 50)
    ax[0][0].tick_params(axis='both', labelsize=axis_fontsize_tick)
    if show_legend: ax[0][0].legend(handles=legend_content, fontsize=s_legend)
    ax[0][0].grid(True)

    ## ---------------- PLOT SEQUENCES PER LABEL ---------------- ##
    if nb_unique_cluster_labels > 1:
        for cluster_label_index, cluster_label in enumerate(unique_cluster_labels):
            idx = np.where(unique_cluster_labels == cluster_label)[0][0]
            legend_content = [Line2D([0], [0], marker='o', color='w', label=label_title + " " + str(cluster_label) + " (n=" + str(list(label_per_seq).count(cluster_label)) + ")", markerfacecolor=unique_cluster_colors[idx], markersize=20)]
            for i, (values, deltas) in enumerate(zip(values_per_seq, timestamps_per_seq)):
                if cluster_label == label_per_seq[i]:
                    ax[cluster_label_index + 1][0].plot(deltas, values, marker='o', linestyle='--', color=unique_cluster_colors[idx], markersize=markersize)
                    if id_patient_per_seq is not None:
                        id_patient = id_patient_per_seq[i]
                        ax[cluster_label_index + 1][0].text(deltas[-1], values[-1], id_patient, fontsize=s_id_patient_text, verticalalignment='top', horizontalalignment='left')
        
            ax[cluster_label_index + 1][0].set_title(title, size=s_title)
            ax[cluster_label_index + 1][0].set_xlabel(x_title, size=axis_fontsize_label)
            ax[cluster_label_index + 1][0].set_ylabel(y_title, size=axis_fontsize_label)
            ax[cluster_label_index + 1][0].set_xlim(min_delta - offset_axis, max_delta + offset_axis)
            ax[cluster_label_index + 1][0].set_ylim(min_value - offset_axis, max_value + offset_axis)
            ax[cluster_label_index + 1][0].tick_params(axis='both', labelsize=axis_fontsize_tick)
            if show_legend: ax[cluster_label_index + 1][0].legend(handles=legend_content, fontsize=s_legend)
            ax[cluster_label_index + 1][0].grid(True)

    return fig

def plot_clustered_sequences_plotly(values_per_seq, timestamps_per_seq, labels_per_seq=None, title=None, xaxis_title=None, yaxis_title=None):
    
    fig = go.Figure()
    all_timestamps = np.concatenate(timestamps_per_seq)
    min_timestamp, max_timestamp = np.min(all_timestamps), np.max(all_timestamps)
    all_values = np.concatenate(values_per_seq)
    min_value, max_value = min(all_values), max(all_values)

    fig.update_layout(yaxis_range=[min_value, max_value], xaxis_range=[min_timestamp, max_timestamp])
    fig.update_layout(title_text=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, title_x=0.5)
    
    if labels_per_seq is None:
        for values, timestamps in zip(values_per_seq, timestamps_per_seq):
            fig.add_trace(go.Scatter(x=timestamps, y=values, mode="lines+markers", line=dict(color="blue")))
    else:
        unique_cluster_labels = np.sort(np.unique(labels_per_seq))  # Sort cluster labels
        unique_cluster_colors = get_label_colors(unique_cluster_labels)
        clusters_known = set()
        
        for timestamps, values, cluster_label in sorted(zip(timestamps_per_seq, values_per_seq, labels_per_seq), key=lambda x: x[2]):
            color = unique_cluster_colors[np.where(unique_cluster_labels == cluster_label)[0][0]]
            show_legend = cluster_label not in clusters_known
            
            if show_legend:
                clusters_known.add(cluster_label)

            fig.add_trace(go.Scatter(
                x=timestamps, y=values, mode="lines+markers",
                line=dict(color=color),
                name=f"Cluster {cluster_label}" if show_legend else None,
                showlegend=show_legend,
                legendgroup=f"Cluster {cluster_label}"  # Group all traces by cluster
            ))

    return fig

def plot_boxplot(X):

    """
    Plot boxplot of each feature from X.
    :param X: Dataframe where each row represent a patient and each column a feature.
    :return: Figure of boxplots.
    """

    fig = make_subplots(rows=1, cols=X.shape[1])

    ft_names = X.columns.values
    idx_fig = 1

    for ft_name in ft_names:
        fig.add_trace(go.Box(y=X[ft_name], name=ft_name, boxpoints="all"), row=1, col=idx_fig)
        idx_fig = idx_fig + 1

    return fig


def plot_boxplot_clusters_per_feature(X, ft_name, label_column):
    """
    Plot boxplot clusters for each feature.
    :param X: Dataframe where each row represents a patient and each column a feature.
    :param ft_name: Name of the feature to plot.
    :param label_column: Column of dataframe that contains cluster labels.
    :return: Figure of boxplots. 
    """
    labels = X[label_column]
    unique_cluster_labels = np.unique(labels)
    unique_cluster_colors = get_label_colors(unique_cluster_labels)
    
    fig = make_subplots(rows=1, cols=1, y_title=ft_name, x_title="Cluster labels")

    for i, c_label in enumerate(unique_cluster_labels):
        fig.add_trace(go.Box(
            y=X[X[label_column] == c_label][ft_name], 
            name="Cluster " + str(c_label), 
            boxpoints="all",
            marker=dict(color=unique_cluster_colors[i])  # Assign color
        ), row=1, col=1)

    return fig

def silhouette_visualizer(X, cluster_labels, n_clusters, X_is_distance_matrix=True):
    # Calculate silhouette scores
    if X_is_distance_matrix: metric = "precomputed"
    else: metric = "euclidean"
    silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)
    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)

    # Get colors for each label
    unique_cluster_labels = np.unique(cluster_labels)
    print(unique_cluster_labels)
    unique_cluster_colors = get_label_colors(unique_cluster_labels)

    # Plot silhouette analysis
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 10))
    y_lower = 10
    for i, cluster_label in enumerate(unique_cluster_labels):  # Number of clusters
        # Aggregate the silhouette scores for samples in cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == cluster_label]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Fill the silhouette plot
        color = unique_cluster_colors[i]
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plot with cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, cluster_label)
        y_lower = y_upper + 10 # 10 for spacing between plots

    ax1.set_title("Silhouette of samples for each cluster")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # Plot the average silhouette score as a red dashed line
    avg_line = ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette Score")

    # Add legend for the red line
    ax1.legend(handles=[avg_line], loc="lower right")

    ax1.set_yticks([])  # Clear the y-axis labels / ticks
    return fig

def _get_embedding(umap_object):
    if hasattr(umap_object, "embedding_"):
        return umap_object.embedding_
    elif hasattr(umap_object, "embedding"):
        return umap_object.embedding
    else:
        raise ValueError("Could not find embedding attribute of umap_object")

def interactive(
    umap_object,
    labels=None,
    values=None,
    hover_data=None,
    theme=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    point_size=None,
    subset_points=None,):

    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    points = _get_embedding(umap_object)
    if subset_points is not None:
        if len(subset_points) != points.shape[0]:
            raise ValueError(
                "Size of subset points ({}) does not match number of input points ({})".format(
                    len(subset_points), points.shape[0]
                )
            )
        points = points[subset_points]

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    data = pd.DataFrame(_get_embedding(umap_object), columns=("x", "y"))

    if labels is not None:
        data["label"] = labels

        if color_key is None:
            unique_cluster_labels = np.unique(labels)
            num_labels = unique_cluster_labels.shape[0]
            color_key = rgb_to_hex(
                plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            )

        if isinstance(color_key, dict):
            data["color"] = pd.Series(labels).map(color_key)
        else:
            unique_cluster_labels = np.unique(labels)
            if len(color_key) < unique_cluster_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_cluster_labels)}
            data["color"] = pd.Series(labels).map(new_color_key)

        colors = "color"

    elif values is not None:
        data["value"] = values
        palette = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
        colors = btr.linear_cmap(
            "value", palette, low=np.min(values), high=np.max(values)
        )

    else:
        colors = matplotlib.colors.rgb2hex(plt.get_cmap(cmap)(0.5))

    if subset_points is not None:
        data = data[subset_points]
        if hover_data is not None:
            hover_data = hover_data[subset_points]

    if points.shape[0] <= width * height // 10:

        if hover_data is not None:
            tooltip_dict = {}
            for col_name in hover_data:
                data[col_name] = hover_data[col_name]
                tooltip_dict[col_name] = "@" + col_name
            tooltips = list(tooltip_dict.items())
        else:
            tooltips = None

        data["alpha"] = 1

        bpl.output_notebook(hide_banner=True)
        data_source = bpl.ColumnDataSource(data)

        plot = bpl.figure(
            width=width,
            height=height,
            tooltips=tooltips,
            background_fill_color=background,
        )
        plot.circle(
            x="x",
            y="y",
            source=data_source,
            color=colors,
            size=point_size,
            alpha="alpha",
        )

        plot.grid.visible = False
        plot.axis.visible = False

        
    else:
        if hover_data is not None:
            warn(
                "Too many points for hover data -- tooltips will not"
                "be displayed. Sorry; try subssampling your data."
            )
        if interactive_text_search:
            warn(
                "Too many points for text search." "Sorry; try subssampling your data."
            )
        hv.extension("bokeh")
        hv.output(size=300)
        hv.opts('RGB [bgcolor="{}", xaxis=None, yaxis=None]'.format(background))
        if labels is not None:
            point_plot = hv.Points(data, kdims=["x", "y"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("color"),
                color_key=color_key,
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        elif values is not None:
            min_val = data.values.min()
            val_range = data.values.max() - min_val
            data["val_cat"] = pd.Categorical(
                (data.values - min_val) // (val_range // 256)
            )
            point_plot = hv.Points(data, kdims=["x", "y"], vdims=["val_cat"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("val_cat"),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        else:
            point_plot = hv.Points(data, kdims=["x", "y"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count(),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )

    return plot