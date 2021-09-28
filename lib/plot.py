"""
Created on September 5 2021

This module contains most of the plot function of this package.

@author: Ofir Magdaci (@Magdaci)

"""

import PIL
import pandas as pd
import numpy as np
import string
from matplotlib import rcParams, pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
from plotly import graph_objects as go
from plotly.graph_objs import Layout
from plotly.subplots import make_subplots
from sklearn.metrics import auc

from lib.params import COLORS, COLUMNS, CONSTANTS
from lib.utils import yard_2_meter
from sklearn import preprocessing

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Lato']


# Basics
def plot_pitch(field_dimensions=CONSTANTS.PITCH_DIMENSIONS, **kwargs):
    """ plot_pitch
    Plots a football pitch. All distance units converted to meters.
    :param field_dimensions: (length, width) of field in meters. Default is (106,68)
    :param kwargs: may hold many additional parameters. Please follow meter based dimensions.
        - line_width: line width to pitch [=2]
        - marker_size: market size for scatters on the pitch [=20]
        - pitch_colors_style: colors of the pitch - 'classic', 'lab' (default)
        - pitch_paddings: (horizontal padding, vertical padding) [=3]
        - pitch_line_color
        - six_box_width
        - six_box_length
        - goal_line_width
        - area_width
        - area_length
        - corner_radius
        - centre_circle_radius
        -
    :return: fig, ax - figure and ax objects
    """
    line_width = kwargs.get('line_width', 2)
    marker_size = kwargs.get('marker_size', 20)
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', CONSTANTS.FIGSIZE))
    pitch_colors_style = kwargs.get('pitch_colors_style', 'classic')

    if pitch_colors_style == 'classic':
        ax.set_facecolor(COLORS.FOREST_GREEN)
        pitch_line_color = 'whitesmoke'
        point_color = kwargs.get('point_color', COLORS.W)
        grass_line_width = kwargs.get('grass_line_width', field_dimensions[0] / kwargs.get('num_grass_lines', 14))
        grass_grid = np.arange(-field_dimensions[0] / 2, field_dimensions[0] / 2, grass_line_width)
        for i in range(len(grass_grid)):
            if i % 2 == 1:
                continue
            rect = patches.Rectangle((grass_grid[i], -field_dimensions[1] / 2),
                                     width=grass_line_width, height=field_dimensions[1],
                                     fill=True, color=COLORS.DARK_GREEN, edgecolor=COLORS.DARK_GREEN, zorder=0)

            # Add the patch to the Axes
            ax.add_patch(rect)

    elif pitch_colors_style == 'lab':
        ax.set_facecolor(COLORS.WHITE)
        pitch_line_color = kwargs.get('pitch_line_color', COLORS.BLACK)
        point_color = kwargs.get('point_color', COLORS.K)

    elif pitch_colors_style.lower() in list(COLORS.__dict__.values()):
        ax.set_facecolor(pitch_colors_style.lower())
        pitch_line_color = kwargs.get('pitch_line_color', COLORS.BLACK)
        point_color = kwargs.get('point_color', COLORS.K)

    else:
        raise ImportError

    pitch_paddings = kwargs.get('pitch_paddings', (5, 5))
    half_pitch_length, half_pitch_width = field_dimensions[0] / 2, field_dimensions[1] / 2
    sides = [-1, 1]

    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    six_box_width = kwargs.get('six_box_width', yard_2_meter(20))
    six_box_length = kwargs.get('six_box_length', yard_2_meter(6))
    goal_line_width = kwargs.get('goal_line_width', yard_2_meter(8))
    area_width = kwargs.get('area_width', yard_2_meter(44))
    area_length = kwargs.get('area_length', yard_2_meter(18))
    corner_radius = kwargs.get('corner_radius', yard_2_meter(1))
    centre_circle_radius = kwargs.get('D_pos', yard_2_meter(10))

    # Plot the half pitch
    ax.plot([0, 0], [-half_pitch_width, half_pitch_width], pitch_line_color, linewidth=line_width, zorder=1)
    ax.scatter(0.0, 0.0, marker='o', facecolor=pitch_line_color, linewidth=0, s=marker_size, zorder=1)
    y = np.linspace(-1, 1, 50) * centre_circle_radius
    x = np.sqrt(centre_circle_radius ** 2 - y ** 2)
    ax.plot(x, y, pitch_line_color, linewidth=line_width, zorder=1)
    ax.plot(-x, y, pitch_line_color, linewidth=line_width, zorder=1)

    for side_ in sides:
        # Pitch boundary
        ax.plot([-half_pitch_length, half_pitch_length], [side_ * half_pitch_width, side_ * half_pitch_width],
                pitch_line_color, linewidth=line_width, zorder=1)
        ax.plot([side_ * half_pitch_length, side_ * half_pitch_length], [-half_pitch_width, half_pitch_width],
                pitch_line_color, linewidth=line_width, zorder=1)

        # Corner
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius ** 2 - y ** 2)
        ax.plot(side_ * (half_pitch_length - x), -half_pitch_width + y, pitch_line_color,
                linewidth=line_width, zorder=1)
        ax.plot(side_ * (half_pitch_length - x), half_pitch_width - y, pitch_line_color, linewidth=line_width, zorder=1)

        # Goal posts & line
        ax.plot([side_ * half_pitch_length, side_ * half_pitch_length],
                [-goal_line_width / 2., goal_line_width / 2.],
                point_color + 's', markersize=marker_size / 10, linewidth=line_width, zorder=1)

        # Nets
        def draw_net(sign, half_pitch_length: float, goal_line_width: float, net_width: float, net_thickness=0.5):
            # Length - x axis, width - y axis
            min_vertical, min_horizontal = sign * half_pitch_length, - goal_line_width / 2
            max_vertical = sign * (half_pitch_length + net_width + net_thickness)
            max_horizontal = goal_line_width / 2

            if sign == -1:
                temp = min_vertical
                min_vertical = max_vertical
                max_vertical = temp

            for vertical_line in np.arange(min_vertical, max_vertical + net_thickness, net_thickness):
                for horizontal_line in np.arange(min_horizontal, max_horizontal + net_thickness, net_thickness):
                    ax.plot([vertical_line, vertical_line], [min_horizontal, max_horizontal],
                            pitch_line_color, linewidth=line_width / 2, zorder=1)
                    ax.plot([min_vertical, max_vertical], [horizontal_line, horizontal_line],
                            pitch_line_color, linewidth=line_width / 2, zorder=1)

        draw_net(side_, half_pitch_length, goal_line_width, net_width=kwargs.get('net_width', 3),
                 net_thickness=kwargs.get('net_thickness', 0.5))

        for side__ in sides:
            # Six yard box - horizontal
            ax.plot([side_ * half_pitch_length, side_ * (half_pitch_length - six_box_length)],
                    [side__ * six_box_width / 2., side__ * six_box_width / 2.],
                    pitch_line_color, linewidth=line_width)

            # Penalty area - horizontal
            ax.plot([side_ * half_pitch_length, side_ * (half_pitch_length - area_length)],
                    [side__ * area_width / 2., side__ * area_width / 2.],
                    pitch_line_color, linewidth=line_width)

        # Six yard box - vertical
        ax.plot([side_ * (half_pitch_length - six_box_length), side_ * (half_pitch_length - six_box_length)],
                [-six_box_width / 2., six_box_width / 2.],
                pitch_line_color, linewidth=line_width, zorder=1)

        # Penalty area - vertical
        ax.plot([side_ * (half_pitch_length - area_length), side_ * (half_pitch_length - area_length)],
                [-area_width / 2., area_width / 2.], pitch_line_color, linewidth=line_width, zorder=1)

        # Penalty spot
        ax.scatter(side_ * (half_pitch_length - kwargs.get('penalty_spot', yard_2_meter(12))),
                   0.0, marker='o', facecolor=pitch_line_color, linewidth=0, s=marker_size, zorder=1)

        # D area
        y = np.linspace(-1, 1, 50) * kwargs.get('D_length', yard_2_meter(8))
        x = np.sqrt(kwargs.get('D_radius', yard_2_meter(10)) ** 2 - y ** 2) + kwargs.get('D_pos', yard_2_meter(12))
        ax.plot(side_ * (half_pitch_length - x), y, pitch_line_color, linewidth=line_width, zorder=1)

        # Remove axis labels, ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        x_max = field_dimensions[0] / 2. + pitch_paddings[0]
        ax.set_xlim([-x_max, x_max])
        y_max = field_dimensions[1] / 2. + pitch_paddings[1]
        ax.set_ylim([-y_max, y_max])
        ax.set_axisbelow(True)

    ax.text(-half_pitch_length * 0.2, half_pitch_width * 0.8, "Ofir Magdaci", alpha=0.5, fontsize=20, zorder=2)
    return fig, ax


def radar_chart(radar_data: pd.DataFrame, baselines: dict = None, categories_display_names: list = None,
                normalize_data: bool = False) -> go.Figure:
    '''
    Plots a radar chart for the data, using 0-100 scale
    :param radar_data: DataFrame of metrics for the given player. The player name is in the 'name' column
    :param baselines: dict of baselines data: {baseline_name: baseline radar_data, ...} where baseline radar_data
                        has the same format as radar_data
    :param categories_display_names: list of optional categories names
    :param normalize_data: bool, whether to normalize the data to percentiles (if True) or keep as is (if False)
    :return: Plotly figure of of the radar data
    '''
    radar_data = radar_data.copy()
    radar_data.set_index('name', inplace=True)
    player_name = str(radar_data.index[0])

    categories = categories_display_names if categories_display_names is not None else radar_data.columns

    def _handle_data(_data: pd.DataFrame):
        data = _data.values
        if normalize_data:
            percentile_transformer = preprocessing.QuantileTransformer()
            x_scaled = percentile_transformer.fit_transform(data)
            _normalized = pd.DataFrame(x_scaled, columns=[categories])
            _normalized.set_index(radar_data.index, inplace=True)
            _normalized = _normalized.apply(lambda val: val * 100, axis=1)
        else:
            _normalized = data.copy()

        return _normalized

    if normalize_data:
        normalized = _handle_data(radar_data)
        radar_data.values[0] = normalized.mean(axis=0)

    # Build figure and a radar chart for player
    fig_data = [go.Scatterpolar(r=radar_data.values[0], theta=categories, fill='toself', name=str(player_name),
                                dr=10, r0=0)]

    # Add baselines radar charts
    if baselines is not None:
        for _baseline, baseline_df in baselines.items():
            if normalize_data:
                normalized = _handle_data(baseline_df)
                baseline_df.values = normalized.mean(axis=0)

            for ix, baseline_subcategory in baseline_df.iterrows():
                fig_data.append(go.Scatterpolar(r=baseline_subcategory, theta=categories, fill='toself',
                                                name=f"{ix} average",
                                                dr=10, r0=0,
                                                visible='legendonly'))
    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            title=go.layout.Title(text=f'{string.capwords(player_name)} skills radar chart'),
            polar={'radialaxis': {'visible': True}},
            showlegend=True,
        )
    )

    fig.update_layout(template='plotly_white')
    return fig


def plot_metric_by_dimension(events_df: pd.DataFrame, metric_column: str, prob_column: str, groupby_column: str,
                             lift=True, figax=None, metric_display_name=None, filter_other=False, **kwargs) -> (
        plt.figure, plt.axis):
    '''
    Analyze metric column (e.g., COLUMNS.XG), using prob column (e.g., COLUMNS.GOAL)
    :param events_df: DataFrame of events data according to StatsBomb format
    :param metric_column:  the metric column (str) to be evaluated against the prob_column. For example. COLUMNS.GOAL
    :param prob_column: the probability column (str) to be evaluated against the metric_column. For example. COLUMNS.XG
    :param lift: if True, function plots the lift over
    :param figax: (figure, axis) if required to plot on
    :param groupby_column: 'body_part', 'play_pattern_name', 'shot_type_name'...
    :param metric_display_name: display name for metric_column
    :param filter_other: whether to remove shots with type='Other' or not.
    :return: (figure, axis)
    '''
    df = events_df.copy()
    if figax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', CONSTANTS.LANDSCAPE_FIGSIZE))

    else:
        fig, ax = figax

    df['body_part'] = df['shot_body_part_name'].copy()
    if metric_display_name is None:
        metric_display_name = metric_column

    df = df[events_df[metric_column].notna()]
    df = df[events_df[prob_column].notna()]
    for col in [metric_column, prob_column]:
        df[col] = df[col].astype(float)

    df[groupby_column] = df[groupby_column].apply(lambda val: val.lower().replace(" ", "_"))
    aggr_df = df[[metric_column, prob_column, groupby_column]].groupby(by=groupby_column) \
        .agg({metric_column: [np.mean, np.std], prob_column: [np.mean, np.std]}).reset_index(drop=False)
    aggr_df.columns = [groupby_column, f"mean_{metric_column}", f"std_{metric_column}",
                       f"mean_{prob_column}", f"std_{prob_column}"]
    if filter_other:
        aggr_df = aggr_df[aggr_df[groupby_column] != 'Other']
    ix_2_cats = dict(enumerate([val for val in df[groupby_column].unique() if val is not np.nan]))
    cats_2_ix = {val: key for key, val in ix_2_cats.items()}
    aggr_df[f'{groupby_column}_ix'] = aggr_df[groupby_column].apply(lambda val: cats_2_ix[val])

    # Plot prob by bar
    bar_height = 0.5
    if lift:
        lifts = aggr_df[f"mean_{metric_column}"] / aggr_df[f"mean_{prob_column}"]
        y = lifts
        ax.set_xticks(np.arange(0.4, 2.2, 0.2))
        ax.set_xlim(np.arange(0.4, 2.))
        ax.vlines(1, ymin=aggr_df[f'{groupby_column}_ix'].min(), ymax=aggr_df[f'{groupby_column}_ix'].max(),
                  linestyle='--', color=COLORS.LIGHTGREY)
    else:
        y = aggr_df[f"mean_{prob_column}"]
        ax.set_xticks(np.arange(0., 1.2, 0.2))
        ax.set_xlim(kwargs.get('xlim', (0, 1.)))

    ax.barh(aggr_df[f'{groupby_column}_ix'], y, height=bar_height, align='center', color=COLORS.DIMGREY)
    ax.set_yticks(aggr_df[f'{groupby_column}_ix'])
    ax.set_yticklabels([string.capwords(cat).replace('_', ' ') for cat in aggr_df[groupby_column]])
    ax.set_xlabel(f"{groupby_column.capitalize().replace('_', ' ')} {'LIFT' if lift else ''}")

    for direction in ['right', 'top']:
        ax.spines[direction].set_visible(False)

    if lift:
        ax.set_title(f"{metric_display_name.capitalize()} conversion LIFT by {groupby_column.replace('_', ' ')}")
    else:
        ax.set_title(f"Mean {metric_display_name.capitalize()} by {groupby_column.replace('_', ' ')}")
    return fig, ax


def success_vs_failure_scatter_plot(events_df: pd.DataFrame, metric_column: str, prob_column: str, figax=None,
                                    **kwargs) -> (plt.figure, plt.axis):
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', CONSTANTS.LANDSCAPE_FIGSIZE), dpi=300)

    min_y, max_y, = kwargs.get('y_success', -0.5), kwargs.get('max_y', 0.5)
    y_success, y_fail = kwargs.get('y_success', -0.25), kwargs.get('y_fail', 0.25)
    fail_label, success_label = 'Missed', 'Scored'

    events_df = events_df[events_df[metric_column].notna()]
    if 'color' not in events_df.columns:
        events_df['color'] = events_df[metric_column].apply(lambda res: COLORS.FOREST_GREEN if res else COLORS.RED)

    events_df['prob_column_high_res_bins'] = events_df[prob_column].apply(
        lambda x: np.round(np.round(x * 50, 0) / 100, 2))
    events_df = events_df.sort_values(by=['prob_column_high_res_bins'], ascending=True)
    events_df['y_vals'] = events_df[metric_column].apply(lambda val: y_success if val else y_fail)

    # prepare scatter points sizes
    scatter_sizes = events_df['prob_column_high_res_bins'].value_counts().to_dict()
    scatter_sizes = {str(key): val for key, val in scatter_sizes.items()}
    events_df['scatter_sizes'] = events_df[prob_column].apply(
        lambda x: scatter_sizes[str(np.round(np.round(x * 50, 0) / 100, 2))])

    # Plot
    ax.scatter(events_df[prob_column], events_df['y_vals'], color=events_df['color'], edgecolors=COLORS.WHITE,
               s=events_df['scatter_sizes'])
    ax.set_yticks([y_fail, y_success])
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.set_yticklabels([fail_label, success_label])
    ax.set_xlabel("xG value", fontsize=CONSTANTS.AXIS_LABEL_FONT_SIZE * 0.8)
    ax.set_ylim(min_y, max_y)
    for direction in ['right', 'top', 'left']:
        ax.spines[direction].set_visible(False)

    return (fig, ax)


def conversion_vs_expected_plot(events_df: pd.DataFrame, metric_column: str, prob_column: str,
                                **kwargs) -> go.Figure:
    '''
    Plots the xG conversion distribution for all xG values that exist in the data for the player. Additionally, the
    figure shows lift values, categorized as easy and hard (defined in CONSTANTS.HARD_XG, CONSTANTS.EASY_XG).
    :param events_df: DataFrame of events data according to StatsBomb format
    :param metric_column: the metric column (str) to be evaluated against the prob_column. For example. COLUMNS.GOAL
    :param prob_column: the probability column (str) to be evaluated against the metric_column. For example. COLUMNS.XG
    :param kwargs: metric_display_name[=metric_column]
                    easy_threshold =[CONSTANTS.EASY_XG]
                    hard_threshold[=CONSTANTS.HARD_XG]
    :return: Plotly graph_objects.Figure
    '''
    metric_display_name = kwargs.get('metric_display_name', metric_column)
    easy_threshold, hard_threshold = kwargs.get('easy_threshold', CONSTANTS.EASY_XG), \
                                     kwargs.get('hard_threshold', CONSTANTS.HARD_XG)

    events_df = events_df[events_df[metric_column].notna()]
    events_df['metric_prob_bin'] = events_df[prob_column].apply(lambda x: np.round(x, 1))
    if 'color' not in events_df.columns:
        events_df['color'] = events_df[metric_column].apply(lambda res: COLORS.FOREST_GREEN if res else COLORS.RED)

    # Validate/Convert bool value to int
    events_df[metric_column] = events_df[metric_column].apply(lambda val: int(val))

    # Aggregate probabilities bins with conversion
    aggr_df = events_df[[metric_column, 'metric_prob_bin', prob_column]].groupby(by='metric_prob_bin') \
        .agg({metric_column: [np.mean, np.std], prob_column: [np.mean, np.size]}).reset_index(drop=False)
    aggr_df.columns = ['metric_prob_bin', f"mean_{metric_display_name}", f"std_{metric_display_name}",
                       f"mean_{prob_column}", "count"]

    scatter_colors = [
        COLORS.FOREST_GREEN if aggr_df[f"mean_{metric_display_name}"].iloc[i] >= aggr_df[f"mean_{prob_column}"].iloc[i]
        else COLORS.FIREBRICK for i in
        range(aggr_df[f"mean_{prob_column}"].shape[0])]

    # Lifts text calculation
    lift = np.round(float(np.mean(aggr_df[f"mean_{metric_display_name}"] / aggr_df[f"mean_{prob_column}"])), 2)
    easy_aggr_df = aggr_df[aggr_df[f"mean_{prob_column}"] >= easy_threshold]
    hard_aggr_df = aggr_df[aggr_df[f"mean_{prob_column}"] <= hard_threshold]
    easy_lift = np.round(np.mean(easy_aggr_df[f"mean_{metric_display_name}"] / easy_aggr_df[f"mean_{prob_column}"]), 2)
    hard_lift = np.round(np.mean(hard_aggr_df[f"mean_{metric_display_name}"] / hard_aggr_df[f"mean_{prob_column}"]), 2)

    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        title="Conversion VS. Expected Conversion"
    )

    aggr_df['size'] = aggr_df['count'].apply(lambda c_: min(15, max(8, c_ * 0.5)))
    aggr_df['hover_data'] = aggr_df.apply(lambda row: f"Mean prob {row[f'mean_{prob_column}']}, "
                                                      f"conversion ({row[f'mean_{metric_display_name}']}x baseline"
                                                      f"Num shots observed within bin: {row['count']}",
                                          axis=1)
    aggr_df['color'] = scatter_colors
    fig = go.Figure(layout=layout)

    x = np.arange(0, 1, 0.01)
    fig.add_trace(go.Scatter(x=x, y=x, line=dict(color=COLORS.DIMGREY, width=3, dash='dash'), name=prob_column))
    fig.add_trace(go.Scatter(x=aggr_df[f"mean_{prob_column}"], y=aggr_df[f"mean_{metric_display_name}"],
                             text=[i for i in range(len(x))],
                             name='Actual Conversion',
                             hovertemplate="<b>Bin %{text}</b><br><br>" +
                                           "Mean xG: %{x}<br>" +
                                           "Conversion: %{y}<br>" +
                                           "Num shots in bin: %{marker.size}",
                             mode='lines+markers',
                             line=dict(color=COLORS.FOREST_GREEN, width=2),
                             marker=dict(size=aggr_df['size'],
                                         color=aggr_df['color']),
                             hoveron='points',
                             ))
    fig.update_xaxes(showgrid=True, zeroline=True, showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=True, zeroline=True, showline=True, linewidth=1, linecolor='black')
    txt = [f"xG LIFT index: {lift}", f"Easy shots (>{easy_threshold}) LIFT = {easy_lift}",
           f"Tough shots (<{hard_threshold}) LIFT = {hard_lift}"]
    fig.add_annotation(x=0.8, y=0.3, text=txt[0], showarrow=False)
    fig.add_annotation(x=0.8, y=0.2, text=txt[1], showarrow=False)
    fig.add_annotation(x=0.8, y=0.1, text=txt[2], showarrow=False)

    easy_level_y, tough_level_y = 0.5, 0.5
    text_padding = 0.05
    easy_level_x = np.arange(CONSTANTS.EASY_XG, 1, 0.01)
    tough_shots_x = np.arange(0, CONSTANTS.HARD_XG, 0.01)
    fig.add_annotation(x=np.mean(easy_level_x), y=easy_level_y + text_padding, text='Easy shots',
                       showarrow=False)
    fig.add_trace(go.Scatter(x=easy_level_x, y=[easy_level_y for val in easy_level_x],
                             line=dict(color=COLORS.LIGHTGREY, width=1, dash='dot')))
    fig.add_annotation(x=0, y=tough_level_y + text_padding, text='Tough shots', align='right', showarrow=False)
    fig.add_trace(go.Scatter(x=tough_shots_x, y=[tough_level_y for val in tough_shots_x],
                             line=dict(color=COLORS.LIGHTGREY, width=1, dash='dot')))

    return fig


# Evolution figures
def _build_radar_frame(values, categories):
    categories = [*categories, categories[0]]
    data = go.Scatterpolar(r=values, theta=categories, fill='toself', name='Percentiles')
    layout = go.Layout(
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )
    return data, layout


def player_radar_chart_evolution(players_metrics_by_seasons: pd.DataFrame, player_name: str,
                                 metrics_columns: iter) -> go.Figure:
    '''
    Function animates actions radar charts over seasons
    :param players_metrics_by_seasons: DataFrame of players metrics grouped by season name
    :param player_name: str, name of player to analyze his evolution
    :param metrics_columns: iterator of metrics to pass to the radar chart
    :return: Plotly graph_objects.Figure object
    '''

    # Grouping data by seasons
    players_metrics_by_seasons = players_metrics_by_seasons[
        players_metrics_by_seasons[COLUMNS.PLAYER_NAME] == player_name.lower()]
    seasons = players_metrics_by_seasons[COLUMNS.SEASON_NAME].unique().tolist()
    seasons.sort()

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Season:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # Set radar chart for t=0
    season = seasons[0]
    dataset_by_season = pd.Series(
        players_metrics_by_seasons[players_metrics_by_seasons[COLUMNS.SEASON_NAME] == season].iloc[0])
    radar_zero, layout_dict = _build_radar_frame(dataset_by_season, metrics_columns)

    # Iterating seasons
    fig_data = []
    for season_ in seasons:
        dataset_by_season = pd.Series(
            players_metrics_by_seasons[players_metrics_by_seasons[COLUMNS.SEASON_NAME] == season_].iloc[0])
        radar_zero, layout_dict = _build_radar_frame(dataset_by_season, metrics_columns)
        fig_data.append(radar_zero)

        slider_step = {"args": [[season_], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}],
                       "label": season_,
                       "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    # Create figure
    fig = go.Figure(
        data=[radar_zero],
        layout=go.Layout(width=600, height=600,
                         polar={'radialaxis': {'visible': True}},
                         showlegend=True,
                         title=f"{player_name} evolution over seasons",
                         hovermode="closest",
                         updatemenus=[dict(type="buttons",
                                           buttons=[
                                               dict(label="Play", method="animate",
                                                    args=[None, {"frame": {"duration": 500, "redraw": True},
                                                                 "fromcurrent": True, "transition": \
                                                                     {"duration": 300, "easing": "quadratic-in-out"}}
                                                          ]),
                                               dict(label="Pause", method="animate",
                                                    args=[[None], {"frame": {"duration": 0, "redraw": True},
                                                                   "mode": "immediate",
                                                                   "transition": {"duration": 0}}]),
                                           ],
                                           direction="left",
                                           pad={"r": 10, "t": 87},
                                           showactive=False,
                                           x=0.1,
                                           xanchor="right",
                                           y=0,
                                           yanchor="top"
                                           )],
                         sliders=[sliders_dict]
                         ),
        frames=[go.Frame(data=fig_data[i]) for i in range(len(seasons))]
    )

    return fig


def player_actions_heatmap_evolution(player_actions: pd.DataFrame, matches_metadata: pd.DataFrame) -> go.Figure:
    '''
    Function animates actions heatmaps charts over seasons. It analyzes the frequency and location evolution of actions.
    :param player_actions: DataFrame of events data (StatsBomb format) with the required action types to analyze only.
    :param matches_metadata: DataFrame of matches metadata (see data_handler.py > def matches_metadata)
    :return: Plotly graph_objects.Figure
    '''
    # Merging players actions data with matches_metadata
    player_actions[COLUMNS.MATCH_ID] = player_actions[COLUMNS.MATCH_ID].astype(str)
    player_actions[COLUMNS.MATCH_ID] = player_actions[COLUMNS.MATCH_ID].apply(lambda val: val.split('.')[0])
    matches_metadata[COLUMNS.MATCH_ID] = matches_metadata[COLUMNS.MATCH_ID].astype(str)
    matches_metadata[COLUMNS.MATCH_ID] = matches_metadata[COLUMNS.MATCH_ID].apply(lambda val: val.split('.')[0])
    player_actions_w_metadata = player_actions.merge(matches_metadata, on=COLUMNS.MATCH_ID)

    player_actions_w_metadata = player_actions_w_metadata.sort_values(by=[COLUMNS.SEASON_NAME], ascending=True)
    # Creating figure
    fig = px.density_heatmap(player_actions_w_metadata, x=COLUMNS.START_X, y=COLUMNS.START_Y,
                             nbinsx=34, nbinsy=24, range_x=(-58, 58), range_y=(-39, 39),
                             facet_col=COLUMNS.ACTION_TYPE,
                             color_continuous_scale="reds",
                             title='Player actions heatmap over seasons',
                             animation_frame=COLUMNS.SEASON_NAME)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    bg = PIL.Image.open(f"{CONSTANTS.PITCH_LAB_BACKGROUND}")
    fig.add_layout_image(dict(source=bg, x=0, y=1, sizex=1, sizey=1, sizing="contain", layer='above', opacity=0.2))
    fig.add_layout_image(dict(source=bg, x=1 / 3, y=1, sizex=1, sizey=1, sizing="contain", layer='above', opacity=0.2))
    fig.add_layout_image(dict(source=bg, x=2 / 3, y=1, sizex=1, sizey=1, sizing="contain", layer='above', opacity=0.2))

    return fig


def comparison_conversion_vs_expected_plot(events_df: pd.DataFrame, metric_column: str, prob_column: str,
                                           players: iter, players_shot_counts: dict, **kwargs) -> go.Figure:
    '''
    Compares given players in metric_column conversion VS. prob_column. Produces conversion_vs_expected_plot for each
    player.
    :param events_df: DataFrame of events data according to StatsBomb format
    :param metric_column:  the metric column (str) to be evaluated against the prob_column. For example. COLUMNS.GOAL
    :param prob_column: the probability column (str) to be evaluated against the metric_column. For example. COLUMNS.XG
    :param players: iterable object of player to compare
    :param players_shot_counts: {player_name: shot count (int), ...} for each player in players
    :param kwargs: 'prob_display_name'[=prob_column], 'metric_display_name' [=metric_column]
    :return: Plotly graph_objects.Figure
    '''
    prob_column_label = kwargs.get('prob_display_name', prob_column)
    metric_display_name = kwargs.get('metric_display_name', metric_column)

    events_df = events_df[events_df[metric_column].notna()]
    if 'color' not in events_df.columns:
        events_df['color'] = events_df[metric_column].apply(lambda res: COLORS.FOREST_GREEN if res else COLORS.RED)

    # Split to 0.1 width bins
    events_df['metric_prob_bin'] = events_df[prob_column].apply(lambda x: np.round(x, 1))

    # Convert bool value to int
    events_df[metric_column] = events_df[metric_column].apply(lambda val: int(val))

    aggr_dfs = []
    lifts = {}
    for player_i in players:
        curr_df = events_df.copy().loc[events_df[COLUMNS.PLAYER_NAME] == player_i.lower()]
        curr_df = curr_df[[metric_column, 'metric_prob_bin', prob_column]].groupby(by='metric_prob_bin') \
            .agg({metric_column: [np.mean, np.std], prob_column: [np.mean, np.size]}).reset_index(drop=False)
        curr_df.columns = ['metric_prob_bin', f"mean_{metric_display_name}", f"std_{metric_display_name}",
                           f"mean_{prob_column_label}", "count"]
        curr_df[COLUMNS.PLAYER_NAME] = player_i

        # Lifts text calculation
        if players_shot_counts[player_i.lower()] > 100:
            # AUC
            x = curr_df[f"mean_{prob_column_label}"].tolist()
            y = curr_df[f"mean_{metric_display_name}"].tolist()
            # Add (0,0) and (1, 1) coordinates for an accurate AUC
            lift = 2 * auc([0] + x + [1], [0] + y + [1])
        else:
            # Too few shots -> approximation by dividing the means
            lift = np.round(
                float(np.mean(curr_df[f"mean_{metric_display_name}"] / curr_df[f"mean_{prob_column_label}"])), 2)
        easy_shots_aggr_df = curr_df[curr_df[f"mean_{prob_column_label}"] >= CONSTANTS.EASY_XG]
        hard_shots_aggr_df = curr_df[curr_df[f"mean_{prob_column_label}"] <= CONSTANTS.HARD_XG]
        easy_level_lift = np.round(np.mean(easy_shots_aggr_df[f"mean_{metric_display_name}"] /
                                           easy_shots_aggr_df[f"mean_{prob_column_label}"]), 2)
        hard_level_lift = np.round(np.mean(hard_shots_aggr_df[f"mean_{metric_display_name}"] /
                                           hard_shots_aggr_df[f"mean_{prob_column_label}"]), 2)
        lifts[player_i] = {'lift': lift, 'easy_lift_index': easy_level_lift, 'hard_lift_index': hard_level_lift}
        aggr_dfs.append(curr_df)

    # Aggregate probabilities bins with conversion
    aggr_df = pd.concat(aggr_dfs, axis=0)
    aggr_df.columns = ['metric_prob_bin', f"mean_{metric_display_name}", f"std_{metric_display_name}",
                       f"mean_{prob_column_label}", 'count', COLUMNS.PLAYER_NAME]

    scatter_colors = [
        COLORS.FOREST_GREEN if aggr_df[f"mean_{metric_display_name}"].iloc[i] >=
                               aggr_df[f"mean_{prob_column_label}"].iloc[i]
        else COLORS.FIREBRICK for i in
        range(aggr_df[f"mean_{prob_column_label}"].shape[0])]

    aggr_df['size'] = aggr_df['count'].apply(lambda c_: min(15, max(8, c_ * 0.5)))
    aggr_df['hover_data'] = aggr_df.apply( \
        lambda row: f"Player {row[COLUMNS.PLAYER_NAME]}"
                    f"Mean prob {row[f'mean_{prob_column_label}']}, conversion ({row[f'mean_{metric_display_name}']}x baseline"
                    f"Num shots observed within bin: {row['count']}",
        axis=1)
    aggr_df['color'] = scatter_colors

    x = np.arange(0, 1, 0.01)
    easy_level_x = np.arange(CONSTANTS.EASY_XG, 1, 0.4)
    tough_shots_x = np.arange(0, CONSTANTS.HARD_XG, 0.01)
    text_padding = -0.05

    num_columns = int(len(players) // 2)
    num_rows = int(np.ceil(len(players) / num_columns))
    players_order = list(lifts.items())
    players_order.sort(key=lambda val: val[1]['lift'], reverse=True)

    fig = make_subplots(rows=num_rows, cols=num_columns, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[player_[0] for player_ in players_order],
                        vertical_spacing=0.1, horizontal_spacing=0.01)

    easy_shot_y, tough_shots_y = 0.5, 0.5
    for i, player_i_data in enumerate(players_order):
        player_i = player_i_data[0]
        # In Plotly, rows and columns start from 1, not from zero
        _row = int(np.floor(i / num_columns) + 1)
        _col = 1 + i % num_columns

        curr_aggr_df = aggr_df[aggr_df[COLUMNS.PLAYER_NAME] == player_i]
        fig.add_trace(
            go.Scatter(x=x, y=x, line=dict(color=COLORS.DIMGREY, width=3, dash='dash'), name=prob_column_label),
            row=_row, col=_col)
        fig.add_trace(
            go.Scatter(x=curr_aggr_df[f"mean_{prob_column_label}"], y=curr_aggr_df[f"mean_{metric_display_name}"],
                       # hover_data=aggr_df['hover_data'],
                       text=[i for i in range(len(x))],
                       name='Actual Conversion',
                       hovertemplate="<b>Bin %{text}</b><br><br>" +
                                     f"Mean {prob_column_label}:" + "%{x}<br>" +
                                     "Conversion: %{y}<br>",
                       # "Num observations in bin: %{marker.size}",
                       mode='lines+markers',
                       line=dict(color=COLORS.FOREST_GREEN, width=2),
                       marker=dict(size=curr_aggr_df['size'],
                                   color=curr_aggr_df['color']),
                       hoveron='points',
                       ),
            row=_row, col=_col)
        fig.update_layout(yaxis_range=[0, 1.05])
        fig.update_xaxes(showgrid=True, zeroline=True, showline=True, linewidth=1, linecolor='black',
                         row=_row, col=_col)
        if _col == 1:
            fig.update_yaxes(title_text=f'Mean {metric_display_name}', row=_row, col=_col)
        if _row == num_rows:
            fig.update_xaxes(title_text=f'Mean {prob_column_label}', row=_row, col=_col)

        txt = [f"xG Lift AUC: {np.round(lifts[player_i]['lift'], 2)}",
               f"Easy level LIFT = {lifts[player_i]['easy_lift_index']}",
               f"Tough level LIFT = {lifts[player_i]['hard_lift_index']}"]
        fig.add_annotation(x=0.75, y=0.2, text=txt[0], showarrow=False, row=_row, col=_col)
        fig.add_annotation(x=0.75, y=0.15, text=txt[1], showarrow=False, row=_row, col=_col)
        fig.add_annotation(x=0.75, y=0.1, text=txt[2], showarrow=False, row=_row, col=_col)
        fig.add_annotation(x=0.75, y=0.05, text=f'Num shots: {players_shot_counts[player_i.lower()]}', showarrow=False,
                           row=_row, col=_col)

        if kwargs.get('show_levels_boundaries', False):
            fig.add_annotation(x=np.mean(easy_level_x),
                               y=easy_shot_y + text_padding,
                               text='Easy difficulty',
                               showarrow=False, align='left',
                               row=_row, col=_col)
            fig.add_trace(go.Scatter(x=easy_level_x,
                                     y=[easy_shot_y for val in easy_level_x],
                                     line=dict(color=COLORS.LIGHTGREY, width=1, dash='dot')),
                          row=_row, col=_col)
            fig.add_annotation(x=CONSTANTS.HARD_XG,
                               y=tough_shots_y + text_padding,
                               text='Tough difficulty',
                               showarrow=False, align='left',
                               row=_row, col=_col)
            fig.add_trace(go.Scatter(x=tough_shots_x,
                                     y=[tough_shots_y for val in tough_shots_x],
                                     line=dict(color=COLORS.LIGHTGREY, width=1, dash='dot')),
                          row=_row, col=_col)

    fig.update_layout(title_text=f"Comparison of {prob_column_label} distribution plots", showlegend=False,
                      template='plotly_white')
    fig.update_xaxes(range=[0, 1], dtick=0.1)
    fig.update_yaxes(range=[0, 1], dtick=0.1)

    return fig
