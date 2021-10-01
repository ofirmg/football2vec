"""
Created on September 5 2021

This module covers skill analysis figures presented in Data-Driven Evaluation of Football Players' Skills,
posted on Towards Data Science.

@author: Ofir Magdaci (@Magdaci)

"""

import os
import chart_studio
import chart_studio.plotly as py
import pandas as pd
import plotly.graph_objects as go

from lib.data_handler import load_matches_metadata
from lib.data_processing import get_enriched_events_data, get_players_metrics_df
from lib.params import PLAYERS, COLUMNS, ANALYSIS_PARAMS, PLOTLY_USERNAME, PLOTLY_API_KEY
from lib.plot import radar_chart, comparison_conversion_vs_expected_plot


def skill_comparison_analysis(players_metrics: pd.DataFrame, events_data: pd.DataFrame, players: iter = None,
                              metrics_columns=ANALYSIS_PARAMS.DEFAULT_XG_METRICS,
                              metrics_columns_labels=ANALYSIS_PARAMS.DEFAULT_XG_METRICS_LABELS, **kwargs) \
        -> (go.Figure, dict):
    '''
    Performs comparison plot of conversion vs expected, for the given players
    :param players_metrics: a DataFrame of conversion and LIFT stats for players - lifts for each shot type, etc...
    :param events_data: DataFrame of events data to use.
    :param metrics_columns: columns names from players_metrics_df to use as metrics for the analysis.
                            Default value ANALYSIS_PARAMS.DEFAULT_XG_METRICS
    :param metrics_columns_labels: labels for metrics to use for the analysis, subset of players_metrics_df coloumns.
                            Default value ANALYSIS_PARAMS.DEFAULT_XG_METRICS_LABELS
    :param players: iterator of players names (strings) to iterate
    :return: Plotly figure
    '''
    if players is None:
        players = [PLAYERS.MESSI, PLAYERS.LUIS_SUAREZ, PLAYERS.GRIEZMANN,
                   PLAYERS.COUTINHO, PLAYERS.NEYMAR, PLAYERS.PIQUE]

    metric_column = COLUMNS.GOAL
    prob_column = COLUMNS.XG
    if events_data is None:
        events_data = get_enriched_events_data()

    players_lowercase = [val.lower() for val in players]
    players_shot_counts = players_metrics.loc[players_lowercase, f'{COLUMNS.IS_SHOT}:sum'].to_dict()
    fig = comparison_conversion_vs_expected_plot(events_data, metric_column, prob_column,
                                                 players, players_shot_counts,
                                                 metric_display_name=metric_column.capitalize(),
                                                 prob_display_name='xG')
    # Save all skill numbers into the output dictionary
    players_stats = {}
    for player in players:
        # SCORE OF EACH ACTION TO ANALYZE, USING PERCENTILES
        curr_player_metrics = players_metrics.loc[player.lower()].to_dict()
        players_stats[player] = {metrics_columns_labels[metrics_columns.index(metric_)]:
                                     [curr_player_metrics[metric_]] for metric_ in metrics_columns}

    if kwargs.get('plotly_export', False):
        chart_studio.tools.set_credentials_file(username=PLOTLY_USERNAME, api_key=PLOTLY_API_KEY)
        py.plot(fig, filename='Comparison of xG Lifts plots', auto_open=True)

    if kwargs.get('show', False):
        print('Comparison of xG Lifts plots')
        fig.show()

    return fig, players_stats


def radar_chart_w_baselines(players_metrics: pd.DataFrame, player_name: str, baselines_metrics: list = None,
                            metrics_columns=None, metrics_columns_labels=None, show: bool = True,
                            return_fig: bool = False, plotly_export=False, baselines_to_use: list = None):
    '''
    Produces a radar chart for player metrics with the addition of baselines' metrics
    :param players_metrics: a DataFrame of conversion and LIFT stats for players - lifts for each shot type, etc...
    :param player_name: player to analyze and use for the radar chart
    :param baselines_metrics: dictionary of baselines' metrics
    :param metrics_columns: columns names from players_metrics_df to use as metrics for the analysis.
                            Default value ANALYSIS_PARAMS.DEFAULT_XG_METRICS
    :param metrics_columns_labels: labels for metrics to use for the analysis, subset of players_metrics_df coloumns.
                            Default value ANALYSIS_PARAMS.DEFAULT_XG_METRICS_LABELS
    :param show: bool, whether to show the radar figure chart or not
    :param return_fig: bool, whether to return the radar figure chart or not
    :param baselines_to_use: filters the baselines, if not None
    :return: Plotly.graph_objects.Figure, in case return_fig set to True
    '''
    if metrics_columns is None:
        metrics_columns = ANALYSIS_PARAMS.DEFAULT_XG_METRICS
        if metrics_columns_labels is None:
            metrics_columns_labels = ANALYSIS_PARAMS.DEFAULT_XG_METRICS_LABELS
    elif metrics_columns_labels is None:
        metrics_columns_labels = metrics_columns[:]

    player_metrics = players_metrics.loc[player_name.lower()].to_dict()

    # Formatting
    radar_data = pd.DataFrame({metrics_columns_labels[metrics_columns.index(metric_)]: [100 * player_metrics[metric_]]
                               for metric_ in metrics_columns})
    radar_data['name'] = player_name
    for baseline in baselines_metrics:
        if 0 <= baselines_metrics[baseline].iloc[0, 0] <= 1:
            baselines_metrics[baseline] = baselines_metrics[baseline][metrics_columns] * 100

    # Filter baselines if required
    if baselines_to_use is not None:
        for baseline_dim in baselines_metrics:
            relevant_baselines = [val for val in baselines_metrics[baseline_dim].index if val in baselines_to_use]
            baselines_metrics[baseline_dim] = baselines_metrics[baseline_dim].loc[relevant_baselines]
    fig = radar_chart(radar_data, baselines=baselines_metrics)

    if plotly_export:
        chart_studio.tools.set_credentials_file(username=PLOTLY_USERNAME, api_key=PLOTLY_API_KEY)
        py.plot(fig, filename=f"{player_name.replace(' ', '_')}_skill_radar_chart", auto_open=True)

    if show:
        print(f'radar chart for {player_name}')
        fig.show()
    if return_fig:
        return fig


plotly_export = False
save_artifacts = False

if __name__ == '__main__':
    os.chdir('../')
    events_df = get_enriched_events_data(verbose=True, save_artifacts=save_artifacts)
    matches_metadata = load_matches_metadata(verbose=True)
    players_metrics_df, baselines = get_players_metrics_df(events_df, matches_metadata, save_artifacts=save_artifacts,
                                                           min_actions_count=30, min_subactions_count=10,
                                                           verbose=True)
    all_baselines = ANALYSIS_PARAMS.BASELINES_TO_USE[COLUMNS.COMPETITION_NAME] + ANALYSIS_PARAMS.BASELINES_TO_USE[
        COLUMNS.POSITION]
    radar_chart_w_baselines(players_metrics_df, PLAYERS.NEYMAR, baselines, baselines_to_use=all_baselines,
                            plotly_export=plotly_export)
    radar_chart_w_baselines(players_metrics_df, PLAYERS.MESSI, baselines, baselines_to_use=all_baselines,
                            plotly_export=plotly_export)
    fig, players_stats = skill_comparison_analysis(players_metrics_df, events_df, plotly_export=plotly_export,
                                                   show=True)
