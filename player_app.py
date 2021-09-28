"""
Created on September 5 2021

Streamlit app as UI for delivering skill analysis and Player2Vec representations.

@author: Ofir Magdaci (@Magdaci)

"""

import string
import pandas as pd
import PIL
import streamlit as st

from lib.models import plot_embeddings
from lib.params import COLUMNS, PLAYERS, ANALYSIS_PARAMS, SKILLS, BADGES, CONSTANTS
from lib.app_parts import get_player_events_data, get_teams2players, output_most_similar_players, \
    get_embeddings_objects, get_players_metrics_by_seasons
from lib.data_processing import build_data_objects
from lib.plot import player_radar_chart_evolution, player_actions_heatmap_evolution, plot_metric_by_dimension, \
    conversion_vs_expected_plot
from lib.skill_analysis import radar_chart_w_baselines
from lib.utils import get_player_image

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

with st.spinner(text='Loading events dataset'):
    enriched_events_data, matches_metadata, players_metadata, teams_metadata, players_metrics_df, baselines = \
        build_data_objects(return_objects=True)
    team_2_players = get_teams2players(enriched_events_data)

default_team_name = 'barcelona'
default_player_name = PLAYERS.MESSI
all_teams = list(enriched_events_data[COLUMNS.TEAM_NAME].unique())
all_teams.sort()
all_teams_display = [string.capwords(name_) for name_ in all_teams]

###########################################  Sidebar  ###########################################
with st.sidebar:
    team_name = st.selectbox('Select team', all_teams_display, index=all_teams.index(default_team_name))

    if team_name is None:
        player_name_lower = default_player_name
    else:
        team_name = team_name.lower()
        team_players = team_2_players[team_name]
        team_players.sort()
        team_players_display = [string.capwords(name_) for name_ in team_players]
        default_player_ix = team_players_display.index(string.capwords(default_player_name)) \
            if team_name == default_team_name else 0
        player_name = st.selectbox(f"Select from {string.capwords(team_name)} players:", team_players_display,
                                   index=default_player_ix)
        player_name_lower = player_name.lower()

# Get DataFrame of player's events only
curr_player_events_data = get_player_events_data(enriched_events_data, player_name_lower)

###########################################  Metadata  ###########################################

profile_pic_col, metadata_col = st.beta_columns((1, 2))
metadata_col.title(player_name)
metadata_col.write("Most frequent positions (%):")
player_positions = curr_player_events_data[[COLUMNS.POSITION, COLUMNS.MATCH_ID]].drop_duplicates()[COLUMNS.POSITION]
metadata_col.write(pd.DataFrame(100 * player_positions.value_counts() / player_positions.shape[0]).round(1).head(3))

try:
    curr_player_metadata = players_metadata[player_name]
    metadata_col.write(f"Nationality: {curr_player_metadata['country_name']}")

except KeyError:
    metadata_col.write('Metadata is not available for this player')

####################################  Player metrics area  #####################################
# SCORE OF EACH ACTION TO ANALYZE, USING PERCENTILES
curr_player_metrics = players_metrics_df.loc[player_name_lower].to_dict()

config_col_left, config_center, config_col_right = st.beta_columns((3, 3, 2))
metrics_columns = ANALYSIS_PARAMS.DEFAULT_XG_METRICS
metrics_columns_labels = ANALYSIS_PARAMS.DEFAULT_XG_METRICS_LABELS
skills_metrics = config_col_left.multiselect('Radar chart skills', options=metrics_columns_labels,
                                             default=metrics_columns_labels,
                                             help="Select skills to be included in the radar chart.")
# Get selected baselines from all possible baselines
competitions_baselines = ANALYSIS_PARAMS.BASELINES_TO_USE[COLUMNS.COMPETITION_NAME]
positions_baselines = ANALYSIS_PARAMS.BASELINES_TO_USE[COLUMNS.POSITION]
all_baselines = competitions_baselines + positions_baselines
baselines_to_use = config_center.multiselect('Radar chart baselines', options=all_baselines, default=all_baselines,
                                             help="Select baselines to include in the analyses.")

# Badges (e.g., great finisher, dribbler... use badges PNGs)
percentile_threshold = config_col_right.slider("Badge threshold for percentiles based skills:", 0.0, 1.0, 0.9, 0.05,
                                               help="Players with higher skill percentile value than this threshold "
                                                    "will get an appropriate badge.")
lift_threshold = config_col_right.slider("Badge threshold for lift badge:", 0.0, 3.0, 1.1, 0.025,
                                         help="Players with higher lift value than this threshold will get "
                                              "an appropriate badge.")
player_skills = []
MODE = 'images'
for skill in SKILLS:
    threshold = lift_threshold if skill not in () else percentile_threshold
    if skill in [COLUMNS.XA, COLUMNS.DRIBBLE_WON]:
        col = f"{skill}:mean:percentile"
        threshold = percentile_threshold
    else:
        col = f"{skill}:LIFT"
        threshold = lift_threshold

    if curr_player_metrics[col] > lift_threshold:
        player_skills.append(skill)

st.subheader("Player skills:")
cols = st.beta_columns(6)
for i, skill in enumerate(player_skills):
    if MODE == 'images':
        cols[i].image(PIL.Image.open(BADGES.PATHS[skill]), width=50, caption=BADGES.CAPTIONS[skill])
    else:
        cols[i].write(f' - {BADGES.CAPTIONS[skill]}')

##################################  Radar Chart with baselines  ###################################
fig = radar_chart_w_baselines(players_metrics_df, player_name, baselines, baselines_to_use=all_baselines,
                              show=False, return_fig=True)
st.plotly_chart(fig, use_container_width=True)

######################################  Player events data  #######################################
player_shots = curr_player_events_data[(curr_player_events_data[COLUMNS.ACTION_TYPE] == 'Shot') &
                                       (curr_player_events_data[COLUMNS.LOCATION].notna())]

player_passes = curr_player_events_data[(curr_player_events_data[COLUMNS.ACTION_TYPE] == 'Pass') &
                                        (curr_player_events_data[COLUMNS.LOCATION].notna())]

player_dribbles = curr_player_events_data[(curr_player_events_data[COLUMNS.ACTION_TYPE] == 'Dribble') &
                                          (curr_player_events_data[COLUMNS.LOCATION].notna())]

player_headers = player_shots[player_shots['shot_body_part_name'].isin(['head', 'Head'])]

##########################################  Profile pic  ###########################################

profile_picture = get_player_image(player_name_lower)
profile_pic_col.image(profile_picture, output_format='png')

####################################################################################################
###########################################  Analysis Area  ########################################
####################################################################################################

evol_expander = st.beta_expander("Player evolution analysis", expanded=False)
with evol_expander:
    st.subheader("Player evolution over seasons")
    players_metrics_by_seasons, seasons_baselines = get_players_metrics_by_seasons(enriched_events_data,
                                                                                   matches_metadata)
    figure_columns = metrics_columns + [COLUMNS.PLAYER_NAME, COLUMNS.SEASON_NAME]
    # Skills evolution using radar chart
    fig = player_radar_chart_evolution(players_metrics_by_seasons[figure_columns], player_name, metrics_columns)
    st.plotly_chart(fig, use_container_width=True)

    # Actions frequency and location evolution using heatmaps
    player_actions = pd.concat([player_dribbles, player_shots, player_passes], axis=0)
    fig = player_actions_heatmap_evolution(player_actions, matches_metadata)
    fig_size = (CONSTANTS.PITCH_DIMENSIONS[0] * 3, CONSTANTS.PITCH_DIMENSIONS[1] * 3)
    st.plotly_chart(fig, use_container_width=True)

# ####################################################################################################
# ########################################  Skill Analysis  ##########################################
# ####################################################################################################
xg_expander = st.beta_expander("xG analysis", expanded=True)
with xg_expander:
    metric_column = COLUMNS.GOAL
    prob_column = COLUMNS.XG

    fig11 = conversion_vs_expected_plot(curr_player_events_data, metric_column, prob_column,
                                        metric_display_name=metric_column.capitalize(),
                                        prob_display_name='xG')
    st.plotly_chart(fig11, use_container_width=True)

    fig_r1, ax = plot_metric_by_dimension(curr_player_events_data, COLUMNS.GOAL, COLUMNS.XG, 'body_part',
                                          metric_display_name='xG')
    st.pyplot(fig_r1, use_container_width=True)

####################################################################################################
########################################  Player embeddings  #######################################
####################################################################################################
model_name = "Player2Vec"
st.subheader(f"{model_name} section")

# Filtering configurations
filter_1, filter_2, filter_3, col_4 = st.beta_columns(4)
show_all_players = filter_1.radio("Players to show:", ['All players', 'Team players'], index=0).lower()
color_by = filter_2.radio("Color embeddings by:", ['Player', 'Match'], index=0).lower()
label_players = col_4.checkbox('Label team players', value=False,
                               help='If checked, the team members will be approachable by name via the legend')
if color_by.lower() == 'player':
    player_embeddings_color_attr_labels = {'Player position': COLUMNS.POSITION, 'Player nationality': 'country_name'}
    player_embeddings_color_attr_label = filter_3.radio("Player color attribute:",
                                                        ['Player position', 'Player nationality'], index=0)
    embeddings_color_attr = player_embeddings_color_attr_labels[player_embeddings_color_attr_label]
else:
    embeddings_color_attr = filter_3.radio("Match color attribute:",
                                           [COLUMNS.SEASON_NAME, COLUMNS.COMPETITION_NAME, 'coach', 'game_outcome',
                                            'home_team_country_name'], index=0)

# Get all objects needed for the embeddings part
player_2_vec_model, players_corpus, players_embeddings, players_matches_embeddings = get_embeddings_objects(
    enriched_events_data, players_metadata, model_name=model_name)

with st.spinner(text=f"Loading {show_all_players} embeddings"):
    # Set embedded_vocab - the document names (player, match) and their representation
    embedded_vocab = {players_corpus.documents_names[i]: players_matches_embeddings.vectors_docs[i]
                      for i in range(len(players_corpus.documents_names))}

    # Set plot_players_embeddings - the population for the figure; and players_to_label
    players_to_label = []
    if show_all_players.lower() == 'all players':
        plot_players_embeddings = players_embeddings.copy()
    else:
        plot_players_embeddings = {player_: players_embeddings.loc[player_] for player_ in
                                   players_embeddings.index.copy() if player_.lower() in team_players}
        plot_players_embeddings = pd.DataFrame.from_dict(plot_players_embeddings, orient='index')
        embedded_vocab = {key: val for key, val in embedded_vocab.items() if key[0].lower() in team_players}

    plot_players = list(plot_players_embeddings.index.copy())
    # Forcing naming format to be the same
    if plot_players[0].islower():
        players_metadata = {key.lower(): val for key, val in players_metadata.items()}

# Add players to label
if label_players:
    players_to_label = [_player for _player in plot_players if _player.lower() in team_players]

if color_by == 'match':
    plot_players_embeddings = pd.DataFrame.from_dict(embedded_vocab, orient='index')

    # Get all matches of embedded (player, match) items
    matches_list = [ix.split(players_corpus.separator)[1] for ix in embedded_vocab]
    if '.json' in matches_list[0]:
        matches_list = [match_.split('.')[0] for match_ in matches_list]
    # Colored by matches - get all information about matches
    embeddings_colors = matches_metadata[matches_metadata[COLUMNS.MATCH_ID].isin(matches_list)]
    # Change COLUMNS.MATCH_ID to string & export to dict
    embeddings_colors[COLUMNS.MATCH_ID] = embeddings_colors[COLUMNS.MATCH_ID].astype(str)
    embeddings_colors.set_index(COLUMNS.MATCH_ID, inplace=True)
    embeddings_colors = embeddings_colors[embeddings_color_attr].to_dict()
    # Get the relevant attribute for each match in matches_list.
    # If a match is missing (attribute=np.nan), put 'other' to fill.
    embeddings_colors = [embeddings_colors.get(match, 'other') for match in matches_list]
else:
    # Color by position
    embeddings_colors = [players_metadata[player_][embeddings_color_attr] for player_ in plot_players]

fig = plot_embeddings(plot_players_embeddings, players_to_label, docs_features=True, docs_data=players_metadata,
                      doc_name_separator=players_corpus.separator, model_name=model_name,
                      colors=embeddings_colors,
                      save_fig=False, show=False,
                      title='UMAP projection of PlayerMatch2Vec (player, match)')
# Show embeddings plot
st.plotly_chart(fig, use_container_width=True)

# Show most similar players using the embeddings
if players_embeddings.index[0].islower():
    # Forcing naming format
    player_name = player_name.lower()
curr_player_embeddings = players_embeddings.loc[player_name]
output_most_similar_players(curr_player_embeddings, player_name, model_name=model_name)
