"""
Created on September 5 2021

This module contain many utils functions for more comfortable Streamlit app writing, keeping your code leaner.

@author: Ofir Magdaci (@Magdaci)

"""

import pandas as pd
import numpy as np
import streamlit as st
import os
import pickle
from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors

from lib.data_processing import create_players_metrics_df, get_enriched_players_metadata
from lib.data_processing import get_enriched_events_data as _get_enriched_events_data
from lib.data_processing import get_players_metrics_df as _get_players_metrics_df
from lib.data_handler import load_matches_metadata
from lib.models import Player2Vec
from lib.params import COLUMNS, PATHS, CONSTANTS, ARTIFACTS, MODELS_ARTIFACTS


import sys
from lib import data_processing
sys.modules['data_processing'] = data_processing

@st.cache(allow_output_mutation=True)
def get_enriched_events_data():
    return _get_enriched_events_data()


@st.cache(allow_output_mutation=True)
def get_doc2vec_data(model_name):
    player_2_vec_model = Doc2Vec.load(os.path.join(MODELS_ARTIFACTS, f"{model_name}.model"))
    players_matches_embeddings = KeyedVectors.load(os.path.join(MODELS_ARTIFACTS, f"{model_name}.wordvectors"),
                                                   mmap='r')

    corpus_obj_path = os.path.join(MODELS_ARTIFACTS, f"{model_name}_corpus.pickle")
    with open(corpus_obj_path, 'rb') as f:
        players_corpus = pickle.load(f)

    players_embeddings_path = os.path.join(MODELS_ARTIFACTS, f"{model_name}_embeddings.pickle")
    with open(players_embeddings_path, 'rb') as f:
        players_embeddings = pickle.load(f)

    return player_2_vec_model, players_matches_embeddings, players_corpus, players_embeddings


def get_embeddings_objects(enriched_events_data, players_metadata, model_name='Player2Vec'):
    required_artifacts = [os.path.join(MODELS_ARTIFACTS, f"{model_name}.model"),
                          os.path.join(MODELS_ARTIFACTS, f"{model_name}.wordvectors"),
                          os.path.join(MODELS_ARTIFACTS, f"{model_name}_corpus.pickle"),
                          os.path.join(MODELS_ARTIFACTS, f"{model_name}_embeddings.pickle")
                          ]
    missing_artifacts = []
    for required_path in required_artifacts:
        if not os.path.exists(required_path):
            missing_artifacts.append(required_path)

    if len(missing_artifacts) > 0:
        st.warning(f"Not all models artifacts are available in the artifacts directory: {ARTIFACTS}.\n"
                   f"Missing artifacts: {missing_artifacts}\n"
                   f"Please run main.py or download the data-package (see README file)")
        if st.button("Create on-the-fly",
                     help="This will build all models via the UI. Unless your machine is very strong, "
                          "we recommend to avoid this action and follow steps in the README file"):
            player_2_vec_model, events_data, players_corpus, players_embeddings, players_matches_embeddings = Player2Vec(
                enriched_events_data, players_metadata, force_similarities=True, save_artifacts=False)
            return player_2_vec_model, events_data, players_corpus, players_embeddings, players_matches_embeddings
        else:
            st.write('Waiting for action...')
            return None, None, None, None
    else:
        # Load all required artifacts
        player_2_vec_model, players_matches_embeddings, players_corpus, players_embeddings = get_doc2vec_data(
            model_name)

    return player_2_vec_model, players_corpus, players_embeddings, players_matches_embeddings


@st.cache(allow_output_mutation=True)
def get_players_metrics_by_seasons(events_df: pd.DataFrame, matches_metadata):
    metrics_by_seasons_path = PATHS.PLAYERS_METRICS_BY_SEASON
    seasons_baslines_path = PATHS.BASELINE_BY_SEASONS_METRICS_PATH
    if os.path.exists(metrics_by_seasons_path):
        print('\nLoading existing players_metrics_by_seasons')
        players_metrics_by_seasons = pd.read_csv(metrics_by_seasons_path)
        with open(seasons_baslines_path, 'rb') as f:
            baselines = pickle.load(f)
    else:
        print('Starting: CREATE players_metrics_by_seasons')
        players_metrics_by_seasons, baselines = create_players_metrics_df(events_df,
                                                                          matches_metadata,
                                                                          player_dimensions=[COLUMNS.PLAYER_NAME,
                                                                                             COLUMNS.SEASON_NAME],
                                                                          save_artifacts=True,
                                                                          baselines_path=seasons_baslines_path,
                                                                          baseline_dimensions=[],
                                                                          metrics_df_path=metrics_by_seasons_path)
    return players_metrics_by_seasons, baselines


@st.cache(allow_output_mutation=True)
def get_player_events_data(events_data: pd.DataFrame, player_name_: str):
    events_data_ = events_data[events_data[COLUMNS.PLAYER_NAME] == player_name_]
    return events_data_


@st.cache(allow_output_mutation=True)
def get_players_metadata(events_data=None) -> dict:
    if os.path.exists(PATHS.ENRICH_PLAYERS_METADATA_PATH):
        with open(PATHS.ENRICH_PLAYERS_METADATA_PATH, 'rb') as f:
            enriched_players_metadata = pickle.load(f)
        return enriched_players_metadata
    else:
        enriched_players_metadata = get_enriched_players_metadata(events_data)
        return enriched_players_metadata


@st.cache(allow_output_mutation=True)
def get_matches_metadata(save_artifacts=False):
    if not os.path.exists(PATHS.MATCHES_METADATA_PATH):
        df = load_matches_metadata(save_artifacts=save_artifacts)
        return df

    return pd.read_csv(PATHS.MATCHES_METADATA_PATH)


@st.cache(allow_output_mutation=True)
def get_teams2players(events_data):
    path = os.path.join(ARTIFACTS, "team_2_players.pickle")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            team_2_players = pickle.load(f)
    else:
        team_2_players = events_data[[COLUMNS.TEAM_NAME, COLUMNS.PLAYER_NAME]].drop_duplicates() \
            .groupby(COLUMNS.TEAM_NAME).agg(list)[COLUMNS.PLAYER_NAME].to_dict()
        team_2_players = {key: [val_ for val_ in val if val_ is not np.nan] for key, val in team_2_players.items()}
        with open(path, 'wb') as f:
            pickle.dump(team_2_players, f, protocol=pickle.HIGHEST_PROTOCOL)
    return team_2_players


@st.cache(allow_output_mutation=True)
def get_get_docs_similarities(model_name):
    docs_similarities_path = os.path.join(MODELS_ARTIFACTS, f"{model_name}_docs_similarities.pickle")
    if os.path.exists(docs_similarities_path):
        with open(docs_similarities_path, 'rb') as f:
            docs_similarities = pickle.load(f)
    else:
        raise FileNotFoundError(f"{model_name}_docs_similarities.pickle does not exist. "
                                f"Please run main.py or Player2Vec to create it.")
    return docs_similarities


@st.cache(allow_output_mutation=True)
def get_players_metrics_df(events_data, matches_metadata, verbose=False):
    return _get_players_metrics_df(events_data, matches_metadata, verbose=verbose)


def output_most_similar_players(curr_player_embeddings, player_name: str, model_name: str = 'Player2Vec'):
    if curr_player_embeddings.shape[0] > 0:
        st.subheader(f'Top 10 similar players to {player_name}')
        docs_similarities = get_get_docs_similarities(model_name)
        # Force naming format in case of difference
        if player_name.islower():
            # (player 1 , player 2) : similarity
            docs_similarities = {(key[0].lower(), key[1].lower()): val for key, val in docs_similarities.items()}
        cosine_col, euclidean_col = st.beta_columns(2)

        # Show most similar players by cosine similarity and euclidean distance (the lower the value the higher is the
        # similarity). We start from 1, as the most similar is the player himself.
        with cosine_col:
            curr_player_cosine_sim = {key[1]: val['cosine'] for key, val in docs_similarities.items() if
                                      key[0] == player_name}
            top_cosine_similarities = pd.Series(curr_player_cosine_sim).sort_values(ascending=False)
            top_cosine_similarities = pd.Series(top_cosine_similarities[1:12].index, name='Player name')
            st.write('by *cosine similarity*:')
            st.write(top_cosine_similarities)

        with euclidean_col:
            curr_player_euclidean_sim = {key[1]: val['euclidean'] for key, val in docs_similarities.items() if
                                         key[0] == player_name}
            top_euclidean_similarities = pd.Series(curr_player_euclidean_sim).sort_values(ascending=True)
            top_euclidean_similarities = pd.Series(top_euclidean_similarities[1:12].index, name='Player name')
            st.write('by *euclidean distance*:')
            st.write(top_euclidean_similarities)
    else:
        st.write('No embedding representation is available for this player')


def get_footer():
    st.write(CONSTANTS.CREDITS_TEXT)
