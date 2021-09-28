"""
Created on September 5 2021

Data processing module of football2vec. Contains classes: FootballTokenizer, Corpus.
Also contains the build_data_objects function and its nested function for building the core data objects.

@author: Ofir Magdaci (@Magdaci)

"""

import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import os
import pickle
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

from lib.data_handler import load_all_events_data, load_players_metadata, load_matches_metadata, get_teams_metadata
from lib.language_patterns import check_if_shot_scored, check_if_one_one_one_chance, check_if_shot_outside_box, \
    check_if_dribble_won
from lib.params import COLUMNS, PATHS, ARTIFACTS
from lib.utils import to_metric_centered_coordinates, get_location_bin

tqdm.pandas()


class FootballTokenizer:
    def __init__(self, **kwargs):
        self.tokens_encoder = OrdinalEncoder()
        self.num_x_bins = kwargs.get('num_x_bins', 5)
        self.num_y_bins = kwargs.get('num_y_bins', 5)
        self.actions_to_ignore_outcome = kwargs.get('actions_to_ignore_outcome', ['duel'])

    def tokenize_action(self, action: pd.Series) -> str:
        '''
        Convert action - a record of StatsBomb events data - to a string token
        :param action: Series, a single action
        :return: token - string
        '''
        action_name = action.get('type', action.get(COLUMNS.ACTION_TYPE, np.nan))
        if action_name is np.nan:
            return np.nan
        else:
            action_name = action_name.lower()

        token = f'<{action_name}>'

        # Add location
        if action['location'] is not np.nan and action['location'] is not np.nan:
            if isinstance(action['location'], str):
                x, y = ast.literal_eval(action['location'])
            elif isinstance(action['location'], list) or isinstance(action['location'], tuple):
                x, y = action['location']
            else:
                raise ValueError("Unfamiliar value for action location:", action['location'])
            location_bin = get_location_bin(x, y, num_x_bins=self.num_x_bins, num_y_bins=self.num_y_bins)
            token = f"{location_bin}".replace(" ", "") + token

        return token.replace(" ", "_")


class Corpus:
    def __init__(self, **kwargs):
        '''
        corpus - list of paragraphs. Resolution/ aggregation is determined by self.aggr_columns.
            self.aggr_columns aggregation is executed by string grouping, separated by self.separator
            Tokens of language are stored in vocabulary, and their encodings in vocabulary_ix.
            Transformation from tokens <-> tokens encodings are allowed by self.ix_2_token & self.token_2_ix.
        documents - name of documents, when used for Doc2Vec
        :param kwargs:
        :type kwargs:
        '''
        self.aggr_columns = kwargs.get('aggr_columns', None)
        if self.aggr_columns is None:
            self.aggr_columns = ['match_id', 'period', 'possession']
        self.ft = kwargs.get('tokenizer', FootballTokenizer(
            actions_to_ignore_outcome=kwargs.get('actions_to_ignore_outcome', ['duel'])))

        self.separator = kwargs.get('separator', '-')
        # Init None attributes
        self.corpus = None
        self.vocabulary = None
        self.vocabulary_ix = None
        self.ix_2_token = None
        self.token_2_ix = None
        self.documents_names = None
        self.verbose = kwargs.get('verbose', False)

    def build_corpus(self, events_data: pd.DataFrame, allow_concat_documents_=True, **kwargs) -> pd.DataFrame:
        '''
        Build corpus using given vocab_data. Associates actions with matching tokens, aggregate them to sentences,
                    and then to documents.
        :param events_data: pd.DataFrame of StatsBomb events data
        :param allow_concat_documents_: whether to allow concatenation of sentences to documents if < min length limit
        :param kwargs:
        :return: vocab_data with new 'token' column. All object attributes are updated.
        '''
        if self.verbose:
            print(f"vocab_data size: {events_data.shape}\n")

        vocab_data = events_data.copy()

        if self.verbose: print('\nStart Tokenization')
        vocab_data['token'] = vocab_data.progress_apply(lambda action:
                                                        self.ft.tokenize_action(action),
                                                        axis=1)
        events_data['token'] = vocab_data['token'].copy()
        vocab_data = vocab_data[~vocab_data['token'].isna()]
        if self.verbose:
            print('Done.')
            print(f"Vocab_data size after processing and removing NAs tokens: {events_data.shape}\n")

        vocabulary = [val for val in vocab_data['token'].unique() if val is not np.nan]
        vocabulary.extend(['oov'])
        if self.verbose:
            print(f'Raw length of vocabulary: (including oov)', len(vocabulary))

        # Create mappers of token to index and vice versa
        ix_2_token = dict(enumerate(vocabulary))
        ix_2_token = {str(key): val for key, val in ix_2_token.items()}
        token_2_ix = {val: key for key, val in ix_2_token.items()}

        # Set the appropriate token index for each action
        vocab_data['token_ix'] = vocab_data['token'].apply(
            lambda token: token_2_ix.get(token, token_2_ix['oov']))

        # Keep only columns relevant for sentences grouping
        vocab_data = vocab_data[['token_ix', 'token'] + self.aggr_columns]

        for col in self.aggr_columns:
            vocab_data[col] = vocab_data[col].astype(str)
        vocab_data['aggr_key'] = vocab_data[self.aggr_columns].apply(
            lambda vec: self.separator.join(vec), axis=1)

        # Create sentences and documents
        sentences = vocab_data[['aggr_key', 'token_ix']].groupby('aggr_key')
        sentences = sentences['token_ix'].agg(list).reset_index()
        documents = sentences['aggr_key'].tolist()
        sentences = sentences['token_ix'].tolist()

        sampling_window = kwargs.get('sampling_window', 5)
        corpus = []
        if self.verbose:
            print('\nBuilding sentences...')

        if not allow_concat_documents_:
            # If we can't concatenate sentences --> add to corpus sentences that are longer than min threshold
            self.documents_names = []
            if self.verbose: print('\nBuilding Documents...')
            for i, doc_ in tqdm(enumerate(sentences)):
                if len(doc_) >= sampling_window:
                    corpus.append(doc_[:])
                    self.documents_names.append(documents[i])
            if self.verbose: print('Final number of documents_names:', len(self.documents_names))
        else:
            # Paragraphs can be merged and concatenated
            # If we can concatenate multiple short sentences (shorter than min threshold) to longer sentences > merge
            if self.verbose:
                print('\nConcatenating Documents to build sampling_window sized documents...')
            cum_actions_length = 0
            cum_actions = []

            for sentence_ in tqdm(sentences):
                if len(sentence_) >= sampling_window:
                    corpus.append(sentence_[:])
                else:
                    cum_actions.extend(sentence_[:])
                    cum_actions_length += len(sentence_)

                    if cum_actions_length >= sampling_window:
                        corpus.append(cum_actions[:])
                        cum_actions_length = 0
                        cum_actions = []

        if self.verbose:
            print('Final number of sentences:', len(corpus))

        # Update vocabulary
        corpus_flat = set([subitem for item in corpus for subitem in item if type(item) is list])
        vocaulary_ix = set([token_2_ix[token_] for token_ in vocabulary])

        # Update vocabulary after merging sentences
        vocabulary_ix = corpus_flat.intersection(vocaulary_ix)
        vocabulary = [ix_2_token[token_ix] for token_ix in vocabulary_ix]
        if self.verbose:
            print('Final length of vocabulary:', len(vocabulary))

        # Set class properties
        self.corpus = corpus
        self.vocabulary = vocabulary
        self.vocabulary_ix = vocabulary_ix
        self.ix_2_token = ix_2_token
        self.token_2_ix = token_2_ix

        return events_data


def get_enriched_events_data(force_create=False, verbose=False, save_artifacts=False, **kwargs) -> pd.DataFrame:
    '''
    Build enriched events_data DataFrame. It apply to_metric_centered_coordinates on the data, and adds features:
        -  COLUMNS.START_X, COLUMNS.START_Y
        -  COLUMNS.GOAL, COLUMNS.OUTBOX_SHOT, COLUMNS.FREE_KICK, COLUMNS.HEADER, COLUMNS.DRIBBLE_WON, COLUMNS.PENALTY,
            COLUMNS.ONE-ON-ONE, - bool indicators if the events matches the filter
            COLUMNS.XA - expected goal (xG) of the shot resulted from the current pass (if exists)
    :param force_create: whether to force create or try to load existing file [bool]
    :param save_artifacts: bool, whether to save the artifacts in to params.PATHS.ARTIFACTS
    :return: enriched_events_data
    '''
    path_prefix = kwargs.get('path_prefix', None)
    if path_prefix is not None:
        enriched_events_data_path = os.path.join(path_prefix, PATHS.ENRICH_EVENTS_DATA_PATH)
    else:
        enriched_events_data_path = PATHS.ENRICH_EVENTS_DATA_PATH

    if os.path.exists(enriched_events_data_path) and not force_create:
        if verbose: print('Loading existing enriched_events_data...')
        return pd.read_csv(enriched_events_data_path)
    else:
        if verbose: print('Building enriched_events_data...')
        events_data = load_all_events_data(verbose=verbose)

        # Covert key names to lower case as best practice, despite the confusion it may cause
        for col in [COLUMNS.TEAM_NAME, COLUMNS.PLAYER_NAME]:
            events_data[col] = events_data[col].apply(lambda name_: name_.lower() if isinstance(name_, str) else name_)

        if verbose: print(' - Handling coordinates and location...')
        events_data = to_metric_centered_coordinates(events_data)
        events_data[COLUMNS.START_X] = events_data[COLUMNS.LOCATION].apply(
            lambda val: val[0] if isinstance(val, tuple) or isinstance(val, list) else val)
        events_data[COLUMNS.START_Y] = events_data[COLUMNS.LOCATION].apply(
            lambda val: val[1] if isinstance(val, tuple) or isinstance(val, list) else val)

        if verbose: print(' - Adding xG, xA, OUTBOX_SHOT, etc...')
        events_data[COLUMNS.GOAL] = events_data.apply(lambda event_: float(check_if_shot_scored(event_)), axis=1)
        events_data[COLUMNS.OUTBOX_SHOT] = events_data.apply(
            lambda event_: float(check_if_shot_outside_box(event_)), axis=1)
        events_data[COLUMNS.HEADER] = events_data.apply(
            lambda event_: 1 if pd.notna(event_['shot_body_part_name']) \
                                and event_['shot_body_part_name'].lower() == 'head' else 0, axis=1)
        events_data[COLUMNS.DRIBBLE_WON] = events_data.apply(lambda event_: float(check_if_dribble_won(event_)), axis=1)
        events_data[COLUMNS.PENALTY] = events_data.apply(
            lambda event_: 1 if pd.notna(event_['shot_type_name'])
                                and event_['shot_type_name'].lower() == 'penalty' else 0, axis=1)
        events_data[COLUMNS.ONE_ON_ONE] = events_data.apply(lambda event_:
                                                            float(check_if_one_one_one_chance(event_)), axis=1)

        events_data[COLUMNS.FREE_KICK] = events_data.apply(
            lambda event_: 1 if pd.notna(event_['shot_type_name']) and event_['shot_type_name'].lower() == 'free kick'
            else 0, axis=1)

        # xA = xG of receiver
        events_data[COLUMNS.XA] = events_data['pass_assisted_shot_id'].apply( \
            lambda shot_id: events_data.loc[events_data['id'] == shot_id, COLUMNS.XG].iloc[0] \
                if isinstance(shot_id, str) else np.nan)
        pass_recipient = 'pass_recipient_name'
        events_data[pass_recipient] = events_data[pass_recipient].apply(
            lambda val: val.lower() if isinstance(val, str) else val)

        events_data[COLUMNS.DRIBBLE_WON] = events_data.apply(lambda action: check_if_dribble_won(action), axis=1)
        events_data[COLUMNS.IS_SHOT] = events_data.apply(lambda action: 1 \
            if action[COLUMNS.ACTION_TYPE] == 'Shot' else 0, axis=1)

        if save_artifacts:
            print(f' - Saving to {enriched_events_data_path}...')
            if not os.path.exists(ARTIFACTS):
                if verbose: print('Creating new ARTIFACTS folder')
                os.makedirs(ARTIFACTS)
            events_data.to_csv(enriched_events_data_path)

        return events_data


def get_enriched_players_metadata(events_data, force_create=False, path_prefix='', verbose=False,
                                  save_artifacts=False) -> dict:
    '''
    Combines players metadata given in the dataset and enriches it with events_data (vocab_data) information:
        Adds player_name, team_name, position_name per player (take most frequent), jersey_number (take most frequent)
    :param save_artifacts: whether to save the artifacts in to params.PATHS.ARTIFACTS
    :param force_create: whether to force create or try to load existing file [bool]
    :param events_data: data frame of events data
    :param verbose: print control
    '''
    enriched_players_metadata_path = path_prefix + PATHS.ENRICH_PLAYERS_METADATA_PATH

    if os.path.exists(enriched_players_metadata_path) and not force_create:
        if verbose:
            print(f'\nLoading players metadata')
        with open(enriched_players_metadata_path, 'rb') as f:
            return pickle.load(f)
    else:
        players_metadata = load_players_metadata(force_create=force_create)

        players_metadata['player_name_lower'] = players_metadata[COLUMNS.PLAYER_NAME].apply(
            lambda val: val.lower() if isinstance(val, str) else val)

        vocab_data_cp = events_data[events_data[COLUMNS.PLAYER_NAME].notna()].copy()
        players_2_positions = vocab_data_cp[
            [COLUMNS.PLAYER_NAME, COLUMNS.POSITION, COLUMNS.MATCH_ID]].copy() \
            .drop_duplicates(). \
            groupby([COLUMNS.PLAYER_NAME, COLUMNS.POSITION]). \
            agg({COLUMNS.MATCH_ID: np.size}).reset_index().sort_values(by=COLUMNS.MATCH_ID, ascending=False)
        players_2_positions = players_2_positions.drop_duplicates(subset=[COLUMNS.PLAYER_NAME], keep='first')
        players_2_positions.set_index(COLUMNS.PLAYER_NAME, inplace=True)
        players_2_positions = players_2_positions.to_dict(orient='index')

        players_2_jersey_num = players_metadata[[COLUMNS.PLAYER_NAME, 'jersey_number']].copy(). \
            groupby([COLUMNS.PLAYER_NAME]). \
            agg({'jersey_number': np.size}).reset_index().sort_values(by='jersey_number', ascending=False)
        players_2_jersey_num.set_index(COLUMNS.PLAYER_NAME, inplace=True)
        players_2_jersey_num = players_2_jersey_num.to_dict(orient='index')

        matches_metadata = load_matches_metadata()
        matches_metadata[COLUMNS.MATCH_ID] = matches_metadata[COLUMNS.MATCH_ID].astype(str)
        matches_metadata = matches_metadata.set_index(COLUMNS.MATCH_ID)
        matches_metadata = matches_metadata.to_dict(orient='index')
        vocab_data_cp['match_date'] = vocab_data_cp[COLUMNS.MATCH_ID].apply(
            lambda match_: matches_metadata.get(str(match_).split('.')[0], {'match_date': 'unknown'})['match_date'])
        players_2_teams = vocab_data_cp.sort_values(by=['match_date'], ascending=False)[
            [COLUMNS.PLAYER_NAME, COLUMNS.TEAM_NAME]].copy()
        players_2_teams = players_2_teams.drop_duplicates(subset=[COLUMNS.PLAYER_NAME], keep='first')
        players_2_teams = players_2_teams.set_index(COLUMNS.PLAYER_NAME)
        players_2_teams = players_2_teams.to_dict(orient='index')

        # Add most frequent position name into metadata > position_name
        players_metadata[COLUMNS.POSITION] = players_metadata[COLUMNS.PLAYER_NAME].apply( \
            lambda name_: players_2_positions.get(name_.lower(), {COLUMNS.POSITION: np.nan})[COLUMNS.POSITION])
        players_metadata['jersey_number'] = players_metadata[COLUMNS.PLAYER_NAME].apply( \
            lambda name_: players_2_jersey_num.get(name_, {'jersey_number': np.nan})['jersey_number'])
        players_metadata['team_name'] = players_metadata[COLUMNS.PLAYER_NAME].apply( \
            lambda name_: players_2_teams.get(name_.lower(), {COLUMNS.TEAM_NAME: ''})[COLUMNS.TEAM_NAME])
        players_metadata = players_metadata.drop_duplicates(subset=[COLUMNS.PLAYER_NAME], keep='first')
        players_metadata = players_metadata.set_index(COLUMNS.PLAYER_NAME)
        players_metadata = players_metadata.to_dict(orient='index')

        if verbose:
            print('DONE.')

        if save_artifacts:
            if not os.path.exists(os.path.join(path_prefix, ARTIFACTS)):
                if verbose: print('Creating new ARTIFACTS folder')
                os.makedirs(os.path.join(path_prefix, ARTIFACTS))

            print(f'Saving enrich_players_metadata to {enriched_players_metadata_path}')
            with open(enriched_players_metadata_path, 'wb') as f:
                pickle.dump(players_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        return players_metadata


def create_players_metrics_df(enriched_events_data: pd.DataFrame, matches_metadata: pd.DataFrame, force_create=False,
                              **kwargs) -> (pd.DataFrame, dict):
    '''
    Build a DataFrame of conversion and LIFT stats for players - xG, xA, lifts for each shot type, etc...
    It aggregates all players actions results within a single DataFrame.
    :param enriched_events_data: pd.DataFrame, enriched version of StatsBomb events data (see get_enriched_events_data)
    :param matches_metadata: DataFrame - adds season_name, competition_name for each match in the dataset
    :param force_create: whether to force create or try to load existing file [bool]
    :return: players_metrics_df (pd.DataFrame), baselines (dict) - the metrics output for variety of benchmarks
    '''
    verbose = kwargs.get('verbose', False)
    metrics_df_path = kwargs.get('metrics_df_path', PATHS.PLAYERS_METRICS_PATH)
    save_artifacts = kwargs.get('save_artifacts', False)
    baselines_path = kwargs.get('baselines_path', PATHS.BASELINE_PLAYERS_METRICS_PATH)
    baseline_dimensions = kwargs.get('baseline_dimensions', [COLUMNS.COMPETITION_NAME, COLUMNS.POSITION])
    if (os.path.exists(metrics_df_path) and os.path.exists(baselines_path)) and not force_create:
        if verbose:
            print('\nLoading existing players_metrics_df')
        players_metrics_df = pd.read_csv(metrics_df_path)

        with open(baselines_path, 'rb') as f:
            baselines = pickle.load(f)
    else:
        if verbose: print('Creating players_metrics_df:\n> Columns formatting')

        # Add matches_metadata to enriched_events_data, allowing grouping metrics for baselines
        enriched_events_data[COLUMNS.MATCH_ID] = enriched_events_data[COLUMNS.MATCH_ID].astype(str)
        enriched_events_data[COLUMNS.MATCH_ID] = enriched_events_data[COLUMNS.MATCH_ID].apply(
            lambda val: val.split('.')[0])
        matches_metadata[COLUMNS.MATCH_ID] = matches_metadata[COLUMNS.MATCH_ID].astype(str)
        matches_metadata[COLUMNS.MATCH_ID] = matches_metadata[COLUMNS.MATCH_ID].apply(lambda val: val.split('.')[0])
        enriched_events_data[COLUMNS.ASSISTS] = enriched_events_data[COLUMNS.ASSISTS].apply(lambda val: int(val) \
            if isinstance(val, bool) else val)
        enriched_events_data[COLUMNS.DRIBBLE_WON] = enriched_events_data[COLUMNS.DRIBBLE_WON].astype(float)

        if verbose: print('>> Done.\n> Merging events data with matches metadata')
        enriched_events_data = enriched_events_data.merge(matches_metadata, on=COLUMNS.MATCH_ID)
        if verbose: print('>> Done.\n> Keeping only male players for evaluation')
        enriched_events_data = enriched_events_data[
            enriched_events_data['home_team_home_team_gender'].apply(lambda gender: gender.lower() == 'male')]

        # Aggr dimensions. By default - player_name
        player_dimensions = kwargs.get('player_dimensions', [COLUMNS.PLAYER_NAME])

        if verbose: print(f'>> Done.\n> Creating shot types probabilities and scoring features...')
        # Prepare columns for lift
        enriched_events_data[f'{COLUMNS.SHOOTING}:{COLUMNS.XG}'] = enriched_events_data[COLUMNS.GOAL] * \
                                                         enriched_events_data[COLUMNS.XG]
        enriched_events_data[f'{COLUMNS.SHOOTING}:{COLUMNS.GOAL}'] = enriched_events_data[COLUMNS.GOAL].copy()

        # Sub-categories of shooting
        shots_types = [COLUMNS.HEADER, COLUMNS.OUTBOX_SHOT, COLUMNS.FREE_KICK, COLUMNS.PENALTY, COLUMNS.ONE_ON_ONE]
        for shot_type in tqdm(shots_types):
            # shot_type:xg - the xg of each shot_type event. float x 1 if the event, else 0 or np.nan
            # shot_type:goal - goal by shot_type event. 1 x 1 if the event occur, else 0 or np.nan
            enriched_events_data[f'{shot_type}:{COLUMNS.XG}'] = enriched_events_data[shot_type] * \
                                                                enriched_events_data[COLUMNS.XG]
            enriched_events_data[f'{shot_type}:{COLUMNS.GOAL}'] = enriched_events_data[shot_type] * \
                                                                  enriched_events_data[COLUMNS.GOAL]

        if verbose:
            print('>> Done.\n> Creating Plotly xG distribution plot...')
            fig = px.histogram(enriched_events_data, x=COLUMNS.XG, labels={'x': 'xG', 'y': 'count'},
                               title=f'xG distribution plot:')
            fig.show()
            print('> Aggregating data by player')

        # Calculate players_metrics_df
        metrics_columns = [COLUMNS.XA, COLUMNS.DRIBBLE_WON, COLUMNS.IS_SHOT, COLUMNS.GOAL] + shots_types + \
                          [f'{shot_type}:{COLUMNS.XG}' for shot_type in shots_types] + \
                          [f'{shot_type}:{COLUMNS.GOAL}' for shot_type in shots_types] + \
                          [f'{COLUMNS.SHOOTING}:{COLUMNS.XG}', f'{COLUMNS.SHOOTING}:{COLUMNS.GOAL}']

        # Agg DataFrame to produce the metrics
        columns = player_dimensions + metrics_columns
        players_metrics_df = enriched_events_data[columns].groupby(player_dimensions) \
            .agg([np.mean, np.sum, np.std]).reset_index()
        if len(player_dimensions) == 1:
            players_metrics_df.set_index(players_metrics_df.columns[0], inplace=True)
            columns = [col for col in player_dimensions if col != COLUMNS.PLAYER_NAME]
        else:
            columns = player_dimensions[:]

        # Exclude player_name since it is the index
        for col in metrics_columns:
            columns.extend([f"{col}:mean", f"{col}:sum", f"{col}:std"])
        players_metrics_df.columns = columns

        if verbose:
            print('>> Done.\n> Shots distribution:')
            print(players_metrics_df[f'{COLUMNS.IS_SHOT}:sum'].describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.]))
            print('> Aggregating data by player')

        if verbose: print('>> Done.\n> Removing players with less than min_actions_count shots')

        # Filter players with too few shots
        min_actions_count = kwargs.get('min_actions_count', 30)
        min_subactions_count = kwargs.get('min_subactions_count', 10)
        players_metrics_df = players_metrics_df[players_metrics_df[f"{COLUMNS.IS_SHOT}:sum"] > min_actions_count]
        players_w_data = set(list(players_metrics_df.index))

        if verbose: print('>> Done.\n> Calculating LIFTS')

        for shot_type in shots_types + [COLUMNS.SHOOTING]:
            # Lift by shot type = total goals scored by shot type / total xG achieved by shot type
            players_metrics_df[f'{shot_type}:LIFT'] = players_metrics_df[[f'{shot_type}:{COLUMNS.GOAL}:sum',
                                                                          f'{shot_type}:{COLUMNS.XG}:sum']].apply( \
                lambda row: row[f'{shot_type}:{COLUMNS.GOAL}:sum'] / row[f'{shot_type}:{COLUMNS.XG}:sum']
                if row[f'{shot_type}:{COLUMNS.XG}:sum'] > 0 else 1, axis=1)

            if shot_type in shots_types:
                # For players with sum current shot_type < num_actions_count -> put np.nan
                players_metrics_df[f'{shot_type}:LIFT'] = players_metrics_df.apply( \
                    lambda row: row[f'{shot_type}:LIFT'] if row[f'{shot_type}:sum'] > min_subactions_count
                    else np.nan, axis=1)

            # For players with not examples, we have to fill their lift values. Baseline = 1,
            # means the player performed similar to his achieved xG
            players_metrics_df[f'{shot_type}:LIFT'].fillna(1, inplace=True)

        # Percentile transformer on all metrics averages using QuantileTransformer
        percentiles_columns = [f"{col_}:percentile" for col_ in
                               players_metrics_df.select_dtypes(include='number').columns]

        if verbose: print('>> Done.\n> Applying QuantileTransformer')
        quantile_transformer = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
        percentile_df = quantile_transformer.fit_transform(players_metrics_df.select_dtypes(include='number'), y=None)
        percentile_df = pd.DataFrame(percentile_df, columns=percentiles_columns, index=players_metrics_df.index.copy())
        players_metrics_df = pd.merge(players_metrics_df, percentile_df, left_index=True, right_index=True)

        if verbose:
            print(f'>> Done.\n> Creating Plotly xG LIFT distribution plot...')
            fig = px.histogram(players_metrics_df, x='shooting:LIFT', nbins=100,
                               labels={'x': 'xG Lift', 'y': 'Number of player'},
                               title=f'xG LIFT distribution plot')
            fig.show()
            print('>> Done.\n> Preparing baselines...')

        # Set benchmarks by competition, position name
        baselines = {}
        if len(baseline_dimensions) > 0:
            baseline_columns = []
            for col in metrics_columns:
                baseline_columns.extend([f"{col}:mean", f"{col}:sum", f"{col}:std"])

            # Baseline aggregate the relevant events according to the baseline_dimension.
            # Only player with sufficient amount of shots should be considered: players_w_data
            baselines_population = enriched_events_data[enriched_events_data[COLUMNS.PLAYER_NAME].isin(players_w_data)]
            for baseline_dimension in baseline_dimensions:
                baselines[baseline_dimension] = baselines_population[[baseline_dimension] + metrics_columns]\
                    .copy().groupby(baseline_dimension).agg([np.mean, np.sum, np.std])
                # baselines[baseline_dimension].set_index(baselines[baseline_dimension].columns[0], inplace=True)
                baselines[baseline_dimension].columns = baseline_columns
                for shot_type in shots_types + [COLUMNS.SHOOTING]:
                    # Lift by shot type = total goals scored by shot type / total xG achieved by shot type
                    baselines[baseline_dimension][f'{shot_type}:LIFT'] = baselines[baseline_dimension][
                                                                             f'{shot_type}:{COLUMNS.GOAL}:sum'] \
                                                                         / baselines[baseline_dimension][
                                                                             f'{shot_type}:{COLUMNS.XG}:sum']
                baseline_percentile_df = quantile_transformer.transform(
                    baselines[baseline_dimension].select_dtypes(include='number'))
                baseline_percentile_df = pd.DataFrame(baseline_percentile_df, columns=percentiles_columns,
                                                      index=baselines[baseline_dimension].index.copy())
                baselines[baseline_dimension] = pd.merge(baselines[baseline_dimension], baseline_percentile_df,
                                                         left_index=True,
                                                         right_index=True)
                baselines[baseline_dimension].columns = players_metrics_df.columns

        # Sort columns for easy read
        sorted_cols = list(players_metrics_df.columns)
        sorted_cols.sort()
        players_metrics_df = players_metrics_df[sorted_cols]

        if verbose:
            print('> Done.')
        if save_artifacts:
            print('Saving artifacts...')
            if not os.path.exists(ARTIFACTS):
                if verbose: print('Creating new ARTIFACTS folder')
                os.makedirs(ARTIFACTS)
            players_metrics_df.to_csv(metrics_df_path)
            if len(baseline_dimensions) > 0:
                with open(baselines_path, 'wb') as f:
                    pickle.dump(baselines, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Completed: create_players_metrics_df')
    return players_metrics_df, baselines


def get_players_metrics_df(enriched_events_data, matches_metadata, verbose=False, save_artifacts=False, **kwargs) -> (
        pd.DataFrame, dict):
    '''
    :param enriched_events_data: pd.DataFrame, enriched version of StatsBomb events data (see get_enriched_events_data)
    :param matches_metadata: DataFrame - adds season_name, competition_name for each match in the dataset
    :param verbose: prints control
    :param save_artifacts: whether to export artifacts (players_metrics_df, baselines dict) or not.
    :return: metrics DataFrame and baselines metrics dict
    '''
    if (not os.path.exists(PATHS.PLAYERS_METRICS_PATH)) or (not os.path.exists(PATHS.BASELINE_PLAYERS_METRICS_PATH)):
        df, baselines = create_players_metrics_df(enriched_events_data, matches_metadata, verbose=verbose,
                                                  save_artifacts=save_artifacts, **kwargs)
        return df, baselines

    df = pd.read_csv(PATHS.PLAYERS_METRICS_PATH)
    df.set_index(df.columns[0], inplace=True)
    with open(PATHS.BASELINE_PLAYERS_METRICS_PATH, 'rb') as f:
        baselines = pickle.load(f)
    return df, baselines


def build_data_objects(return_objects=False, verbose=False, **kwargs):
    '''
    Builds all required data object for the UI and further data analysis
    :param kwargs: verbose, force_create, plotly_export, save_artifacts
    :param return_objects: whether to return the created data object or not (for saving the artifacts, for example)
    :return: None if not return_objects, else, return all data objects created:
        - enriched_events_data, matches_metadata, players_metadata, players_metrics_df
    '''
    if verbose:
        print('Starting build_data_objects.\nCreating enriched events_data...')
    enriched_events_data = get_enriched_events_data(**kwargs)
    if verbose:
        print('\n- Done. Starting creating matches metadata...')
    matches_metadata = load_matches_metadata()
    if verbose:
        print('\n- Done. Starting creating players metadata...')
    players_metadata = get_enriched_players_metadata(enriched_events_data)
    if verbose:
        print('\n- Done. Starting creating players and baselines metrics df...')
    players_metrics_df, baselines = get_players_metrics_df(enriched_events_data, matches_metadata, **kwargs)
    if verbose:
        print('\n- Done. Starting creating teams metadata...')
    teams_metadata = get_teams_metadata(**kwargs)
    if verbose:
        print('\n- Done.')

    if kwargs.get('save_artifacts', False):
        if verbose:
            print('\n- Saving artifacts')
        enriched_events_data.to_csv(PATHS.ENRICH_EVENTS_DATA_PATH)
        matches_metadata.to_csv(PATHS.MATCHES_METADATA_PATH)
        with open(PATHS.ENRICH_PLAYERS_METADATA_PATH, 'wb') as f:
            pickle.dump(players_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        teams_metadata.to_csv(PATHS.TEAMS_METADATA_PATH)
        players_metrics_df.to_csv(PATHS.PLAYERS_METRICS_PATH)
        with open(PATHS.BASELINE_PLAYERS_METRICS_PATH, 'wb') as f:
            pickle.dump(baselines, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print('\n- Completed: build_data_objects')
    if return_objects:
        return enriched_events_data, matches_metadata, players_metadata, teams_metadata, players_metrics_df, baselines
