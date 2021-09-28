"""
Created on September 5 2021

This module holds all types of explainers used here: https://towardsdatascience.com/a-deep-dive-into-the-language-of-football-2a2984b6bd21.
It can be easily run as demonstrated in explain.py.

@author: Ofir Magdaci (@Magdaci)

"""

import os
import pickle
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import umap
from gensim.models import Doc2Vec, Word2Vec
from tqdm import tqdm

from lib.data_processing import Corpus
from lib.language_patterns import in_box_scoring_pattern, out_box_scoring_pattern, dribble_pattern, \
    flank_dribble_pattern, \
    forward_pressure_pattern, ANDPattern, ORPattern, _search, \
    is_normal_goal_token, get_tokens_by_regex_pattern, passes_forward_pattern, passes_to_right_pattern, \
    passes_from_right_flank_pattern
from lib.models import plot_embeddings
from lib.params import COLUMNS, PLAYERS, PATHS, SHORTNAMES, DEBUG, ARTIFACTS, MODELS_ARTIFACTS


def estimate_doc_vector(doc2vec_model, doc_: list, steps=10, repetitions=10, norm=True) -> np.array:
    '''
    Infer a given doc_ <repetitions> times and avearaging the results
    :param doc2vec_model: Doc2Vec model to use
    :param doc_: list of tokens/words (indexes) in the langugage
    :param steps: int, number of inference steps to pass to the Doc2Vec model
    :param repetitions: int, number of times to infer the doc_ vector for the mean vector
    :type norm: bool, whether to normalize the output vector or not
    :return: inferred vector (np.array)
    '''
    res = []
    for i in range(repetitions):
        res.append(doc2vec_model.infer_vector(doc_, steps=steps))

    mean_res = np.vstack(res).mean(axis=0)
    if norm:
        return mean_res / np.linalg.norm(mean_res)
    else:
        return mean_res


def create_baseline_player(player_match2vec, players_in_position: list, num_docs_to_use: float, players_corpus) -> (
        np.array, np.array):
    # Sample |S| random matches of players with the matching position_name
    docs_to_ix = {val.lower(): key for key, val in enumerate(players_corpus.documents_names)}
    players_in_position = [val for val in players_in_position if val.lower() in docs_to_ix]
    player_matches_docs = np.random.choice(players_in_position, int(num_docs_to_use))
    player_matches_docs_ix = [docs_to_ix[val] for val in player_matches_docs]

    # Infer vector
    all_baseline_vectors = np.vstack([np.array(player_match2vec[key]) for key in player_matches_docs_ix])
    baseline_vector = all_baseline_vectors.mean(axis=0)

    return baseline_vector, all_baseline_vectors, player_matches_docs_ix


def modify_doc(doc: list, interventions: iter, corpus, limit_interventions=None) -> [list, float, float]:
    '''
    Function receives a doc and modify it by a set of interventions
    :param doc: list of strings
    :param interventions: [{'pattern': regex pattern to identify relevant tokens,
                            'probability': probability to replace pattern,
                            'switch_from': string to replace,
                            'switch_to': string to replace to
    :return: edited doc (list).
    '''
    modified_doc = []
    interventions_count = 0
    modified_token_is_oov_count = 0
    for i, token_ix in enumerate(doc):
        token_was_matched = False
        token_ = corpus.ix_2_token[token_ix]
        for intervention_ in interventions:
            if _search(intervention_['pattern'], token_):
                if np.random.rand() <= intervention_['probability']:
                    # Check limit_interventions
                    if limit_interventions is not None and limit_interventions < interventions_count:
                        break

                    # If token switch occurs:
                    token_was_matched = True
                    modified_token = token_.replace(intervention_['switch_from'], intervention_['switch_to'])
                    if modified_token in corpus.token_2_ix:
                        # The new token is a valid token in the language
                        interventions_count += 1
                        modified_doc.append(corpus.token_2_ix[modified_token])
                        break
                    else:
                        # The new token is out-of-vocabulary
                        modified_token_is_oov_count += 1
                        token_was_matched = False

        # If the token was added so far - add it as-is, without any interventions
        if not token_was_matched:
            modified_doc.append(corpus.token_2_ix[token_])

    return modified_doc, interventions_count, modified_token_is_oov_count


def enrich_doc(doc: list, interventions: iter, players_corpus, limit_interventions=None) -> (list, float):
    '''
    Function receives a doc and adding tokens to according to set of rules
    :param doc: list of strings
    :param interventions: [{'pattern': regex pattern to identify relevant tokens,
                            '<added_action>': name of the action to be added (e.g., '<dribble'>).
                                              Is compared to next_action to validate we do not chain identical actions.
                             'probability': probability to add token,
                             'token_builder': a function that takes relevant token (that matched an intervention)
                                                and return a new token after it
    :return: enriched doc (list).
    '''

    # Iterate
    enriched_doc = []
    intervections_count = 0
    modified_token_is_oov_count = 0
    for i, token_ix in enumerate(doc):
        token_ = players_corpus.ix_2_token[token_ix]
        next_token = players_corpus.ix_2_token[doc[i + 1]] if i + 1 < len(doc) else None
        token_matched = False

        for intervention_ in interventions:
            # Check if pattern match current token
            if _search(intervention_['pattern'], token_):
                # Check if next token is not already the added action (irrelevant if 'skip' activated)
                if next_token is not None and intervention_['<added_action>'] in next_token \
                        and not intervention_.get('skip', False):
                    continue
                # Decide if to append token
                if np.random.rand() <= intervention_['probability']:
                    # Check limit_interventions
                    if limit_interventions is not None and limit_interventions < intervections_count:
                        break

                    intervections_count += 1
                    # Add the current token
                    token_matched = True
                    # Check if intervention means skip the token
                    if intervention_.get('skip', False):
                        break

                    enriched_doc.append(token_ix)

                    # Decide relevant params of the enriched token - e.g., extract proper location
                    token_to_add = intervention_['token_builder'](token_,
                                                                  **intervention_.get('token_builder_args', {}))

                    if token_to_add in players_corpus.token_2_ix:
                        # Add the enriched token
                        enriched_doc.append(players_corpus.token_2_ix[token_to_add])
                        break
                    else:
                        # The new token is out-of-vocabulary
                        modified_token_is_oov_count += 1
                        intervections_count -= 1
                        enriched_doc.pop()
                        token_matched = False

        if not token_matched:
            enriched_doc.append(token_ix)

    return enriched_doc, intervections_count, modified_token_is_oov_count


class ModelExplainer(object):
    pass


class ActionAnalogies(ModelExplainer):
    '''
    ActionAnalogies heavily rely on naming conventions used when building the corpus and the language models.
    However, no error will be raised when deviating these guidelines, and empty output will be returned.
    '''

    def __init__(self, action2vec: Word2Vec, corpus: Corpus, action2vec_embeddings: pd.DataFrame,
                 events_data: pd.DataFrame, **kwargs):
        '''
        :param action2vec: the model to use most_similar() method
        :param action2vec_embeddings: pandas Dataframe of players embeddings (index = player name)
        :param corpus:
        :param kwargs
            - repetitions: number of times to infer docs and average the results, for mitigating randomness
            - k: number of similar tokens to return
        '''

        # Data arguments
        self.min_matches_per_player = kwargs.get('min_matches_per_player', 10)
        self._plot = kwargs.get('plot', False)
        self.verbose = kwargs.get('verbose', False)
        self.model_name = kwargs.get('model_name', 'Player2Vec')
        self.corpus = corpus
        self.action2vec = action2vec
        self.action2vec_embeddings = action2vec_embeddings
        self.vocabulary = self.action2vec_embeddings.index.copy().tolist()
        self.events_data = events_data

        # Analysis arguments
        self.k = kwargs.get('k', 10)
        self.num_examples_per_analogy = kwargs.get('num_examples_per_analogy', 5)

        # supported_skills Dict:
        #   key=skill: str, value=function that return True/False if given token is relevant to skill
        self._supported_tokens_types = {'passes': lambda token: '<pass>' in token,
                                        'in_box_scoring': lambda token: re.match(in_box_scoring_pattern, token),
                                        'out_box_scoring': lambda token: re.match(out_box_scoring_pattern, token),
                                        'dribbling': lambda token: re.match(dribble_pattern, token),
                                        'flank_dribbling': lambda token: re.match(flank_dribble_pattern, token),
                                        'pressure': lambda token: re.match(forward_pressure_pattern, token),
                                        'carry': lambda token: '<carry>' in token,
                                        'offsides': lambda token: '<offside>' in token
                                        }
        self.skills_docs = {}
        self.actions_families = {}
        self.build_tokens_families()

    def get_supported_tokens_families(self):
        return list(self._supported_tokens_types.keys())

    def run_analogy(self, _actions_population: list, _token_modifier_from: str, _token_modifier_to: str,
                    analogy_name='unnamed analogy'):
        '''
        Runs the analogy -
         1. sample token A1 from _actions_population,
         2. modify A1 to A2, by replacing the _token_modifier_from in A with _token_modifier_to\
         3. Sample second token B1, modify it to B2 in the same manner
         4. Find most similar tokens, using action2vec.most_similar to the phrase: A1 - A2 + B2
         Expected answer: B1, or tokens similar to B1
        :param _actions_population: list of language tokens to randomly sample A1, B1
        :param _token_modifier_from: str to change into _token_modifier_to
        :param _token_modifier_to: str to change the _token_modifier_from to
        :param analogy_name: string, name of analysis
        '''
        # Filter actions if they modified version is not in the vocabulary
        actions_population_filtered = [val for val in _actions_population if
                                       val.replace(_token_modifier_from, _token_modifier_to) in self.vocabulary]
        examples_results = []
        if self.verbose:
            print(f"\nAnalogy: {analogy_name}")
        for i in tqdm(range(min(self.num_examples_per_analogy, int(np.floor(len(actions_population_filtered) / 2))))):
            # Get token and find the matching token for the analogy
            A1 = actions_population_filtered[i]
            A2 = A1.replace(_token_modifier_from, _token_modifier_to)
            B1 = actions_population_filtered[-(i + 1)]
            B2 = B1.replace(_token_modifier_from, _token_modifier_to)

            # Transform to normalized vectors
            A1_ix = self.corpus.token_2_ix[A1]
            A2_ix = self.corpus.token_2_ix[A2]
            B2_ix = self.corpus.token_2_ix[B2]

            # king (A1) - queen(A2) = man(B1)-woman(B2) --> woman (B2) + king (A1) - man (B1) = queen(A2)
            # --> positive=['woman', 'king'], negative=['man'] -->
            similar = self.action2vec.most_similar(positive=[A1_ix, B2_ix], negative=[A2_ix])
            if self.verbose:
                print(f' - A1 = {A1} \n - A2 = {A2} \n - B2 =  {B2}')
                print(f" - B1: {self.k} most similar tokens:")
                for sim_ in similar[:self.k]:
                    print(f"   - {self.corpus.ix_2_token[sim_[0]], np.round(sim_[1], 4)}")
                print()
            examples_results.append({'analogy': f' - A1 = {A1} \n - A2 = {A2} \n - B2 =  {B2}',
                                     f'{self.k}_most_similar': similar[:self.k]})

        if self.verbose:
            print('----------------------------------------------------------------')
        return examples_results

    def run_dual_analogy(self, pop_A1, pop_A2, pop_B2, analogy_name=''):
        '''
        This analogy operates between two different tokens, rather getting the second toke by modifying an existing one
        :param pop_A1: words population (list) to sample token A1 from
        :param pop_A2: words population (list) to sample token A2 from
        :param pop_B2: words population (list) to sample token B2 from
        :param analogy_name: string, name of the analogy
        :return: list of most similar words, for each num_examples_per_analogy
        '''
        examples_results = []

        if self.verbose:
            print(f"\nAnalogy: {analogy_name}")
        for i in tqdm(range(self.num_examples_per_analogy)):
            # Get token and find the matching token for the analogy
            A1 = np.random.choice(pop_A1, 1)[0]
            A2 = np.random.choice(pop_A2, 1)[0]
            B2 = np.random.choice(pop_B2, 1)[0]

            # Transform to normalized vecs
            A1_ix = self.corpus.token_2_ix[A1]
            A2_ix = self.corpus.token_2_ix[A2]
            B2_ix = self.corpus.token_2_ix[B2]

            # king-queen = man-woman --> woman + king - man = queen --> positive=['woman', 'king'], negative=['man']
            similar = self.action2vec.most_similar(positive=[A1_ix, B2_ix], negative=[A2_ix])
            if self.verbose:
                print(f' - A1 = {A1} \n - A2 = {A2} \n - B2 =  {B2}')
                print(f" - B1: {self.k} most similar tokens:")
                for sim_ in similar[:self.k]:
                    print(f"   - {self.corpus.ix_2_token[sim_[0]], np.round(sim_[1], 4)}")
            examples_results.append({'analogy': f' - A1 = {A1} \n - A2 = {A2} \n - B2 =  {B2}',
                                     f'{self.k}_most_similar': similar[:self.k]})

        return examples_results

    def build_tokens_families(self):
        self.actions_families['passes'] = [word for word in self.vocabulary if
                                           '<pass>' in word and 'incomplete' not in word
                                           and 'type_na=' not in word and 'aerial_won' not in word]
        self.actions_families['shots'] = [word for word in self.vocabulary if '<shot>' in word]
        self.actions_families['right_foot_shots'] = [word for word in self.actions_families['shots']
                                                     if "body_pa=right" in word]
        self.actions_families['saved_shots'] = [word for word in self.actions_families['shots'] if "=saved" in word]
        inbox_foot_scoring = get_tokens_by_regex_pattern(self.actions_families['shots'], in_box_scoring_pattern)
        self.actions_families['inbox_foot_scoring'] = [token for token in inbox_foot_scoring
                                                       if '|normal|' in token and '_foot' in token and '=goal' in token]
        self.actions_families['short_passes'] = [word for word in self.actions_families['passes'] if "short|" in word]
        self.actions_families['foot_clearances'] = [word for word in self.vocabulary
                                                    if '<clearance>' in word and '_foot' in word]
        self.actions_families['ground_passes'] = [word for word in self.actions_families['passes']
                                                  if "|ground" in word]
        self.actions_families['passes_to_right'] = get_tokens_by_regex_pattern(self.actions_families['passes'],
                                                                               passes_to_right_pattern)
        self.actions_families['passes_forward'] = get_tokens_by_regex_pattern(self.actions_families['passes'],
                                                                              passes_forward_pattern)
        self.actions_families['headers'] = [word for word in self.actions_families['shots'] if "head" in word]
        self.actions_families['passes_from_right_flank'] = get_tokens_by_regex_pattern(self.actions_families['passes'],
                                                                                       passes_from_right_flank_pattern)
        self.actions_families['cross_from_right_flank'] = [token for token in self.actions_families['passes_to_right']
                                                           if "|high" in token]
        self.actions_families['ground_passes_from_right_flank'] = [token for token in
                                                                   self.actions_families['passes_from_right_flank']
                                                                   if '|ground' in token]
        self.actions_families['forward_passes'] = get_tokens_by_regex_pattern(self.actions_families['passes'],
                                                                              re.compile(
                                                                                  "\(4\/5,3\/5\)\<pass\>\:\( \^ \|"))

    def check_model_support(self, analogy_action_family: str):
        if len(self.actions_families[analogy_action_family]) > 1:
            return True
        else:
            return False

    def actions_analogies_similarities(self, analogies: [dict]):
        '''
        Using Word2Vec most_similar method to find best candidates for model analogies
        Printing analogies results
        '''
        res = {}
        for analogy in analogies:
            analogy_name = analogy.get('analogy_name', str(analogy))
            if self.check_model_support(analogy['action_population']):
                res[analogy_name] = self.run_analogy(self.actions_families[analogy['action_population']],
                                                     analogy['token_modifier_from'], analogy['token_modifier_to'],
                                                     analogy_name=analogy_name)
            else:
                print(f"{analogy['action_population']} is not supported by the given vocabulary")
        return res

    def actions_dual_analogies_similarities(self, analogies: [dict]):
        '''
        Using Word2Vec most_similar method to find best candidates for model analogies
        Printing analogies results
        Here, we expect two populations for tokens to be sampled from
        '''
        res = {}
        for analogy in analogies:
            analogy_name = analogy.get('analogy_name', str(analogy))
            pop_A1_support, pop_A2_support = self.check_model_support(analogy['pop_A1']), \
                                             self.check_model_support(analogy['pop_A2'])
            if pop_A1_support and pop_A2_support:
                res[analogy['analogy_name']] = self.run_dual_analogy(self.actions_families[analogy['pop_A1']],
                                                                     self.actions_families[analogy['pop_A2']],
                                                                     self.actions_families[analogy['pop_B2']],
                                                                     analogy_name=analogy_name)
            else:
                if not pop_A1_support:
                    print(f"{analogy['pop_A1']} is not supported by the given vocabulary")
                if not pop_A2_support:
                    print(f"{analogy['pop_A2']} is not supported by the given vocabulary")
            return res

    def default_run(self):
        results = {'analogies': None, 'dual_analogies': None}
        analogies = [
            dict(action_population='passes_to_right', token_modifier_from="- >", token_modifier_to="- <",
                 analogy_name='Pass direction analogy'),

            dict(action_population='passes_forward', token_modifier_from="^ |", token_modifier_to="v |",
                 analogy_name='Pass direction analogy 2'),

            dict(action_population='passes', token_modifier_from="body_pa=right", token_modifier_to="body_pa=left",
                 analogy_name='Pass foot analogy'),

            dict(action_population='passes', token_modifier_from="body_pa=right_foot", token_modifier_to="body_pa=head",
                 analogy_name='Pass head analogy'),

            dict(action_population='passes', token_modifier_from='|ground', token_modifier_to="|high",
                 analogy_name='Pass height'),

            dict(action_population='short_passes', token_modifier_from='short|', token_modifier_to="long|",
                 analogy_name='Pass distance'),

            dict(action_population='saved_shots', token_modifier_from="=saved", token_modifier_to="=goal",
                 analogy_name='Shot score'),

            dict(action_population='right_foot_shots', token_modifier_from="body_pa=right_foot",
                 token_modifier_to="body_pa=head", analogy_name='Shot body part'),

            dict(action_population='forward_passes', token_modifier_from="4/5", token_modifier_to="3/5",
                 analogy_name='Ball goes forward learning')
        ]

        results['analogies'] = self.actions_analogies_similarities(analogies)

        #  A clearance by foot from ground pass ~ head clearance from cross
        dual_analogies = [
            dict(pop_A1='ground_passes_from_right_flank', pop_A2="foot_clearances", pop_B2="cross_from_right_flank",
                 analogy_name='Header to cross from flank ~ shot from a pass from flank')]
        results['dual_analogies'] = self.actions_dual_analogies_similarities(dual_analogies)

        return results


class PlayersAnalogies(ModelExplainer):
    ''' PlayersAnalogies requires players names to work with, assuming named-based representation. '''

    def __init__(self, player_match2vec: Doc2Vec, players_vectors: pd.DataFrame, corpus: Corpus, **kwargs):

        self.player_match2vec = player_match2vec
        self.corpus = corpus
        self.players_vectors = players_vectors
        self.match_sampling = kwargs.get('match_sampling', 10)
        self.num_similar = kwargs.get('num_similar', 10)
        self.export_artifacts = kwargs.get('export_artifacts', False)
        self.verbose = kwargs.get('verbose', False)

        self.player_to_matches = {
            player_: [doc_ for doc_ in corpus.documents_names if player_ in doc_]
            for player_ in self.players_vectors.index}

        self.docs_to_ix = {val: key for key, val in enumerate(corpus.documents_names)}

    def build_artificial_doc(self):
        pass

    def _players_analogy(self, A1, A2, B2, analogy_name='A1 - A2 + B2 ~ ?'):
        similar_docs_names = []
        similar_docs_values = []
        for i in range(self.match_sampling):
            A1_random_match_ix = self.docs_to_ix[np.random.choice(self.player_to_matches[A1])]
            A1_random_vec = self.player_match2vec[A1_random_match_ix]

            A2_random_match_ix = self.docs_to_ix[np.random.choice(self.player_to_matches[A2])]
            A2_random_vec = self.player_match2vec[A2_random_match_ix]

            B2_random_match_ix = self.docs_to_ix[np.random.choice(self.player_to_matches[B2])]
            B2_random_vec = self.player_match2vec[B2_random_match_ix]

            similar = self.player_match2vec.most_similar(positive=[A1_random_vec, B2_random_vec],
                                                         negative=[A2_random_vec],
                                                         topn=100)
            similar_docs_names.extend([int(item[0]) for item in similar])
            similar_docs_values.extend([item[1] for item in similar])

        mean_similar = pd.DataFrame({'doc_ix': similar_docs_names, 'cosine_similarity': similar_docs_values,
                                     'doc_name': [self.corpus.documents_names[ix] for ix in similar_docs_names]
                                     })
        mean_similar[COLUMNS.PLAYER_NAME] = mean_similar['doc_name'].apply(
            lambda str_: str_.split(self.corpus.separator)[0])
        mean_similar = mean_similar[[COLUMNS.PLAYER_NAME, 'cosine_similarity']].groupby(by=[COLUMNS.PLAYER_NAME]) \
            .agg({'cosine_similarity': [np.size, np.mean]})
        mean_similar.columns = ['count', 'mean']
        mean_similar = mean_similar.loc[mean_similar['count'] >= 5] \
            .sort_values(by=['mean'], ascending=False).head(self.num_similar)
        if self.verbose:
            print('Analogy:', analogy_name)
            print(f' - A1 = {A1} \n - A2 = {A2} \n - B2 =  {B2}')
            print(f" - B1: {self.num_similar} most similar tokens:")
            print(mean_similar)
            print()

        return mean_similar

    def players_analogies_analysis(self, triplets: [dict]):
        '''
        Perform players analogies by the given triplets spcification
        :param triplets: {'A1': <player_name: str, 'A2': <player_name: str, 'B2': <player_name: str,
                        analogy_name: str=''}, ..., }
        :type triplets: list of dicts
        :return: {analogy_name: most_similar pd.DataFrame}
        :rtype: dict
        '''
        res = {}
        for run_config in triplets:
            analogy_name = run_config.get('analogy_name',
                                          f"{run_config['A1']} - {run_config['A2']} + {run_config['B2']}")
            if self.verbose:
                print(analogy_name)

            res[analogy_name] = self._players_analogy(run_config['A1'], run_config['A2'], run_config['B2'],
                                                      analogy_name=analogy_name)

        if self.export_artifacts:
            if not os.path.exists(PATHS.EXPLAINERS):
                os.makedirs(PATHS.EXPLAINERS)
            for phrase_ in res:
                res[phrase_].columns = [phrase_, 'cosine_sim']
                res[phrase_].to_csv(os.path.join(PATHS.EXPLAINERS, f'Analogy:{phrase_}.csv'))

        return res

    def default_run(self):
        triplets = [
            {'A1': PLAYERS.PIQUE, 'A2': PLAYERS.LUIS_SUAREZ, 'B2': PLAYERS.BENZEMA,
             'analogy_name': 'Defender to striker: Pique - Suarez + Benzema ~ ?'},

            {'A1': PLAYERS.GRIEZMANN, 'A2': PLAYERS.DEMBELE, 'B2': PLAYERS.NEYMAR,
             'analogy_name': 'Dribble skill: Griezmann - Dembele + Neymar ~ ?'},

            {'A1': PLAYERS.JORDI_ALBA, 'A2': PLAYERS.PIQUE, 'B2': PLAYERS.SERGIO_RAMOS,
             'analogy_name': 'Player position: Alba - Pique + Ramos ~ ?'},

            {'A1': PLAYERS.MANE, 'A2': PLAYERS.SALAH, 'B2': PLAYERS.NEYMAR,
             'analogy_name': 'Player position: Mane - Salah + Neymar ~ ?'},

            {'A1': PLAYERS.JORDI_ALBA, 'A2': PLAYERS.NEYMAR, 'B2': PLAYERS.SALAH,
             'analogy_name': 'Player forward position: Alba - Neymar + Salah ~ ?'},

        ]

        return self.players_analogies_analysis(triplets)


class PlayerSkillsExplainer(ModelExplainer):
    '''
    PlayerSkillsExplainer is especially strict with following naming guidelines.
    Alternatively, you can modify the language patterns in language_patterns.py and corresponding namings in:
    explain.py > PlayerSkillsExplainer class.
    '''

    def __init__(self, player_match2vec: Doc2Vec, players_corpus: Corpus, players_metadata: dict,
                 players_vectors: pd.DataFrame,
                 events_data: pd.DataFrame, **kwargs):

        # Data arguments
        self.combined_variations_plot = kwargs.get('combined_variations_plot', False)
        self.entries_std_per_player = {}
        self.plotly_export = kwargs.get('plotly_export', False)
        self._plot = kwargs.get('plot', False)
        self.verbose = kwargs.get('verbose', False)
        self.save_artifacts = kwargs.get('save_artifacts', False)
        self.model_name = kwargs.get('model_name', 'Player2Vec')
        self.players_metadata = players_metadata
        self.players_corpus = players_corpus
        self.player_match2vec = player_match2vec
        self.players_vectors = players_vectors
        self.events_data = events_data

        # Analysis arguments
        self.steps = kwargs.get('steps', 10)
        self.k = kwargs.get('k', 10)
        self.add_baseline_bias = kwargs.get('add_baseline_bias', False)
        self.dimensions_reducer = umap.UMAP().fit(players_vectors)

        # Utils objects
        self.orig_corpus_players = set(list(self.players_vectors.index.copy()))
        self.docs_to_ix = {val: key for key, val in enumerate(self.players_corpus.documents_names)}
        self.player_to_matches = {player_: [doc_ for doc_ in players_corpus.documents_names if player_ in doc_]
                                  for player_ in players_vectors.index}
        self.player_to_matches_docs = {
            player_: [players_corpus.corpus[int(self.docs_to_ix[match_])] for match_ in matches_]
            for player_, matches_ in self.player_to_matches.items()}

        self.positions = events_data[COLUMNS.POSITION].unique()
        self.positions = [val for val in self.positions if isinstance(val, str)]
        self.actions_mapper = {'modify': self.create_player_variation, 'enrich': self.create_player_enriched_variation}

        # Run arguments
        self._player = None
        self._position_name = None
        self._skill_name = None
        self._variation_action = None
        self._skill_improvements = None
        self._sign = None
        self._player_matches = None
        self._baseline_vector = None
        self._last_action = None
        self._next_action = None

        self.baselines_vectors = None
        self.baselines_docs_ix = None

        self.skill_patterns = {}
        self.skill_success_patterns = {}
        self.modify_patterns = {}
        self.enrich_patterns = {}
        self.variations_stats = {}
        self.build_baseline_vectors()
        self.build_patterns()

    def get_all_skills(self):
        return ['dribble', '-dribble', 'shot', '-shot', 'dribble+shot', '-dribble-shot',
                '-right_foot_use', 'pass_to_back', 'high_passes',
                'enrich_dribble', 'enrich_shots', 'enrich_inbox_shots', 'enrich_goals', 'enrich_interceptions',
                'reduce_dribble', 'reduce_shots']

    def build_patterns(self):
        self.skill_patterns['dribble'] = ANDPattern(['<dribble>'], [])
        self.skill_success_patterns['dribble'] = ANDPattern(['=complete'], [])
        self.skill_patterns['-dribble'] = ANDPattern(['<dribble>'], [])
        self.skill_success_patterns['-dribble'] = ANDPattern(['=complete'], [])
        self.skill_patterns['shot'] = ANDPattern(['<shot>'], [])
        self.skill_success_patterns['shot'] = ANDPattern(['=goal'], [])
        self.skill_patterns['-right_foot_use'] = ORPattern(['<pass>', '<shot>'], [])
        self.skill_success_patterns['-right_foot_use'] = ANDPattern(['left_foot'], [])
        self.skill_patterns['pass'] = ANDPattern(['<pass>'], [])
        self.skill_success_patterns['pass'] = ANDPattern([], ['outcome'])
        self.skill_patterns['interception'] = ANDPattern(['<interception>'], [])
        self.skill_success_patterns['interception'] = ORPattern(['success_in_play', 'won'], [])
        self.skill_patterns['enrich_interceptions'] = self.skill_patterns['interception']
        self.skill_success_patterns['enrich_interceptions'] = self.skill_success_patterns['interception']
        self.skill_patterns['dribble+shot'] = self.skill_patterns['shot']
        self.skill_success_patterns['dribble+shot'] = self.skill_patterns['shot']
        self.skill_patterns['-dribble-shot'] = self.skill_patterns['shot']
        self.skill_success_patterns['-dribble-shot'] = self.skill_patterns['shot']
        self.skill_patterns['enrich_dribble'] = self.skill_patterns['dribble']
        self.skill_success_patterns['enrich_dribble'] = self.skill_success_patterns['dribble']
        self.skill_patterns['reduce_dribble'] = self.skill_patterns['dribble']
        self.skill_success_patterns['reduce_dribble'] = self.skill_success_patterns['dribble']
        self.skill_patterns['enrich_shots'] = self.skill_patterns['shot']
        self.skill_success_patterns['enrich_shots'] = self.skill_success_patterns['shot']
        self.skill_patterns['enrich_shots'] = self.skill_patterns['shot']
        self.skill_success_patterns['enrich_shots'] = self.skill_success_patterns['shot']
        self.skill_patterns['reduce_shots'] = self.skill_patterns['shot']
        self.skill_success_patterns['reduce_shots'] = self.skill_success_patterns['shot']
        self.skill_patterns['enrich_inbox_shots'] = self.skill_patterns['shot']
        self.skill_success_patterns['enrich_inbox_shots'] = self.skill_success_patterns['shot']
        self.skill_patterns['enrich_goals'] = self.skill_patterns['shot']
        self.skill_success_patterns['enrich_goals'] = self.skill_success_patterns['shot']
        self.skill_patterns['pass_to_back'] = self.skill_patterns['pass']
        self.skill_success_patterns['pass_to_back'] = ANDPattern(['( v'], [])
        self.skill_patterns['high_passes'] = self.skill_patterns['pass']
        self.skill_success_patterns['high_passes'] = ANDPattern(['high'], [])

        self.modify_patterns['shot'] = [{'pattern': ANDPattern(['<shot>', 'outcome=blocked'], []),
                                         'switch_from': 'outcome=blocked',
                                         'switch_to': 'outcome=goal'},
                                        {'pattern': ANDPattern(['<shot>', 'outcome=wayward'], []),
                                         'switch_from': 'outcome=wayward',
                                         'switch_to': 'outcome=goal'},
                                        {'pattern': ANDPattern(['<shot>', 'outcome=saved'], []),
                                         'switch_from': 'outcome=saved',
                                         'switch_to': 'outcome=goal'},
                                        {'pattern': ANDPattern(['<shot>', 'outcome=off_t'], []),
                                         'switch_from': 'outcome=off_t',
                                         'switch_to': 'outcome=goal'},
                                        ]
        self.modify_patterns['pass_to_back'] = [{'pattern': ANDPattern(['<pass>', '|low'], []),
                                                 'switch_from': '|low',
                                                 'switch_to': '|high'}]

        self.modify_patterns['high_passes'] = [{'pattern': ANDPattern(['<pass>', '|ground'], []),
                                                'switch_from': '|ground',
                                                'switch_to': '|high'},
                                               {'pattern': ANDPattern(['<pass>', '|low'], []),
                                                'switch_from': '|low',
                                                'switch_to': '|high'},

                                               {'pattern': ANDPattern(['<pass>', '|med'], []),
                                                'switch_from': '|med',
                                                'switch_to': '|high'}
                                               ]

        self.modify_patterns['-shot'] = [{'pattern': ANDPattern(['<shot>', 'outcome=goal'], []),
                                          'switch_from': 'outcome=goal',
                                          'switch_to': 'outcome=wayward'}]

        self.modify_patterns['dribble'] = [{'pattern': ANDPattern(['<dribble>', 'outcome=incomplete'], []),
                                            'switch_from': 'incomplete',
                                            'switch_to': 'complete'}]

        self.modify_patterns['-dribble'] = [{'pattern': ANDPattern(['<dribble>', 'outcome=complete'], []),
                                             'switch_from': 'complete',
                                             'switch_to': 'incomplete'}]
        self.modify_patterns['-dribble-shot'] = self.modify_patterns['-dribble'] + self.modify_patterns['-shot']
        self.modify_patterns['dribble+shot'] = self.modify_patterns['dribble'] + self.modify_patterns['shot']

        self.modify_patterns['-right_foot_use'] = [{'pattern': ORPattern(['left_foot'], []),
                                                    'switch_from': 'left_foot',
                                                    'switch_to': 'right_foot'}]
        self.enrich_patterns['enrich_interceptions'] = [{'pattern': ORPattern([f"<pass>"], []),
                                                         '<added_action>': "<interception>",
                                                         'token_builder': self._build_interception,
                                                         'add_before_pattern': True,
                                                         }]
        self.enrich_patterns['reduce_dribble'] = [
            # Give 2-3 cases where we can add dribbles - find manually in the data
            {'<added_action>': "<>", 'pattern': ANDPattern(['<dribble>'], []), 'skip': True},
        ]
        self.enrich_patterns['reduce_shots'] = [
            # Give 2-3 cases where we can add dribbles - find manually in the data
            {'pattern': ANDPattern(['<shot>'], []), 'skip': True, '<added_action>': "<>"}
        ]

        self._skill_name = 'Shot'
        shots_last_action_distribution = self.analyze_action_context()
        self.enrich_patterns['enrich_shots'] = [
            # Give 2-3 cases where we can add dribbles - find manually in the data
            {'pattern': ORPattern([f"<{action_type.lower()}>"], []),
             '<added_action>': "<shot>",
             # token_builder: function that adds the relevant token according to the last success intervention
             'token_builder': self._build_shot,
             'token_builder_args': {},
             } for action_type in shots_last_action_distribution.index[:5]]

        self.enrich_patterns['enrich_inbox_shots'] = [
            # Give 2-3 cases where we can add dribbles - find manually in the data
            {'pattern': ORPattern([f"<{action_type.lower()}>"], []),
             '<added_action>': "<shot>",
             # token_builder: function that adds the relevant token according to the last success intervention
             'token_builder': self._build_shot,
             'token_builder_args': dict(inbox_shot=True),
             } for action_type in shots_last_action_distribution.index[:5]]

        self.enrich_patterns['enrich_goals'] = [
            # Give 2-3 cases where we can add dribbles - find manually in the data
            {'pattern': ORPattern([f"<{action_type.lower()}>"], []),
             '<added_action>': "<shot>",
             'token_builder': self._build_shot,
             'token_builder_args': dict(goal=True),
             } for action_type in shots_last_action_distribution.index[:5]]

        self._skill_name = 'Dribble'
        dribbles_last_action_distribution = self.analyze_action_context()
        self.enrich_patterns['enrich_dribble'] = [{'pattern': ORPattern([f"<{action_type.lower()}>"], []),
                                                   '<added_action>': "<dribble>",
                                                   # token_builder: function that adds the relevant token according to the situation
                                                   # For dribble: Get position of last action, and add a successful dribble
                                                   'token_builder': self._build_dribble,
                                                   'token_builder_args': {},
                                                   } for action_type in dribbles_last_action_distribution.index[:5]]
        for skill in self.get_all_skills():
            self.variations_stats[skill] = {}

    def players_skills_analysis(self, phrases: list):
        '''

        :param phrases: list of dicts: {'action': 'modify'/'enrich',
                                        'skill_name': e.g. 'dribble',
                                        'player_name': str, full & exact name as appeared in the StatsBomb dataset
        :return: no return, just update the Explainer object and produce plots
        '''
        for phrase_ in phrases:
            self._player_matches = None
            self._player = phrase_[COLUMNS.PLAYER_NAME]
            self._position_name = phrase_[COLUMNS.POSITION]
            self._baseline_vector = self.baselines_vectors[self._position_name]

            skills_to_iterate = phrase_['skill_name']
            if not isinstance(skills_to_iterate, list):
                skills_to_iterate = [skills_to_iterate]
            for i, _skill_name in enumerate(skills_to_iterate):
                self._skill_name = _skill_name
                self._sign = '+' if '-' not in _skill_name else '-'

                if 'enrich' in self._skill_name or 'reduce' in self._skill_name:
                    self._variation_action = 'enrich'
                    self._skill_improvements = self.enrich_patterns[self._skill_name]
                else:
                    self._variation_action = 'modify'
                    self._skill_improvements = self.modify_patterns[self._skill_name]

                if 'baseline' in self._player:
                    self._player_matches = self.baselines_docs_ix[self._position_name]

                self.players_vectors = self.players_vectors.drop_duplicates()
                variations_similarities, variations_docs = self._skill_analysis()
                for variation_, variation_docs_ in variations_docs.items():
                    self.variations_stats[self._skill_name][variation_] = self.calc_skill_stats( \
                        variation_docs_, self.skill_patterns[self._skill_name],  # event pattern
                        self.skill_success_patterns[self._skill_name])  # event success pattern

            # Calculate std of vector representation player's variations. Display vec entries sorted by std DESC
            for skill in skills_to_iterate:
                if len(self.variations_stats[skill]) == 0:
                    continue
                rel_players_vectors = self.players_vectors.copy()
                rel_players_vectors['groupby'] = [val.split(self.players_corpus.separator)[0] for val in
                                                  rel_players_vectors.index.copy()]
                rel_players_vectors = rel_players_vectors.loc[
                    [ix for ix in self.players_vectors.index.copy() if skill in ix]]
                self.entries_std_per_player[skill] = rel_players_vectors.std(axis=0).sort_values(ascending=False)
                if self.verbose:
                    print(f"For the {skill} skill, most changes representation indexes:")
                    print(self.entries_std_per_player[skill])
        if self.combined_variations_plot:
            variations_names = [val for val in self.players_vectors.index if '_enrich_' in val or '_modify_' in val]
            plot_embeddings(self.players_vectors, variations_names,
                            docs_features=True,
                            docs_data=self.players_metadata,
                            doc_name_separator=self.players_corpus.separator, model_name=self.model_name,
                            highlight_selected=False,
                            title=f'UMAP projection of the skill players and players variations',
                            dimensions_reducer=self.dimensions_reducer,
                            show=self.verbose, save_fig=self.save_artifacts, plotly_export=self.plotly_export)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=False,
                            subplot_titles=['Actions modifications, Actions enrichments'],
                            vertical_spacing=0.1, horizontal_spacing=0.01)

        for i, skill in enumerate(self.entries_std_per_player):
            if len(self.entries_std_per_player[skill]) > 0:
                df = self.entries_std_per_player[skill].reset_index()
                df.columns = ['dimension', 'std']
                if 'enrich' in skill or 'reduce' in skill:
                    curr_row = 2
                    action = 'enriched'
                else:
                    curr_row = 1
                    action = 'modified'

                fig.add_trace(go.Bar(x=df['dimension'], y=df['std'], name=skill), row=curr_row, col=1)

                # Set x-axis title
                fig.update_xaxes(title_text="Representation dimensions", row=curr_row, col=1)
                # Set y-axes titles
                fig.update_yaxes(title_text="<b>STD</b>", row=curr_row, col=1)

        fig.update_layout(title_text=f'Dimensions STD VS. skill changed')
        fig.show()

    def analyze_action_context(self, next_action_distribution=False):
        # Understand when an action happens
        actions = self.events_data.loc[self.events_data[COLUMNS.ACTION_TYPE] == self._skill_name, 'index'].copy()
        last_actions = []
        next_actions = []
        for action_ in actions:
            next_action = self.events_data.at[action_ + 1, COLUMNS.ACTION_TYPE]
            next_actions.append(next_action)
            last_action = self.events_data.at[action_ - 1, COLUMNS.ACTION_TYPE]
            last_actions.append(last_action)

        if next_action_distribution:
            next_actions_distribution = pd.Series(next_actions).value_counts()
            next_actions_distribution = next_actions_distribution / next_actions_distribution.sum()
            if self.verbose:
                print(f'\nAction distribution after {self._skill_name}s (ONE ACTION):')
                print(next_actions_distribution)
                print(f'\nAction distribution after {self._skill_name}s (ONE ACTION, NORMED):')
                print(next_actions_distribution)
            return next_action_distribution

        last_action_distribution = pd.Series(last_actions).value_counts()
        last_action_distribution = last_action_distribution / last_action_distribution.sum()
        if self.verbose:
            print(f'\nAction distribution before {self._skill_name}s (ONE ACTION):')
            print(last_action_distribution)

        return last_action_distribution

    def infer_in_vocab_player(self, player_matches):
        player_new_vectors = []
        for i, match_ in enumerate(player_matches):
            if isinstance(match_, list):
                match_doc = match_[:]
            elif isinstance(match_, int):
                match_doc = self.players_corpus.corpus[match_]
            else:
                raise NotImplementedError(f"Illegal type value for match_doc - {type(match_)}")
            # Modify all matches of this player
            inferred_vector = estimate_doc_vector(self.player_match2vec, match_doc, steps=self.steps,
                                                  repetitions=self.k, norm=False)
            player_new_vectors.append(inferred_vector)

        player_vec = np.vstack(player_new_vectors).mean(axis=0)
        return player_vec

    def _get_players_in_position(self):
        # Find all players matching this position
        players_in_position = self.events_data.loc[self.events_data[COLUMNS.POSITION] == self._position_name,
                                                   [COLUMNS.PLAYER_NAME, COLUMNS.MATCH_ID]].drop_duplicates()
        players_in_position = players_in_position.apply(
            lambda x: x[COLUMNS.PLAYER_NAME] + self.players_corpus.separator + str(x[COLUMNS.MATCH_ID])
                      + '.json' if '.json' not in str(x[COLUMNS.MATCH_ID]) else '', axis=1)
        players_in_position = players_in_position.to_list()

        # Calc avg num docs for players in position.
        matches_per_player = self.events_data[[COLUMNS.PLAYER_NAME, COLUMNS.MATCH_ID]].groupby(
            by=[COLUMNS.PLAYER_NAME]).count()
        num_docs_to_use = matches_per_player[COLUMNS.MATCH_ID].median()
        return players_in_position, num_docs_to_use

    def calc_skill_stats(self, _player_docs: [list], event_pattern, success_pattern) -> (np.array, pd.Series):
        '''
        Calculate frequency ang average success rate of the skill
        :param event_pattern: regex pattern to match for counting the event as relevant
        :param success_pattern: regex pattern to identify if the event was successful
        '''
        mean_action_counts = []
        mean_success_rate = []
        action_count = 0
        success_count = 0
        for doc in _player_docs:
            for token_ix in doc:
                token_ = self.players_corpus.ix_2_token[token_ix]
                if _search(event_pattern, token_):
                    action_count += 1
                    if _search(success_pattern, token_):
                        success_count += 1
            mean_action_counts.append(action_count)
            mean_success_rate.append(0 if action_count == 0 else success_count / action_count)
        return np.mean(mean_action_counts), np.mean(mean_success_rate)

    def create_player_variation(self, player_name: str, interventions: list, limit_interventions=None,
                                player_matches=None, inference_bias_vector=None) -> (np.array, pd.Series):
        '''
        Create a new player after executing a list of interventions in his matches.
        :param player_name: name of the player to improve, str
        :param interventions: list of interventions to apply on the player matches
        :param limit_interventions: if None - no limitation, it prevents from making more than limit_interventions changes
        :param inference_bias_vector: vector of inference bias to add if add_baseline_bias is true
        - Globalparam normalize_vector: if True, inferred vector is normalized after inference [and adding bias]
        :return: numpy array of the new player variation, and the k most similar players to it.
        '''
        # Get all player's matches
        if player_matches is None:
            player_matches = self.player_to_matches_docs[player_name]
        _variation_docs = []
        player_new_vectors = []
        interventions_counts = []
        oov_counts = []
        for i, match_ in enumerate(player_matches):
            if isinstance(match_, str):
                match_doc = self.players_corpus.corpus[int(self.docs_to_ix[match_])]
            elif isinstance(match_, list):
                match_doc = match_[:]
            elif isinstance(match_, int):
                match_doc = self.players_corpus.corpus[match_]
            else:
                raise NotImplementedError(f"Illegal type value for match_doc - {type(match_)}")
            # Modify all matches of this player
            player_variation_doc, interventions_count, oov_count = modify_doc(match_doc, interventions,
                                                                              self.players_corpus,
                                                                              limit_interventions=limit_interventions)
            interventions_counts.append(interventions_count)
            oov_counts.append(oov_count)
            _variation_docs.append(player_variation_doc[:])

            # Learn the variation's representation
            player_new_vector = estimate_doc_vector(self.player_match2vec, player_variation_doc, steps=self.steps,
                                                    repetitions=self.k,
                                                    norm=False)

            player_new_vectors.append(player_new_vector)

        player_variation_vec = np.vstack(player_new_vectors).mean(axis=0)

        if self.add_baseline_bias and inference_bias_vector is not None:
            player_variation_vec = player_variation_vec + inference_bias_vector

        distances = pd.DataFrame(self.players_vectors.index.copy(), columns=['player_name'])
        distances['phrase_similarity'] = distances['player_name'].apply( \
            lambda RHS: np.dot(player_variation_vec, self.players_vectors.loc[RHS]) / (
                    np.linalg.norm(player_variation_vec) * np.linalg.norm(self.players_vectors.loc[RHS])))

        if self.verbose:
            print(
                f"{player_name} variation completed with {len(player_matches)} matches "
                f"and {np.mean(interventions_counts)} interventions on avg"
                f"(OOV avg: {np.mean(np.mean(oov_counts))})")
        return player_variation_vec, distances.sort_values(by=['phrase_similarity'], ascending=False), _variation_docs

    def create_player_enriched_variation(self, player_name: str, _interventions: list, player_matches=None,
                                         inference_bias_vector=None):
        '''
        Creates a player variation by enriching his documents - applying
        :param player_name: str
        :param _interventions: list of regex patterns to match. if match - apply _enricher function on it
        :type _interventions: list of dicts
                                - 'token_builder': function that adds the relevant token according to the last success intervention
        :param player_matches: list of lists. Inner lists are the player documents (e.g, for baseline players that
                                their games are not in the player_to_matches object.
                                Each inner list is a list of tokens indexes (not strs).
                                :param inference_bias_vector: vector of inference bias to add if add_baseline_bias is true
        - Globalparam normalize_vector: if True, inferred vector is normalized after inference [and adding bias]
        :return: new eneriched doc (list), most similar docs (df) and player_new_docs
        '''
        # Get all player's matches
        if player_matches is None:
            player_matches = self.player_to_matches_docs[player_name]

        player_new_vectors = []
        player_new_docs = []
        num_interventions = []
        oov_counts = []
        for i, match_ in enumerate(player_matches):
            if isinstance(match_, str):
                match_doc = self.players_corpus.corpus[int(self.docs_to_ix[match_])]
            elif isinstance(match_, list):
                match_doc = match_[:]
            elif isinstance(match_, int):
                match_doc = self.players_corpus.corpus[match_]
            else:
                raise NotImplementedError(f"Illegal type value for match_doc - {type(match_)}")

            player_new_docs.append(match_doc[:])
            # Modify all matches of this player
            new_match_doc, n_interventions, oov_count = enrich_doc(match_doc, _interventions, self.players_corpus,
                                                                   limit_interventions=None)
            oov_counts.append(oov_count)
            num_interventions.append(n_interventions)
            new_match_enriched_doc = estimate_doc_vector(self.player_match2vec, new_match_doc, steps=self.steps,
                                                         repetitions=self.k,
                                                         norm=False)
            player_new_vectors.append(new_match_enriched_doc)

        player_new_vec = np.vstack(player_new_vectors).mean(axis=0)

        if self.add_baseline_bias and inference_bias_vector is not None:
            player_new_vec = player_new_vec + inference_bias_vector

        distances = pd.DataFrame(self.players_vectors.index.copy(), columns=['player_name'])
        distances['phrase_similarity'] = distances['player_name'].apply( \
            lambda RHS: np.dot(player_new_vec, self.players_vectors.loc[RHS]) / (
                    np.linalg.norm(player_new_vec) * np.linalg.norm(self.players_vectors.loc[RHS])))

        if self.verbose:
            print(
                f"{player_name} variation completed with {len(player_matches)} matches "
                f"and {np.mean(np.mean(num_interventions))} interventions on avg "
                f"(OOV avg: {np.mean(np.mean(oov_counts))})")
        return player_new_vec, distances.sort_values(by=['phrase_similarity'],
                                                     ascending=False).head(self.k), player_new_docs

    def get_player_inference_bias_vector(self, _player, player_inferred_vector, players_vectors):
        if _player in players_vectors:
            inference_bias_vector = player_inferred_vector.copy() - players_vectors.loc[_player]
        else:
            inference_bias_vector = np.zeros_like(player_inferred_vector)
        return inference_bias_vector

    def build_baseline_vectors(self):
        # Create a baseline player for each position, so we can give attributes to the player by comparing them to the
        #   appropriate baseline player
        self.baselines_vectors = {}
        self.baselines_docs_ix = {}
        for position_ in self.positions:
            self._position_name = position_
            players_in_curr_position, avg_matches_count = self._get_players_in_position()
            self.baselines_vectors[position_], __, self.baselines_docs_ix[position_] = create_baseline_player(
                self.player_match2vec,
                players_in_curr_position,
                avg_matches_count,
                self.players_corpus)

    def _build_dribble(self, _last_action):
        location = _last_action.split('<')[0]
        x = location[1]

        if int(x) < 3:
            # Fix location to midfield/ attacking position - randomly chose between 4 and 5
            x = str(int(np.random.choice([4, 5])))
            location = f"({x}{location[2:]}"
        return location + "<dribble>:|outcome=complete"

    def _build_shot(self, _last_action, goal=False, foot='right', inbox_shot=False):
        outcome = np.random.choice(['off_t', 'saved', 'goal'], p=[0.45, 0.45, 0.1]) if not goal else 'goal'
        technique = 'normal'
        shot_type = 'open_play'
        body_part = f'{foot}_foot'
        if '<pass>' not in _last_action:
            # Create shot from same location
            location = _last_action.split('<')[0]
        else:
            # Cannot pass to himself, randomize location
            x = str(int(np.random.choice([4, 5]))) if not inbox_shot else '5'
            y = np.random.choice([2, 3, 4]) if not inbox_shot else '3'
            location = f"({x}/5,{y}/5)"

        return location + f"<shot>:|{technique}|outcome={outcome}|body_pa={body_part}|type_na={shot_type}"

    def _build_interception(self, next_action):
        outcome = np.random.choice(['success_in_play', 'won'])
        location = next_action.split('<')[0]
        return location + f"<interception>:|outcome={outcome}"

    def _skill_analysis(self):
        '''
        Perform skill analysis on arguments saved in object self
        '''
        # Create few improved versions of _player with different p and show it on the Player2Vec plot
        players_to_highlight = ['baseline', self._player]
        _variations_similarities = []
        _variations_docs = {}

        # Add the same player, but send to inference to detect bias on inference
        _player_inferred_vec, most_similar, _player_inferred_docs = self.create_player_variation( \
            self._player, [], player_matches=self._player_matches)

        inference_bias_vector = self.get_player_inference_bias_vector(self._player, _player_inferred_vec,
                                                                      self.players_vectors)
        variation_name = f'_{self._player}_inference'
        _columns = [variation_name, f"{variation_name}:similarity"]
        _variations_docs[variation_name] = _player_inferred_docs
        _variations_similarities.append(most_similar)
        players_to_highlight.append(variation_name)
        if variation_name not in self.players_vectors.index:
            self.players_vectors = self.players_vectors.append(pd.Series(_player_inferred_vec, name=variation_name))

        if self._variation_action == 'modify':
            p_values = np.arange(0.1, 1.1, 0.1)
        else:
            p_values = np.concatenate((np.arange(0.02, 0.1, 0.02), np.arange(0.1, 0.6, 0.1)))

        for p in p_values:
            # Transform each action with prob p
            for _skill_improve in self._skill_improvements:
                _skill_improve['probability'] = p
            if self.verbose:
                print(f'Creating {self._player} {self._variation_action} variation for skill'
                      f' {self._skill_name} with p={p}')
            players_vectors = self.players_vectors.drop_duplicates()
            variation_vec, most_similar, variation_docs = self.actions_mapper[self._variation_action]( \
                self._player, self._skill_improvements,
                inference_bias_vector=inference_bias_vector,
                player_matches=self._player_matches)

            variation_name = f"{SHORTNAMES.get(self._player, self._player)}_{self._variation_action}" \
                             f"_{self._sign}{self._skill_name}_{np.round(p, 2)}"
            _variations_docs[variation_name] = variation_docs[:]
            # Copy player metadata to his variation
            if self._player in self.players_metadata:
                self.players_metadata[variation_name] = self.players_metadata[self._player].copy()
            else:
                # Variation of a baseline player / artificial player / player without metadata -> create generic
                self.players_metadata[variation_name] = {'Unnamed: 0': 0, 'player_id': 0000,
                                                         'player_nickname': variation_name, 'jersey_number': 0,
                                                         'country_id': 0, 'country_name': 'Artificial',
                                                         'player_name_lower': variation_name,
                                                         'position_name': self._position_name,
                                                         'team_name': 'Artificial'}

            # Analyze most similar players
            _variations_similarities.append(most_similar)
            if self.verbose:
                print(
                    f'Most similar players to {self._player} skill = {self._variation_action} {self._skill_name} '
                    f'and p={p} - WITH VARIATIONS:')
                print(most_similar.head(self.k))
                print(
                    f'Most similar players to {self._player} skill = {self._variation_action} {self._skill_name} '
                    f'and p={p}: - WITHOUT VARIATIONS')
                print(most_similar[most_similar[COLUMNS.PLAYER_NAME].isin(self.orig_corpus_players)].head(self.k))
            # Add the player variation to future plots
            players_to_highlight.append(variation_name)
            if variation_name not in players_vectors.index:
                self.players_vectors = players_vectors.append(pd.Series(variation_vec, name=variation_name))

        # Show similarities to each version so we will see the original player ger further and others closer
        if self.verbose:
            print(f'\nSkill analysis for {self._variation_action} {self._player} at skill {self._skill_name}')
        _variations_similarities = pd.concat(_variations_similarities, axis=1)
        for _p in p_values:
            _columns.extend([f'{self._variation_action} p={np.round(_p, 2)}:{COLUMNS.PLAYER_NAME}',
                             f'{self._variation_action} p={np.round(_p, 2)}:similarity'])
        _variations_similarities.columns = _columns
        _variations_similarities.to_csv(os.path.join(PATHS.EXPLAINERS,
                                                     f"most_similar_{self._player}_{self._variation_action}_"
                                                     f"skill_{self._skill_name}.csv"))

        # Plot embeddings for all new vectors + baseline player
        if f'baseline_{self._position_name}' not in self.players_vectors.index:
            self.players_vectors = self.players_vectors.append(
                pd.Series(self._baseline_vector, name=f'baseline_{self._position_name}'))

        if self._plot:
            self.players_vectors.index = [SHORTNAMES.get(ix, ix) for ix in self.players_vectors.index]
            plot_embeddings(self.players_vectors, players_to_highlight, docs_features=True,
                            docs_data=self.players_metadata,
                            doc_name_separator=self.players_corpus.separator, model_name=self.model_name,
                            highlight_selected=False,
                            title=f'UMAP projection of the skill {self._skill_name} for player {self._player}',
                            dimensions_reducer=self.dimensions_reducer,
                            show=self._plot, save_fig=self.save_artifacts, plotly_export=self.plotly_export)
        return _variations_similarities, _variations_docs


class LinearDocExplainer(ModelExplainer):
    def __init__(self, player_match2vec: Doc2Vec, players_corpus: Corpus, players_vectors: pd.DataFrame,
                 events_data: pd.DataFrame, **kwargs):
        '''
        :param player_match2vec: Doc2Vec model of player-to-match
        :param players_corpus: Corpus object of the player_match2vec model
        :param players_vectors: pd.DataFrame where the index is players name and values are their Player2Vec representation
        :param events_data: pd.DataFrame of events data, in StatsBomb data format
        :param kwargs:
            - repetitions: number of times to infer docs and average the results, for mitigating randomness
        '''
        # Data arguments
        self.min_matches_per_player = kwargs.get('', 10)
        self.verbose = kwargs.get('verbose', True)
        self.players_corpus = players_corpus
        self.player_match2vec = player_match2vec
        self.players_vectors = players_vectors
        self.events_data = events_data

        # Analysis arguments
        self.steps = kwargs.get('steps', 10)
        self.k = kwargs.get('k', 10)
        self.repetitions = kwargs.get('repetitions', 10)

        # supported_skills Dict:
        #   key=skill: str, value=function that return True/False if given token is relevant to skill
        self._supported_skills = {'scoring': is_normal_goal_token,
                                  'in_box_scoring': lambda token: re.match(in_box_scoring_pattern, token),
                                  'out_box_scoring': lambda token: re.match(out_box_scoring_pattern, token),
                                  'dribbling': lambda token: re.match(dribble_pattern, token),
                                  'flank_dribbling': lambda token: re.match(flank_dribble_pattern, token),
                                  'pressure': lambda token: re.match(forward_pressure_pattern, token),
                                  'carry': lambda token: '<carry>' in token,
                                  'offsides': lambda token: '<offside>' in token
                                  }
        self.skills_docs = {}
        self.skills_vectors = {}
        self.build_skills_documents()

    def get_supported_skills(self):
        return list(self._supported_skills.keys())

    def build_skills_documents(self):
        for skill, func in self._supported_skills.items():
            # Building artificial docs #
            self.skills_docs[skill] = self.build_artificial_doc(func)
            self.skills_vectors[skill] = estimate_doc_vector(self.player_match2vec, self.skills_docs[skill],
                                                             steps=self.steps, repetitions=self.repetitions)

    def build_artificial_doc(self, condition) -> list:
        '''
        Creates artificial Doc2Vec document to serve as an explainer
        :param condition: lambda function to apply on each word in the vocabulary
        :type condition: function
        '''
        relevant_tokens = [token for token in self.players_corpus.vocabulary if condition(token)]
        return relevant_tokens

    def default_run(self):
        run_arguments = [{COLUMNS.PLAYER_NAME: PLAYERS.INIESTA, 'skills': ['scoring', 'in_box_scoring',
                                                                           'outbox_scoring', '-dribbling',
                                                                           ['outbox_scoring', '-dribbling']]},
                         {COLUMNS.PLAYER_NAME: PLAYERS.NEYMAR, 'skills': ['-dribbling',
                                                                          '-flank_dribbling',
                                                                          ['-carry', '-dribbling']]},
                         {COLUMNS.PLAYER_NAME: PLAYERS.BUSQUETS,
                          'skills': ['dribbling', 'scoring', '-pressure', 'in_box_scoring', 'out_box_scoring']},

                         {COLUMNS.PLAYER_NAME: PLAYERS.LUIS_SUAREZ,
                          'skills': ['-scoring', '-inbox_scoring', '-pressure']},

                         {COLUMNS.PLAYER_NAME: PLAYERS.GRIEZMANN,
                          'skills': ['dribbling', 'flank_dribbling', '-pressure']}
                         ]
        return self.players_actions_analogies(run_arguments)

    def players_actions_analogies(self, run_arguments: [dict]) -> (dict, dict):
        '''

        :param run_arguments: {COLUMNS.PLAYER_NAME: PLAYERS.GRIEZMANN, 'skills': [list of strings or nested lists]
        :type run_arguments:
        :return: (1, 2):
            1. distances dict: {f"{player}_{skill}": pd.DataFrame of k most similar players, ...}
            2. dict: {f"{player}_{str(skill)}": linear combination of player and skills vectors [np.array]
        :rtype: (dict, dict)
        '''
        players_vectors = {}
        players_w_skills = {}

        def _iterate_skill(skill):
            neg_factor = 1
            if skill.startswith('-'):
                neg_factor = -1
                skill = skill[1:]
            return self.skills_vectors[skill] * neg_factor

        for _run in run_arguments:
            player = _run[COLUMNS.PLAYER_NAME]
            # Get player vector
            player_vector = np.array(self.players_vectors.loc[player])
            players_vectors[player] = {}
            player_vector_norm = player_vector / np.linalg.norm(player_vector)
            players_vectors[player]['player_vector'] = player_vector
            players_vectors[player]['player_vector_norm'] = player_vector_norm

            for _skill in _run['skills']:
                players_w_skills[f"{player}_{_skill}"] = players_vectors[player]['player_vector_norm']
                if isinstance(_skill, list):
                    for skill_i in _skill:
                        players_w_skills[f"{player}_{str(_skill)}"] += _iterate_skill(skill_i)
                else:
                    players_w_skills[f"{player}_{_skill}"] += _iterate_skill(_skill)

        # Count number of matches per player in order to keep player with at least min_matches_per_player
        matches_per_player = self.events_data[['player_name', 'match_id']].drop_duplicates().groupby(
            by=['player_name']).count() \
            .to_dict(orient='index')

        distances = {}
        for item_name, item in players_w_skills.items():
            distances[item_name] = {}
            for i, (player_i, vec_i) in enumerate(self.players_vectors.iterrows()):
                if matches_per_player[player_i][COLUMNS.MATCH_ID] >= self.min_matches_per_player:
                    distances[item_name][player_i] = np.dot(vec_i, item) / (
                            np.linalg.norm(vec_i) * np.linalg.norm(item))

            similarities = pd.DataFrame.from_dict(distances[item_name], orient='index')
            similarities.columns = ['cosine similarity']
            if self.verbose:
                print(f'Phrase = {item_name}. K most similar by cosine:')
                print(similarities.sort_values(by=['cosine similarity'], ascending=False).head(self.k))
                print()

        return distances, players_w_skills


def Player2Vec_std_analysis(players_metadata: dict, players_to_highlight: list = None, plotly_export=False,
                            save_artifacts=False, _plot=False):
    '''
    STD for Player2Vec figure to capture the span in space
    :param players_metadata: dict, metadata of all players (documents)
    :param players_to_highlight: list of players to highlight in plots, adding label and coloring
    :param  plotly_export: Whether to export plotly figures to plotly studio (see https://chart-studio.plotly.com)
    :param save_artifacts: whether to save the figure to artifacts directory or not
    '''
    model_name = "Player2Vec"
    corpus_obj_path = os.path.join(MODELS_ARTIFACTS, f'{model_name}_corpus.pickle')
    with open(corpus_obj_path, 'rb') as f:
        players_corpus = pickle.load(f)

    player_2_vec_model = Doc2Vec.load(os.path.join(MODELS_ARTIFACTS, f'{model_name}.model'))

    embedded_vocab = {}
    for i, doc_ in enumerate(players_corpus.documents_names):
        embedded_vocab[doc_] = player_2_vec_model.infer_vector(players_corpus.corpus[i])

    embedded_vocab = pd.DataFrame.from_dict(embedded_vocab, orient='index')
    embedded_vocab = embedded_vocab.reset_index()
    embedded_vocab[COLUMNS.PLAYER_NAME] = embedded_vocab['index'].apply(
        lambda ix: ix.split(players_corpus.separator)[0])
    players_embeddings = embedded_vocab.groupby(COLUMNS.PLAYER_NAME).std()

    # Remove NAs (not enough matches for calculation)
    players_embeddings = players_embeddings.dropna()

    model_name += ' Variance'
    plot_embeddings(players_embeddings, players_to_highlight, docs_features=True, docs_data=players_metadata,
                    doc_name_separator=players_corpus.separator, model_name=model_name,
                    color_attr=COLUMNS.POSITION, plotly_export=plotly_export, show=_plot,
                    title=f'UMAP projection of the {model_name}', save_fig=save_artifacts)


def analyze_vector_dimensions_semantics(embeddings_df: pd.DataFrame, k=10) -> pd.DataFrame:
    '''
    Return top k docs/tokens with highest value of each index of the representations
    :param embeddings_df: index = token/doc, columns = vector representation dimensions
    :param k: number of top (samples with highest values) and bottom (samples with lowest values) results to return
    :return: top matching tokens/docs concatenated with the worst matching tokens/doct
    '''
    top_results = {}
    bottom_results = {}

    for col in embeddings_df.columns:
        curr_df = embeddings_df.sort_values(by=col, ascending=False)
        top_results[col] = list(curr_df.head(k).index)[:]
        bottom_results[col] = list(curr_df.tail(k).index)[:]

    top_results = pd.DataFrame.from_dict(top_results, orient='index')
    top_results.columns = [f"Top_{i}" for i in range(top_results.shape[1])]
    bottom_results = pd.DataFrame.from_dict(bottom_results, orient='index')
    bottom_results.columns = [f"Bottom_{-i}" for i in range(top_results.shape[1], 0, -1)]
    return pd.merge(top_results, bottom_results, left_index=True, right_index=True)
