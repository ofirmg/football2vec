"""
Created on September 5 2021

This module is an executer of explainers.py: loads all relevant artifacts and conveniently demonstrates code usage.

@author: Ofir Magdaci (@Magdaci)

"""


import os

from lib.app_parts import get_doc2vec_data
from lib.data_handler import get_teams_metadata
from lib.data_processing import get_enriched_players_metadata, get_enriched_events_data
from lib.explainers import PlayerSkillsExplainer, analyze_vector_dimensions_semantics, Player2Vec_std_analysis, \
    ActionAnalogies, PlayersAnalogies
from lib.models import Action2Vec
from lib.params import COLUMNS, CONSTANTS, PLAYERS

verbose = True
plot = True
export_artifacts = True
plotly_export = False
force_create = False

num_similar = 10
min_matches_per_player = 10
num_examples_per_analogy = 5

if __name__ == '__main__':
    os.chdir('../')
    enriched_events_data = get_enriched_events_data(verbose=verbose)
    vocab_data = enriched_events_data[enriched_events_data[COLUMNS.ACTION_TYPE].isin(CONSTANTS.VOCABULARY)]
    players_metadata = get_enriched_players_metadata(vocab_data, verbose=verbose)
    teams_metadata = get_teams_metadata(verbose=verbose)

    # Action2Vec #
    action2vec, events_data, actions_corpus, action2vec_embeddings = Action2Vec(vocab_data,
                                                                                force_embeddings=force_create)

    # Actions analogies analysis
    actions_explainer = ActionAnalogies(action2vec, actions_corpus, action2vec_embeddings, enriched_events_data,
                                        _plot=plot,
                                        verbose=verbose,
                                        k=num_similar,
                                        num_examples_per_analogy=num_examples_per_analogy,
                                        min_matches_per_player=min_matches_per_player)
    actions_explainer.default_run()

    ##### Player2Vec #####
    model_name = "Player2Vec"

    # player_2_vec_model, players_matches_embeddings, players_corpus, players_embeddings
    player_match2vec, players_matches_embeddings, players_corpus, players_embeddings = get_doc2vec_data(model_name)

    # Analyze_vector_entries_meaning
    analyze_vector_dimensions_semantics(action2vec_embeddings)

    # Add STD for Player2Vec figure to capture the span in space
    Player2Vec_std_analysis(players_metadata, _plot=plot, plotly_export=plotly_export, save_artifacts=export_artifacts)

    # Players analogies analysis
    players_analogies_explainer = PlayersAnalogies(player_match2vec, players_embeddings, players_corpus,
                                                   match_sampling=min_matches_per_player,
                                                   num_similar=num_similar,
                                                   export_artifacts=export_artifacts,
                                                   verbose=verbose)
    players_analogies_explainer.default_run()

    # Players & actions analogies
    players_skills_explainer = PlayerSkillsExplainer(player_match2vec, players_corpus, players_metadata,
                                                     players_embeddings,
                                                     enriched_events_data,
                                                     plotly_export=plotly_export,
                                                     verbose=verbose,
                                                     plot=plot,
                                                     combined_variations_plot=plot)
    phrases = [
        # GRIEZMANN
        {COLUMNS.PLAYER_NAME: PLAYERS.GRIEZMANN, COLUMNS.POSITION: 'Center Forward',
         'skill_name': ['enrich_interceptions']},

        # Dembele
        {COLUMNS.PLAYER_NAME: PLAYERS.DEMBELE, COLUMNS.POSITION: 'Left Wing',
         'skill_name': ['-right_foot_use', 'dribble', 'shot']},

        # De Jong
        {COLUMNS.PLAYER_NAME: PLAYERS.DE_JONG, COLUMNS.POSITION: 'Center Midfield',
         'skill_name': ['high_passes', 'pass_to_back', 'enrich_shots']},

        # INIESTA
        {COLUMNS.PLAYER_NAME: PLAYERS.INIESTA, COLUMNS.POSITION: 'Center Midfield',
         'skill_name': ['reduce_dribble']},

        # Baseline Center Forward player
        {COLUMNS.PLAYER_NAME: f'Center Forward_baseline', COLUMNS.POSITION: 'Center Forward',
         'skill_name': ['enrich_dribble']},

    ]
    players_skills_explainer.players_skills_analysis(phrases)
