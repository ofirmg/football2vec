"""
Created on September 5 2021

This module build all language models: Action2Vec, Player2Vec, as well as additional plot & utils functions.

@author: Ofir Magdaci (@Magdaci)

"""

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
import os
import pandas as pd
import numpy as np
import random
import pickle
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import chart_studio.plotly as py
import chart_studio
from tqdm import tqdm

from lib.params import COLUMNS, CONSTANTS, PLOTLY_USERNAME, PLOTLY_API_KEY, MODELS_ARTIFACTS
from lib.data_processing import Corpus, get_enriched_players_metadata, get_enriched_events_data

import sys
from lib import data_processing
sys.modules['data_processing'] = data_processing

def output_document_length_distribution(model_name: str, corpus: [list]):
    document_length = pd.Series([len(doc) for doc in corpus])
    print(f"{model_name} document_length distribution:\n{document_length.describe()}")
    document_length.hist()
    plt.title(f"{model_name} document_length distribution")
    plt.show()


def export_doc_similarities(model_name: str, docs_embeddings: pd.DataFrame):
    '''
    Exports matrix of similarities between all docs, using both cosine similarity and euclidean distance
    :param model_name: string, name of the model
    :param docs_embeddings: DataFrame of document embeddings. index = docs names, values = vectors
    :return:
    '''
    docs_similarities_path = os.path.join(MODELS_ARTIFACTS, f"{model_name}_docs_similarities.pickle")
    print(f'\nCalculating {model_name} docs_similarities')
    docs_similarities = {}
    for i, (ix, doc_1) in tqdm(enumerate(docs_embeddings.iterrows()), total=docs_embeddings.shape[0]):
        for j, doc_2 in docs_embeddings[i:].iterrows():
            cosine_similarity = np.dot(doc_1, doc_2) / (np.linalg.norm(doc_1) * np.linalg.norm(doc_2))
            euclidean_similarity = np.linalg.norm(doc_1 - doc_2)
            docs_similarities[(ix, j)] = {'cosine': cosine_similarity, 'euclidean': euclidean_similarity}
            # Both are symmetric
            docs_similarities[(j, ix)] = docs_similarities[(ix, j)]
    with open(docs_similarities_path, 'wb') as f:
        pickle.dump(docs_similarities, f, protocol=pickle.HIGHEST_PROTOCOL)


def train_Word2Vec(corpus_obj: Corpus, model_name: str, force=False, min_count=10, embedding_size=32, sampling_window=5,
                   workers=3, epochs=5, verbose=False, save_artifacts=True):
    '''
    Training a Word2Vec model over a given Corpus.
    :param corpus_obj: Corpus object to access its .corpus - list of documents.
    :param model_name: name of model (.e.g, Action2Vec)
    :param force: force to re-train (True) or try to load existent (False)
    :param min_count: minimal number of token appearances [=10]
    :param embedding_size: 'size' for Word2Vec object [=32]
    :param sampling_window: 'window' for Word2Vec object [=5]
    :param workers: for Word2Vec object [=3]
    :param epochs: number of epochs for training [=5]
    :param verbose: prints control
    :param save_artifacts: if True, save all artifacts, else skip saving
    :return: (Word2Vec model, Word2Vec wordvectors)
    '''
    if os.path.exists(os.path.join(MODELS_ARTIFACTS, f"{model_name}.model")) and not force:
        model = Word2Vec.load(os.path.join(MODELS_ARTIFACTS, f"{model_name}.model"))

        # Load back with memory-mapping = read-only, shared across processes.
        word_vectors = KeyedVectors.load(os.path.join(MODELS_ARTIFACTS, f"{model_name}.wordvectors"), mmap='r')
    else:
        # Train model
        corpus = corpus_obj.corpus

        if verbose:
            # Plot histogram of documents length
            output_document_length_distribution(model_name, corpus)
            print('Training model...')

        params = dict(min_count=min_count, size=embedding_size, window=sampling_window, workers=workers)
        model = Word2Vec(corpus, **params)
        model.train(corpus, total_examples=len(corpus), epochs=epochs, compute_loss=True)

        if verbose:
            print('DONE!\n')
        if save_artifacts: model.save(os.path.join(MODELS_ARTIFACTS, f"{model_name}.model"))

        # Store just the words + their trained embeddings.
        word_vectors = model.wv
        if save_artifacts: word_vectors.save(os.path.join(MODELS_ARTIFACTS, f"{model_name}.wordvectors"))

    return model, word_vectors


def train_Doc2Vec(corpus_obj: Corpus, model_name: str, force=False, min_count=10, embedding_size=32, sampling_window=5,
                  workers=3, epochs=10, verbose=False, save_artifacts=False) -> (Doc2Vec, KeyedVectors):
    '''
    Training a Doc2Vec model over a given Corpus.
    :param corpus_obj: Corpus object to access its .corpus - list of documents.
    :param model_name: name of model (.e.g, Player2Vec)
    :param force: force to re-train (True) or try to load existent (False)
    :param min_count: minimal number of token appearances [=10]
    :param embedding_size: 'size' for Doc2Vec object [=32]
    :param sampling_window: 'window' for Doc2Vec object [=5]
    :param workers: for Doc2Vec object [=3]
    :param epochs: number of epochs for training [=10]
    :param save_artifacts: if True, save all artifacts, else skip saving
    :param verbose: prints control
    :return: Doc2Vec model, docs_embeddings
    '''
    if os.path.exists(os.path.join(MODELS_ARTIFACTS, f"{model_name}.model")) and not force:
        model = Doc2Vec.load(os.path.join(MODELS_ARTIFACTS, f"{model_name}.model"))
    else:
        # Train model
        docs = corpus_obj.corpus
        _docs_for_model = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]

        if verbose:
            # Plot histogram of documents length
            output_document_length_distribution(model_name, docs)
            if verbose: print('Training model...')

        params = dict(min_count=min_count, size=embedding_size, window=sampling_window, workers=workers)
        model = Doc2Vec(documents=_docs_for_model, **params)
        model.train(_docs_for_model, total_examples=len(_docs_for_model), epochs=epochs)
        if verbose: print('DONE!\n')
        if save_artifacts: model.save(os.path.join(MODELS_ARTIFACTS, f"{model_name}.model"))

    docs_embeddings = model.docvecs
    if save_artifacts: docs_embeddings.save(os.path.join(MODELS_ARTIFACTS, f"{model_name}.wordvectors"))
    return model, docs_embeddings


def Action2Vec(events_data: pd.DataFrame, force_create=False, num_similar=0, model_args=None, **kwargs):
    # Action2Vec #
    save_artifacts = kwargs.get('save_artifacts', True)
    verbose = kwargs.get('verbose', False)

    model_name = 'Action2Vec'
    if not os.path.exists(MODELS_ARTIFACTS):
        if verbose: print(f'Create {MODELS_ARTIFACTS} directory')
        os.makedirs(MODELS_ARTIFACTS)

    if verbose: print(f'\n{model_name}')

    if model_args is None:
        model_args = dict(force=force_create, min_count=10, embedding_size=32, sampling_window=5, workers=3)
    else:
        default_model_args = dict(force=force_create, min_count=5, embedding_size=32, sampling_window=5, workers=3)
        model_args = default_model_args.update(model_args)

    # Build corpus
    corpus_obj_path = os.path.join(MODELS_ARTIFACTS, f"{model_name}_corpus.pickle")
    if os.path.exists(corpus_obj_path) and not force_create:
        if verbose: print(f'\nLoading existing corpus object & vocabulary data')
        with open(corpus_obj_path, 'rb') as f:
            corpus_obj = pickle.load(f)
    else:
        if verbose: print(f'\nBuilding actions corpus and applying tokenization')
        corpus_obj = Corpus(add_goal_events=False,
                            actions_to_ignore_outcome=['duel'],
                            add_shift_possessions_tokens=False, verbose=verbose)
        events_data = corpus_obj.build_corpus(events_data, sampling_window=model_args['sampling_window'])
        events_data['token_ix'] = events_data['token'].apply( \
            lambda tok_: corpus_obj.token_2_ix.get(tok_, corpus_obj.token_2_ix['oov']))

        if save_artifacts:
            with open(corpus_obj_path, 'wb') as f:
                pickle.dump(corpus_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    action_2_vec, action_2_vec_embeddings = train_Word2Vec(corpus_obj, model_name, verbose=verbose,
                                                           save_artifacts=save_artifacts, **model_args)
    model_vocabulary = list(action_2_vec_embeddings.vocab.keys())
    model_filtered_vocabulary_ix = [token_ for token_ in model_vocabulary if token_ in corpus_obj.vocabulary_ix]
    model_filtered_vocabulary = [corpus_obj.ix_2_token[token_] for token_ in model_filtered_vocabulary_ix]

    if num_similar > 0:
        selected_tokens = random.sample(model_filtered_vocabulary_ix, num_similar)
        for token_ix in selected_tokens:
            token = corpus_obj.ix_2_token[token_ix]
            if verbose: print(f"\nToken = {token}")
            similar = action_2_vec_embeddings.most_similar(token_ix, topn=num_similar)  # get other similar words
            if verbose: print(
                f" - {num_similar} most similar tokens: {[corpus_obj.ix_2_token[sim_[0]] for sim_ in similar]}")

            similar_neg = action_2_vec_embeddings.most_similar(negative=[token_ix])
            if verbose: print(" - {num_similar} most similar tokens to NEGATIVE form: "
                              f"{[corpus_obj.ix_2_token[sim_[0]] for sim_ in similar_neg]}")

    embedded_vocab = pd.DataFrame.from_dict({corpus_obj.ix_2_token[token_ix]: action_2_vec_embeddings[token_ix]
                                             for token_ix in model_filtered_vocabulary_ix}, orient='index')
    if kwargs.get('plot', False):
        print(f'\nPlotting {model_name} embeddings')
        plot_embeddings(embedded_vocab, model_filtered_vocabulary,
                        title=f'UMAP projection of the {model_name}',
                        action_features=True, model_name=model_name)
    return action_2_vec, events_data, corpus_obj, embedded_vocab


def Player2Vec(events_data: pd.DataFrame, players_metadata: dict, force_create=False,
               players_to_highlight: list = None, model_args: dict = None, **kwargs):
    '''
    Train Player2Vec model and plot it for PlayerMatch2Vec and Player2Vec using averaging of player matches
    :param events_data: dataframe of events data for corpus
    :param players_metadata: dict, metadata of all players (documents)
    :param force_create: bool, whether load all artifacts (=default) to build from scratch
    :param players_to_highlight: list of players to highlight in plots, adding label and coloring
    :param model_args: dict of args for Doc2Vec model
    :return: events data (with token column), players_corpus, players_embeddings
    '''
    model_name = kwargs.get('model_name', "Player2Vec")
    verbose = kwargs.get('verbose', False)
    plot = kwargs.get('plot', False)
    save_artifacts = kwargs.get('save_artifacts', True)
    if not os.path.exists(MODELS_ARTIFACTS):
        print(f'Create {MODELS_ARTIFACTS} directory')
        os.makedirs(MODELS_ARTIFACTS)

    if model_args is None:
        model_args = dict(force=force_create, min_count=1, embedding_size=32, sampling_window=2, workers=3)
    else:
        default_model_args = dict(force=force_create, min_count=1, embedding_size=32, sampling_window=2, workers=3,
                                  epochs=10)
        default_model_args.update(model_args)
        model_args = {key: val for key, val in default_model_args.items()}

    # Build corpus
    corpus_obj_path = os.path.join(MODELS_ARTIFACTS, f"{model_name}_corpus.pickle")
    if os.path.exists(corpus_obj_path) and not force_create:
        print(f'\nLoading existing corpus_obj for {model_name}')
        with open(corpus_obj_path, 'rb') as f:
            players_corpus = pickle.load(f)
    else:
        print(f'\nBuilding actions corpus and applying tokenization')
        print(f'Building {model_name} corpus')
        players_aggr_columns = [COLUMNS.PLAYER_NAME, COLUMNS.MATCH_ID]
        players_corpus = Corpus(aggr_columns=players_aggr_columns, separator='/',
                                add_goal_events=False,
                                actions_to_ignore_outcome=['duel'],
                                add_shift_possessions_tokens=False, verbose=verbose)
        players_corpus.build_corpus(events_data, sampling_window=5, allow_concat_documents_=False)
        if save_artifacts:
            with open(corpus_obj_path, 'wb') as f:
                pickle.dump(players_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"Number of documents for model: {len(players_corpus.corpus)}")
    player_2_vec_model, players_matches_embeddings = train_Doc2Vec(players_corpus, model_name, verbose=verbose,
                                                                   save_artifacts=save_artifacts, **model_args)
    if verbose:
        print(f"Number of documents: {len(players_corpus.documents_names)}")
        print(f"Number of sentences in corpus: {len(players_corpus.corpus)}")

    # Build a comfortable dict for collecting players' vectors, later to be the players_embeddings - pd.DataFrame
    embedded_vocab = {}
    for i, doc_ in enumerate(players_corpus.documents_names):
        embedded_vocab[doc_] = player_2_vec_model.infer_vector(players_corpus.corpus[i])

    match_model_name = model_name.split('2')[0] + 'Match' + model_name.split('2')[1]
    embedded_vocab = pd.DataFrame.from_dict(embedded_vocab, orient='index')
    if plot:
        plot_embeddings(embedded_vocab, [], docs_features=True, docs_data=players_metadata,
                        doc_name_separator=players_corpus.separator, model_name=match_model_name,
                        color_attr=COLUMNS.POSITION,
                        title=f'UMAP projection of {match_model_name}')

    # Build players embeddings - averages of PlayerMatch2Vec of his matches
    players_embeddings_path = os.path.join(MODELS_ARTIFACTS, f"{model_name}_embeddings.pickle")
    if os.path.exists(players_embeddings_path) and not force_create:
        print(f'\nLoading existing {model_name} embeddings')
        with open(players_embeddings_path, 'rb') as f:
            players_embeddings = pickle.load(f)
    else:
        print(f'\nBuilding {model_name} embeddings')
        embedded_vocab = embedded_vocab.reset_index()
        embedded_vocab[COLUMNS.PLAYER_NAME] = embedded_vocab['index'].apply(
            lambda ix: ix.split(players_corpus.separator)[0])
        players_embeddings = embedded_vocab.groupby(COLUMNS.PLAYER_NAME).mean()

        if save_artifacts:
            with open(players_embeddings_path, 'wb') as f:
                pickle.dump(players_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    if save_artifacts or force_create:
        export_doc_similarities(model_name, players_embeddings)

    if plot:
        plot_embeddings(players_embeddings, players_to_highlight, docs_features=True, docs_data=players_metadata,
                        doc_name_separator=players_corpus.separator, model_name=model_name,
                        color_attr=COLUMNS.POSITION,
                        title=f'UMAP projection of the {model_name}')

    return player_2_vec_model, events_data, players_corpus, players_embeddings, players_matches_embeddings


def _get_reduced_representation(dimensions_reducer, data: pd.DataFrame):
    '''
    :param dimensions_reducer: a Transformer like function has the fit_transform & transform methods
    :param data: pd.DataFrame to transform
    :return: (dimensions_reducer, transformed data)
    '''
    if dimensions_reducer is None:
        dimensions_reducer = umap.UMAP()
        data = dimensions_reducer.fit_transform(data)
        return dimensions_reducer, data
    else:
        return dimensions_reducer, dimensions_reducer.transform(data)


def plot_embeddings(embedded_vocab: pd.DataFrame, tokens_to_label: list, docs_data: dict = None,
                    doc_name_separator='|', model_name='', show=True, save_fig=True, highlight_selected=False,
                    dimensions_reducer=None,
                    **kwargs):
    """
    Plot UMAP embeddings using plotly.
    :param embedded_vocab: dataframe with vector representation for each word in vocab (keys = words)
    :param tokens_to_label: list of tokens to specifically write their name on the plot
    :param docs_data: required only if docs_features (by kwargs) is True
    :param doc_name_separator: will be used to separate docs names to its aggregate columns.
                                Usually '|' for Action2Vec, and '/' for Doc2Vec.
    :param model_name: model name used for exporting embeddings plot
    :param highlight_selected: if true, tokens_to_label will be colored, by name, while other players will be marked
                                as 'Not highlighted'. This argument override any additional color argument
    :param save_fig: bool, whether save the embedding plot
                    (true by default - will override current plot. Put false on Streamlit)
    :param dimensions_reducer: an object that has transform method, reducing the dimensions of the given input
    :param show: bool, whether to show the embedding plot (true by default).
                If false, will return the figure (for Streamlit)
    :param kwargs:
        - colors - list or similar iterator holds color divider values, e.g., by position_name.
        - action_features - if to apply action_features, like in Action2Vec
        - docs_features - if to apply action_features, like in Doc2Vec. Then, docs_data will be in use
        * IMPORTANT * - one of the values - docs_features or docs_features has to be true, or an error is raised
    """
    if tokens_to_label is None:
        tokens_to_label = []

    embedded_vocab_cp = embedded_vocab.copy()
    dimensions_reducer, embedding = _get_reduced_representation(dimensions_reducer, embedded_vocab_cp)

    for feature in [0, 1]:
        embedded_vocab_cp[f'umap_{feature}'] = embedding[:, feature]

    action_features = kwargs.get('action_features', False)
    docs_features = kwargs.get('docs_features', False)
    if action_features:
        embedded_vocab_cp['custom_data'] = [val.split(')')[0] for val in embedded_vocab_cp.copy()]
        embedded_vocab_cp['hover_name'] = embedded_vocab_cp.index.copy()
        default_colors = [val.split('<')[1].split('>')[0] for val in embedded_vocab_cp.index.copy()]
        embedded_vocab_cp['color'] = default_colors
    elif docs_features:
        if kwargs.get('colors', None) is not None:
            embedded_vocab_cp['color'] = kwargs['colors']
            embedded_vocab_cp['color'].fillna('other', inplace=True)
        else:
            color_attr = kwargs.get('color_attr', 'position_name')
            embedded_vocab_cp['color'] = [docs_data.get(idx.split(doc_name_separator)[0], {color_attr: ''})[color_attr]
                                          if idx.split(doc_name_separator)[0] in docs_data
                                          else idx.split(doc_name_separator)[0]
                                          for idx in embedded_vocab_cp.index.copy()]
        embedded_vocab_cp['label'] = embedded_vocab_cp.index.copy()
        embedded_vocab_cp['hover_name'] = embedded_vocab_cp.index.copy()
        embedded_vocab_cp['custom_data'] = [docs_data.get(idx.split(doc_name_separator)[0],
                                                          {'team_name': ''})['team_name']
                                            if idx.split(doc_name_separator)[0] in docs_data else
                                            idx.split(doc_name_separator)[0] for idx in embedded_vocab_cp.index.copy()]
    else:
        raise ValueError("plot embeddings needs action_features OR docs_features to be True for getting plot's data")

    # Custom selected tokens / docs
    if len(tokens_to_label) > 0 and highlight_selected:
        embedded_vocab_cp['color'] = [idx if idx in tokens_to_label else 'Not highlighted'
                                      for idx in embedded_vocab_cp.index.copy()]

    fig = px.scatter(embedded_vocab_cp, x='umap_0', y='umap_1', color='color',
                     hover_name='hover_name', hover_data=["custom_data", "label"],
                     title=kwargs.get('title', ''))
    for player_ in tokens_to_label:
        if player_ in embedded_vocab_cp.index:
            fig.add_annotation(x=embedded_vocab_cp.at[player_, 'umap_0'],
                               y=embedded_vocab_cp.at[player_, 'umap_1'],
                               text=player_)
    fig.update_traces(textposition='top center')

    if save_fig: fig.write_html(os.path.join(MODELS_ARTIFACTS, f"{model_name}_umap_plot.html"))

    if kwargs.get('plotly_export', False):
        chart_studio.tools.set_credentials_file(username=PLOTLY_USERNAME, api_key=PLOTLY_API_KEY)
        py.plot(fig, filename=kwargs.get('figure_name', f"plotly_{model_name}"), auto_open=True)
    if show:
        fig.show()
    else:
        return fig


def build_language_models(events_data=None, verbose=False, force_create=True, plotly_export=False,
                          save_artifacts=False) -> (Word2Vec, Doc2Vec, dict):
    '''
    The function builds all models of Football2Vec: Action2Vec and Player2Vec
    :param events_data: DataFrame of events data to use. If None (default) -> load DataFrame: get_enriched_events_data
    :param verbose: bool, prints control
    :param force_create: bool, whether to save artifacts or not
    :param plotly_export: bool, whether to export Plotly figures to Plotly studio (see https://chart-studio.plotly.com)
            For the export, fill PLOTLY_USERNAME and PLOTLY_API_KEY in params.py
    :param save_artifacts: bool, whether to save artifacts or not
    :return: action_2_vec, player_2_vec_model, models_outputs
    '''
    if verbose:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)

    if events_data is None:
        events_data = get_enriched_events_data(force_create=False, save_artifacts=save_artifacts, verbose=verbose)

    if verbose:
        print(f"\nevents_data size: {events_data.shape}")

    # Keep only events supported by the language
    events_data = events_data[events_data[COLUMNS.ACTION_TYPE].isin(CONSTANTS.VOCABULARY)]

    # Get enriched players metadata
    players_metadata = get_enriched_players_metadata(events_data, force_create=False)

    # Action2Vec #
    action_2_vec, actions_vocab_data, actions_corpus, actions_embeddings = Action2Vec(events_data,
                                                                                      force_create=force_create,
                                                                                      plotly_export=plotly_export,
                                                                                      save_artifacts=save_artifacts,
                                                                                      verbose=verbose)

    if verbose:
        print('Action2Vec Vocabulary size =', len(actions_corpus.vocabulary))

    # Doc2Vec - players
    model_args = dict(force=force_create, min_count=1, embedding_size=32, sampling_window=1, workers=3)
    player_2_vec_model, players_vocab_data, players_corpus, players_embeddings, players_matches_embeddings = \
        Player2Vec(events_data, players_metadata, force_create=force_create, force_similarities=force_create,
                   players_to_highlight=[], model_args=model_args, plotly_export=plotly_export,
                   save_artifacts=save_artifacts
                   )

    models_outputs = {'actions_vocab_data': actions_vocab_data,
                      'actions_corpus': actions_corpus,
                      'players_corpus': players_corpus,
                      'actions_embeddings': actions_embeddings,
                      'players_embeddings': players_embeddings,
                      'players_matches_embeddings': players_matches_embeddings
                      }

    return action_2_vec, player_2_vec_model, models_outputs
