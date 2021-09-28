"""
Created on September 5 2021

This core module is responsible for loading the raw StatsBomb data and building metadata objects.
It is the only module that in touch with the raw StatsBomb dataset files, making them more easy to use for other modules.

@author: Ofir Magdaci (@Magdaci)

"""

from tqdm import tqdm as tqdm
from lib.params import *
import json
import pandas as pd


def load_all_events_data(dataset_path=PATHS.STATSBOMB_DATA, sub_dir='events', verbose=False):
    data = []
    if verbose:
        print('\nLoading all events data')
    dir_ = os.path.join(dataset_path, sub_dir, '')
    files_ = os.listdir(dir_)
    for match_ in tqdm(files_, total=len(files_)):
        with open(f'{dir_}{match_}') as data_file:
            data_ = json.load(data_file)
            data.append(pd.json_normalize(data_, sep="_").assign(match_id=match_))

    if verbose:
        print(' - COMPLETED\n')
    all_events_data = pd.concat(data)
    return all_events_data


def load_players_metadata(dataset_path=PATHS.STATSBOMB_DATA, sub_dir='lineups', force_create=False):
    data = []
    if os.path.exists(PATHS.PLAYERS_METADATA_PATH) and not force_create:
        print('\nLoading existing all_lineups_data.csv from artifacts directory...')
        return pd.read_csv(PATHS.PLAYERS_METADATA_PATH)

    else:
        print('\nData load STARTED')
        dir_ = f'{dataset_path}/{sub_dir}/'
        files_ = os.listdir(dir_)
        for file_name in tqdm(files_, total=len(files_)):
            with open(f'{dir_}{file_name}') as data_file:
                data_item = json.load(data_file)
                home_line_up, away_line_up = data_item[0], data_item[1]
                for player_ in home_line_up['lineup']:
                    data.append(pd.json_normalize(player_, sep="_"))
                for player_ in away_line_up['lineup']:
                    data.append(pd.json_normalize(player_, sep="_"))

        print('Data load COMPLETED\n')
        all_players_metadata = pd.concat(data)
        all_players_metadata.to_csv(PATHS.PLAYERS_METADATA_PATH)
        return all_players_metadata


def get_teams_metadata(dataset_path=PATHS.STATSBOMB_DATA, sub_dir='matches', force_create=False, path_prefix='',
                       save_artifacts=False, verbose=False, **kwargs):
    '''
    team_name, nation, list of competitions participated
    '''
    data = []
    tm_path = path_prefix + PATHS.TEAMS_METADATA_PATH
    if os.path.exists(tm_path) and not force_create:
        if verbose:
            print('\nLoading existing teams_metadata.csv from artifacts directory...')
        return pd.read_csv(tm_path)

    else:
        if verbose:
            print('\nData load STARTED')
        dir_ = f'{path_prefix}{dataset_path}/{sub_dir}/'
        competitions_dirs = [name_ for name_ in os.listdir(dir_) if name_.isnumeric()]
        for competitions_dir_ in tqdm(competitions_dirs, total=len(competitions_dirs)):
            files_ = os.listdir(os.path.join(dir_, competitions_dir_))
            for file_name in files_:
                with open(f'{dir_}/{competitions_dir_}/{file_name}') as data_file:
                    data_item = json.load(data_file)
                    for item_ in data_item:
                        data.append(pd.json_normalize(item_, sep="_"))

        if verbose:
            print('Data load COMPLETED\n')
        all_teams_metadata = pd.concat(data)
        # Take relevant column for both sides: home and away
        cols = [COLUMNS.MATCH_ID, 'season_season_name', 'stadium_name',
                'competition_competition_name', 'competition_country_name', 'competition_stage_name']
        # Take home team data
        home_teams_metadata = all_teams_metadata[cols + ['home_team_home_team_name', 'home_team_country_name',
                                                         'home_team_home_team_gender']]
        # Take away team data
        away_teams_metadata = all_teams_metadata[cols + ['away_team_away_team_name', 'away_team_country_name',
                                                         'away_team_away_team_gender']]
        # Shared mapping
        cols_mapper = {'season_season_name': 'season_name', 'competition_competition_name': 'competition_name'}
        home_teams_metadata.rename(columns=cols_mapper, inplace=True)
        away_teams_metadata.rename(columns=cols_mapper, inplace=True)
        # Separate mapping
        home_teams_metadata.rename(columns={'home_team_home_team_name': COLUMNS.TEAM_NAME,
                                            'home_team_home_team_gender': COLUMNS.TEAM_GENDER,
                                            'home_team_managers': COLUMNS.TEAM_MANAGERS,
                                            'home_team_country_name': COLUMNS.COUNTRY_NAME}, inplace=True)
        away_teams_metadata.rename(columns={'away_team_away_team_name': COLUMNS.TEAM_NAME,
                                            'away_team_away_team_gender': COLUMNS.TEAM_GENDER,
                                            'away_team_managers': COLUMNS.TEAM_MANAGERS,
                                            'away_team_country_name': COLUMNS.COUNTRY_NAME}, inplace=True)
        # Concat vertically home and away metadata together (each row is a match metadata of one team)
        all_teams_metadata = pd.concat([home_teams_metadata, away_teams_metadata], axis=0)
        all_teams_metadata = all_teams_metadata.drop_duplicates()

        if save_artifacts:
            if not os.path.exists(f"{path_prefix}{ARTIFACTS}"):
                os.makedirs(f"{path_prefix}{ARTIFACTS}")
            if verbose:
                print(f"Saving teams_metadata to artifact: {tm_path}\n")
            all_teams_metadata.to_csv(tm_path)
        return all_teams_metadata


def load_matches_metadata(dataset_path=PATHS.STATSBOMB_DATA, sub_dir='matches', force_create=False, path_prefix='',
                          save_artifacts=False, verbose=False) -> pd.DataFrame:
    '''
    Build matches metadata DataFrame - adds season_name, competition_name for each match in the dataset
    '''
    data = []
    mm_path = path_prefix + PATHS.MATCHES_METADATA_PATH
    if os.path.exists(mm_path) and not force_create:
        if verbose:
            print('\nLoading existing matches_metadata.csv from artifacts directory...')
        return pd.read_csv(mm_path)

    else:
        if verbose:
            print('\nData load STARTED')
        dir_ = f'{dataset_path}/{sub_dir}/'
        competitions_dirs = os.listdir(dir_)
        for competitions_dir_ in tqdm(competitions_dirs, total=len(competitions_dirs)):
            try:
                files_ = os.listdir(os.path.join(dir_, competitions_dir_))
            except NotADirectoryError:
                continue
            for file_name in files_:
                with open(f'{dir_}/{competitions_dir_}/{file_name}') as data_file:
                    data_item = json.load(data_file)
                    for item_ in data_item:
                        data.append(pd.json_normalize(item_, sep="_"))

        if verbose:
            print('Data load COMPLETED\n')
        matches_metadata = pd.concat(data)
        cols_mapper = {'season_season_name': 'season_name', 'competition_competition_name': 'competition_name'}
        matches_metadata.rename(columns=cols_mapper, inplace=True)

        matches_metadata = matches_metadata.drop_duplicates(subset=[COLUMNS.MATCH_ID])

        if save_artifacts:
            if not os.path.exists(f"{ARTIFACTS}"):
                os.makedirs(f"{path_prefix}{ARTIFACTS}")
            if verbose:
                print(f"Saving teams_metadata to artifact: {mm_path}\n")
            matches_metadata.to_csv(mm_path)
        return matches_metadata
