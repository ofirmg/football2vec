"""
Created on September 5 2021

This module holds definitions for identifying patterns, actions or a collection of them by type, outcome, etc.
It is mainly used by explaines.py and data_processing.py. Some are not used and are merely for further demonstration.

@author: Ofir Magdaci (@Magdaci)

"""

import ast
import re
import warnings
import numpy as np
import pandas as pd

from lib.params import COLUMNS, CONSTANTS
from lib.utils import yard_2_meter


class Pattern(object):
    def __init__(self, words_to_match: list, words_to_exclude: list):
        self.words_to_match = words_to_match
        self.words_to_exclude = words_to_exclude

    def search(self, _str: str):
        raise NotImplementedError("")


class ANDPattern(Pattern):
    def search(self, _str: str):
        for _word in self.words_to_exclude:
            if _word in _str:
                return False
        for _word in self.words_to_match:
            if _word not in _str:
                return False
        return True


class ORPattern(Pattern):
    def search(self, _str: str):
        for _word in self.words_to_exclude:
            if _word in _str:
                return False
        for _word in self.words_to_match:
            if _word in _str:
                return True
        return False


# Regex patterns of tokens families
in_box_scoring_pattern = re.compile("\(5\/5,[2-4]{1}\/5\)\<shot\>")
out_box_scoring_pattern = re.compile("\([3-4]\/5,[1-5]{1}\/5\)\<shot\>")

passes_to_right_pattern = re.compile("\(3\/5,3\/5\)\<pass\>\:\( \- \>")
passes_forward_pattern = re.compile("\(3\/5,3\/5\)\<pass\>\:\( \^ \|")
passes_from_right_flank_pattern = re.compile("\(5\/5,5\/5\)\<pass\>\:\( \- \<")

forward_pressure_pattern = re.compile("\([3-5]\/5,[1-5]{1}\/5\)\<pressure\>")

dribble_pattern = re.compile("\([3-5]\/5,[1-5]{1}\/5\)\<dribble\>:\|outcome=complete")
dribble_past_pattern = re.compile("\([3-5]\/5,[1-5]{1}\/5\)\<dribbled_past\>")
flank_dribble_pattern = re.compile("\([1,5]\/5,[1-5]{1}\/5\)\<dribble\>:\|outcome=complete")
flank_dribble_past_pattern = re.compile("\(1,5]\/5,[1-5]{1}\/5\)\<dribbled_past\>")

# Improving
better_shots = [{'pattern': ANDPattern(['<shot>', 'outcome=blocked'], []),
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

passes_backwards = [{'pattern': ANDPattern(['<pass>', '^'], []),
                     'switch_from': '^',
                     'switch_to': 'v'}]

worse_shots = [{'pattern': ANDPattern(['<shot>', 'outcome=goal'], []),
                'switch_from': 'outcome=goal',
                'switch_to': 'outcome=wayward'}]

better_dribble = [{'pattern': ANDPattern(['<dribble>', 'outcome=incomplete'], []),
                   'switch_from': 'incomplete',
                   'switch_to': 'complete'}]

worse_dribble = [{'pattern': ANDPattern(['<dribble>', 'outcome=complete'], []),
                  'switch_from': 'complete',
                  'switch_to': 'incomplete'}]
switch_to_right_leg = [{'pattern': ORPattern(['left_foot'], []),
                        'switch_from': 'left_foot',
                        'switch_to': 'right_foot'}]


def _search(_pattern, _token):
    '''
    Search within _token using _pattern object - regex of Pattern
    :param _pattern: pattern to match in the token
    :type _pattern: regex pattern or Pattern object
    :param _token: token (str) to search in
    :return: regex search results / book result of Pattern
    '''
    if isinstance(_pattern, Pattern):
        return _pattern.search(_token)
    else:
        return re.search(_pattern, _token)


def get_tokens_by_regex_pattern(vocabulary: list, re_pattern):
    '''
        Function receives a vocabulary (list) and a regex pattern and return all matching tokens (or None if no match)
        :param vocabulary: list of string tokens that form our vocabulary
        :param re_pattern: regex pattern to match
        :return: Bool of the condition result
    '''
    relevant_tokens = [token for token in vocabulary if re.search(re_pattern, token)]
    if len(relevant_tokens) > 0:
        return relevant_tokens
    else:
        warnings.warn(f"No match for pattern {str(re_pattern)}")
        return []


def is_normal_goal_token(token):
    '''
        Function receives a token/word (str) and check it this token/word is a normal shot goal
        :param token: string, description of a word in our football language
        :return: Bool of the condition result
    '''
    return True if 'outcome=goal' in token and '|normal|' in token else False


def is_head_goal_token(token):
    '''
        Function receives a token/word (str) and check it this token/word is a header goal
        :param token: string, description of a word in our football language
        :return: Bool of the condition result
    '''
    return True if 'outcome=goal' in token and '|head|' in token else False


# Check-if functions, receive events, not shots necessarily. They can be executed over all events

def check_if_shot_scored(action: pd.Series):
    '''
        Function receives an action event as pd.Series and check if this action is a successful shot (i.e., goal)
        :param action: StatsBomb event row, pd.Series.
        :return: Bool of the result or np.nan in case the Series does not have COLUMNS.ACTION_TYPE field
    '''
    if COLUMNS.ACTION_TYPE in action:
        if action[COLUMNS.ACTION_TYPE].lower() == 'shot':
            outcome_ = action.get('shot_outcome_name', np.nan)
            if outcome_ is not np.nan and outcome_.lower() == 'goal':
                return True
            else:
                return False
    return np.nan


def check_if_one_one_one_chance(action: pd.Series):
    '''
    Function receives an action event as pd.Series and check if this action is an one-on-one shot
    Assuming relative coordinates around pitch center
    :param action: StatsBomb event row, pd.Series.
    :return: Bool of the result or np.nan in case the Series does not have COLUMNS.ACTION_TYPE field
    '''
    if COLUMNS.ACTION_TYPE in action:
        if action[COLUMNS.ACTION_TYPE].lower() == 'shot':
            shot_freeze = action.get('shot_freeze_frame', np.nan)
            if shot_freeze is not np.nan:
                # Shot location is centric around pitch-mid pointץ Units - meters
                shot_x_location = action[COLUMNS.LOCATION][0]
                for players_location in shot_freeze:
                    # Skip goalkeeper and teammates
                    if players_location['position'] == 'Goalkeeper' or players_location['teammate']:
                        continue
                    # These location starts from left bottom corner of the pitch. Units - yards
                    player_location_x = yard_2_meter(players_location['location'][0]) - \
                                        CONSTANTS.PITCH_DIMENSIONS[0] / 2

                    if shot_x_location < player_location_x:
                        return False
                # Else -> the shooter is closest
                return True
            else:
                return False
    return np.nan


def check_if_shot_outside_box(action: pd.Series, pitch_dimensions=CONSTANTS.PITCH_DIMENSIONS):
    '''
    Function receives an action event as pd.Series and check if this action is an outside-box shot
    Assuming relative coordinates around pitch center
    :param action: StatsBomb event row, pd.Series.
    :param pitch_dimensions: dimensions in meters of the pitch, to calculate box location
    :return: Bool of the result or np.nan in case the Series does not have COLUMNS.ACTION_TYPE field
    '''
    if action[COLUMNS.ACTION_TYPE] != 'Shot':
        return np.nan

    if pd.isna(action['location']):
        return np.nan

    if isinstance(action['location'], str):
        x, y = ast.literal_eval(action['location'])
    elif isinstance(action['location'], list) or isinstance(action['location'], tuple):
        x, y = action['location']
    else:
        print("action['location]", action['location'])
        raise ValueError("Unfamiliar value for action location:", action['location'])

    # Within box if y is within +- 16.5 from y center and distance from x to x end of pitch < 16.5
    # or if x distance from goal is more than 16.5 meters
    if x > (pitch_dimensions[0] / 2) - 16.5 and abs(y) < 16.5:
        return False
    else:
        return True


def check_if_dribble_won(action: pd.Series):
    '''
    Function receives an action event as pd.Series and check if this action is a successful dribble
    :param action: StatsBomb event row, pd.Series.
    :return: Bool of the result or np.nan in case the Series does not have COLUMNS.ACTION_TYPE field
    '''
    if COLUMNS.ACTION_TYPE in action:
        if action[COLUMNS.ACTION_TYPE].lower() == 'dribble':
            outcome_ = action.get('dribble_outcome_name', np.nan)
            if outcome_ is not np.nan and outcome_.lower() == 'complete':
                return True
            elif outcome_ is not np.nan and outcome_.lower() == 'incomplete':
                return False
            else:
                raise ValueError("Unfamiliar value for dribble_outcome_name:", outcome_)
    return np.nan
