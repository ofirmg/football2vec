"""
Created on September 5 2021

This module contains several utils functions for processing StatsBomb data.

@author: Ofir Magdaci (@Magdaci)

"""

import ast
import operator
import os

import numpy as np
from PIL import Image

from lib.params import CONSTANTS, PATHS


def to_metric_centered_coordinates(data, field_dimensions=CONSTANTS.PITCH_DIMENSIONS):
    '''
    Convert positions units to meters with origin at centre circle
    '''

    x_columns = [c for c in data.columns if (c[-2:].lower() == '_x')]
    y_columns = [c for c in data.columns if c[-2:].lower() == '_y']
    for x_col in x_columns:
        data[x_col] = yard_2_meter(data[x_col]) - 0.5 * field_dimensions[0]
    for y_col in y_columns:
        data[y_col] = -1 * (yard_2_meter(data[y_col]) - 0.5 * field_dimensions[1])

    for col_ in [col for col in data.columns if '_location' in col or 'location' in col]:
        data[col_] = data[col_].apply(lambda str_: ast.literal_eval(str_) if isinstance(str_, str) else str_)
        data[col_] = data[col_].apply(lambda val: (yard_2_meter(val[0]) - 0.5 * field_dimensions[0],
                                                   -1 * (yard_2_meter(val[1]) - 0.5 * field_dimensions[1]))
        if isinstance(val, tuple) or isinstance(val, list) else val)
    return data


def yard_2_meter(val):
    return 0.9144 * val


def get_player_image(player_name, use_real_players_images=True):
    '''
    Return player image according to player_name. If not image file match this name, most similar file will be returned,
        or a placeholder in case of no matches at all.
    :param player_name: name of players as appears in StatsBomb dataset (str).
    :param use_real_players_images: if True, the function will search for a file name most similar to the given
    player_name. Else - use placeholder image
    :return: PIL.Image object
    '''
    if not use_real_players_images:
        return os.path.join(Image.open(PATHS.PLAYERS_IMAGES, f'{CONSTANTS.PLAYER_IMAGE_PLACEHOLDER}.png'))

    player_name = player_name.replace(' ', '_').lower()
    # ITERATE OVER IMAGES
    file_paths = list(os.listdir(PATHS.PLAYERS_IMAGES))
    if f"{player_name}.png" in file_paths:
        return Image.open(os.path.join(PATHS.PLAYERS_IMAGES, f'{player_name}.png'))
    if f"{player_name}.jpg" in file_paths:
        return Image.open(os.path.join(PATHS.PLAYERS_IMAGES, f'{player_name}.jpg'))

    matching = {file_: 0 for file_ in os.listdir(PATHS.PLAYERS_IMAGES)}

    # Split
    files_formats = {}
    player_name_parts = set(player_name.lower().split('_'))
    for image_path in file_paths:
        image_file = image_path.split('/')[-1].split('.')[0]
        # Calculate matching scores
        image_file_parts = set(image_file.lower().split('_'))
        matching[image_file] = len(image_file_parts.intersection(player_name_parts)) / len(image_file_parts)
        files_formats[image_file] = image_path.split('/')[-1].split('.')[1]

    # TAKE NAME WITH HIGHEST COVERAGE
    player_image = max(matching.items(), key=operator.itemgetter(1))[0]

    # IF HIGHEST COVERAGE IS ZERO -> RETURN PLACEHOLDER
    if matching[player_image] > 0:
        img = Image.open(os.path.join(PATHS.PLAYERS_IMAGES, f'{player_image}.{files_formats[player_image]}'))
        img.save(os.path.join(PATHS.PLAYERS_IMAGES, f'{player_name}.{files_formats[player_image]}'))
        return img
    else:
        return os.path.join(Image.open(PATHS.PLAYERS_IMAGES, f'{CONSTANTS.PLAYER_IMAGE_PLACEHOLDER}.png'))


def get_location_bin(x, y, pitch_dimensions=CONSTANTS.PITCH_DIMENSIONS, output='bin_rel',
                     num_x_bins: int = 5, num_y_bins: int = 5, rel_coordinates=True) -> (int, int):
    '''

    :param x: float, x value [-pitch_dimensions[0] / 2, pitch_dimensions[0] / 2]
    :param y: float, y value [-pitch_dimensions[1] / 2, pitch_dimensions[1] / 2]
    :param pitch_dimensions: (x, y) of pitch dimensions
    :param num_x_bins: number of bins to split the length of the pitch (along pitch_dimensions[0])
    :param num_y_bins: number of bins to split the width of the pitch (along pitch_dimensions[1])
    :param rel_coordinates: if True, coordinates assumed to be relative to pitch center
    :param output: 'bin_name', 'bin_rel', or 'bin_ix'
    :return:
    '''

    bin_names = {'x': {3: ['back', 'med', 'fwd'],
                       4: ['back', 'mback', 'mfwd', 'fwd']},
                 'y': {3: ['left', 'center', 'right'],
                       4: ['left', 'mleft', 'mright', 'right'],
                       5: ['left', 'mleft', 'enter', 'mright', 'right']}}

    bin_x_width, bin_y_width = np.ceil(pitch_dimensions[0] / num_x_bins), np.ceil(pitch_dimensions[1] / num_y_bins)

    if rel_coordinates:
        x, y = x + pitch_dimensions[0] / 2, y + pitch_dimensions[1] / 2

    # Extract bin values [0, num bins - 1]
    bin_x = int(min(np.floor(x / bin_x_width), num_x_bins - 1))
    bin_y = int(min(np.floor(y / bin_y_width), num_y_bins - 1))

    if output == 'bin_ix':
        return bin_x, bin_y
    elif output == 'bin_rel':
        return f"({str(bin_x + 1)}/{str(num_x_bins)}, {str(bin_y + 1)}/{str(num_y_bins)})"
    else:
        x_labels, y_labels = bin_names['x'][num_x_bins], bin_names['y'][num_y_bins]
        return x_labels[bin_x], y_labels[bin_y]
