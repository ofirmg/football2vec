"""
Created on September 5 2021

Main run file of football2vec. Covers building data objects and language models, using StatsBomb data.

Data and pre-made artifacts for download are available via the README file

@author: Ofir Magdaci (@Magdaci)

"""

from datetime import datetime
import numpy as np
from lib.data_processing import build_data_objects
from lib.models import build_language_models

force_create = False  # Whether to force override all artifacts, or to try load existing artifacts
verbose = True  # Prints control
plotly_export = False  # Whether to export Plotly figures to Plotly studio (see https://chart-studio.plotly.com)
save_artifacts = True  # Whether to save the artifacts in to params.PATHS.ARTIFACTS

if __name__ == '__main__':
    t0 = datetime.now()
    build_data_objects(verbose=verbose,
                       force_create=force_create,
                       plotly_export=plotly_export,
                       save_artifacts=save_artifacts)

    t1 = datetime.now()
    action_2_vec, player_2_vec_model, models_outputs = build_language_models(verbose=verbose,
                                                                             force_create=force_create,
                                                                             plotly_export=plotly_export,
                                                                             save_artifacts=save_artifacts)
    t_end = datetime.now()
    print('Total run time:', np.round((t_end-t0).seconds/60, 2), 'minutes')
    print('Total run time for build_data_objects:', np.round((t1-t0).seconds/60, 2), 'minutes')
    print('Total run time for build_language_models:', np.round((t_end-t1).seconds/60, 2), 'minutes')

    import platform
    print('Machine specs:')
    print('- Machine:', platform.machine())
    print('- Machine version', platform.version())
    print('- Machine platform', platform.platform())
    print('- Machine uname', platform.uname())
    print('- Machine system', platform.system())
    print('- Machine processor', platform.processor())