'''
Will process a scene using all the models in the package
and save a .nc file containg all the model outputs for
the scene to a given directory.
'''

import sys
import os
import click
from glob import glob
import numpy as np
from time import time, localtime
import json
import warnings
main_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(main_dir)
import Processing.preprocessor as prep
import Processing.postprocessor as postp
import Processing.kerasNN as KNN

warnings.filterwarnings("ignore")

def pad_print(to_print, r=True):
    terminal_width = os.get_terminal_size()[0]
    padded_string = str(to_print).ljust(terminal_width, ' ')
    if r:
        print(padded_string, end='\r')
    else:
        print(padded_string)

def full_analysis(full_path_to_scene, save_directory, era_directory, use_era, night_mode=False, all=True, analysis=['cloud_id']):
    '''
    Takes an AHI data folder for a scene and analyses it.
    Will output a .nc file containing:
        - cloud identification mask (binary)
        - cloud identification mask (raw, continuous values)
        - cloud phase (binary)
        - cloud phase (raw, continuous values)
        - cloud top height (value in km)
    if <all> is True and all the models are available. Otherwise,
    the NNs to be used can be set using the <analysis> parameter.
    The available models can be found under AHINN/Models.
    NB//
         Only use the name of the network type, e.g. cloud_id.
         Application of night and day algorithms is automated.
    The .nc file will have the name:
        ahi_nn_analysis_<scene_datetime>.nc
    where <scene_datetime> has the format:
        YYYYmmdd_HHMM
    from the scene being analysed. The .nc file will be stored
    in the <save_directory> directory.
    NB//
        The algorithms will be applied across the whole scene,
        even though the pixels may not be the correct type. Make
        sure to mask the non-identification values with the binary
        masks to get accurate representations of the scene.

    :param full_path_to_scene:
    :param save_directory:
    :param all:
    :param analysis:
    :return:
    '''
    ### Load scene and preproces it ###
    scn = prep.read_h8_folder(full_path_to_scene)
    proc_dict = prep.preprocess_scene(
        scn=scn, 
        era_dir=era_directory,
        include_era=use_era
    )
    ### Check for available
    model_tail = 'w-era' if use_era else 'wo-era'
    available_models = glob(os.path.join(main_dir, 'Models', f'*_nn_{model_tail}.h5'))
    ### Check if <all> is True and correct available models if <all> is False ###
    if not all:
        available_models = [
            os.path.join(
                os.path.dirname(available_model),
                '_'.join(
                    os.path.basename(available_model).split('_')[:-3] +
                    [os.path.basename(available_model).split('_')[-2]]
                )
            )
            for available_model
            in available_models
            if '_'.join(os.path.basename(available_model).split('_')[:2]) in analysis
        ]
    else:
        available_models = [
            os.path.join(
                os.path.dirname(available_model),
                '_'.join(
                    os.path.basename(available_model).split('_')[:-3] +
                    [os.path.basename(available_model).split('_')[-2]]
                )
            )
            for available_model
            in available_models
        ]
    available_models = list(set(available_models)) # Remove duplicates
    for model in available_models:
        print(model)
    dn_dict = {
        'day': 'Day',
        'night': 'Night',
        'twilight': 'Twilight'
    }
    for model in available_models:
        base_model_name = os.path.basename(model)
        model_dir = os.path.dirname(model)
        base_model_dict = {}
        for day_or_night_twilight, Day_or_Night_or_Twilight in dn_dict.items():
            inputs = proc_dict[Day_or_Night_or_Twilight]['Inputs']
            if inputs is not None:
                model_day_or_night_twilight = 'night' if night_mode else day_or_night_twilight
                specific_model_name = '_'.join(
                    base_model_name.split('_')[:-1] + [model_day_or_night_twilight, 'nn', f'{model_tail}.h5']
                )
                model = KNN.KerasNeuralNetwork(
                    load_nn=True,
                    model_dir=model_dir,
                    model_name=specific_model_name
                )
                predictions = model.predict(inputs)
                base_model_dict[Day_or_Night_or_Twilight] = predictions.flatten()
                ### Clean up RAM ###
                del inputs
                del model
                del predictions
        proc_dict[base_model_name] = base_model_dict
    proc_dict = postp.postprocess_analysed_data(
        proc_dict, 
        use_era,
        night_mode
    )
    postp.postprocessed_scene_to_nc(
        scn,
        proc_dict,
        save_directory,
        use_era
    )

example_ahi_folder = glob(os.path.join(main_dir, 'Example', '**', '*.DAT*'), recursive=True)
example_ahi_folder = os.path.split(os.path.dirname(example_ahi_folder[0]))[-1] if len(example_ahi_folder) > 0 else '20200105_0500'
@click.command()
@click.option(
    '--path_to_scene', '-p',
    default=os.path.join(main_dir, 'Example', example_ahi_folder),
    help='The full/path/to/scene, including the name of the folder ' +
         'that contains the scene data to be analysed. The scene folder ' +
         'has the format YYYYmmdd_HHMM.'
)

@click.option(
    '--save_dir', '-s',
    default=os.path.join(main_dir, 'Example'),
    help='The directory where the analysis data will be saved after processing.'
)

@click.option(
    '--era_dir', '-e',
    default=os.path.join(main_dir, 'Example', 'ERA5'),
    help='The directory where the ERA5 data for the scene can be found.'
)

@click.option(
    '--interactive', '-i',
    default='True',
    help='Will ask the user if they want to analysis the scene, giving ' +
         'the directory the scene is held and the scene name.'
)

@click.option(
    '--use_defaults', '-d',
    default='False',
    help='Will use the defualts from defualts.json for processing. ' +
         'This will overide the Click default options, even if ' + 
         'they are inputted by the user.'
)

@click.option(
    '--use_era', '-u',
    default='True',
    help='Will use models trained to use ERA5 data if True or ' +
         'models that only require AHI files if False.'
)
@click.option(
    '--night_mode', '-n',
    default='False',
    help='Will force the analysis to be carried out by the night models only, even during day and twilight.'
)

def main(path_to_scene, save_dir, era_dir, interactive, use_defaults, use_era, night_mode):
    '''
    Will analyse the scene specified by path_to_scene and store the output data in save_dir.
    If interactive is True, will wait for user input before carrying analysis. Can be turned
    off for mass-analysis.
    '''
    start = time()
    use_defaults = True if use_defaults.lower() in ['true', '1', 't', 'y', 'yes'] else False
    use_era = True if use_era.lower() in ['true', '1', 't', 'y', 'yes'] else False
    night_mode = True if use_era.lower() in ['true', '1', 't', 'y', 'yes'] else False
    if use_defaults:
        with open(os.path.join(main_dir, 'defaults.json'), 'r') as f:
            defaults_dict = json.load(f)
    # Set correct path to the AHI folder for analysis
    corrected_path_to_scene = path_to_scene.split(os.sep)
    corrected_path_to_scene = corrected_path_to_scene[:-1] if corrected_path_to_scene[-1] == '' else corrected_path_to_scene
    corrected_path_to_scene = os.sep.join(corrected_path_to_scene)
    folder_name = corrected_path_to_scene.split(os.sep)[-1]
    if len(corrected_path_to_scene.split(os.sep)) > 1:
        folder_location = os.sep.join(corrected_path_to_scene.split(os.sep)[:-1])
    elif use_defaults:
        AHIBaseDir = defaults_dict['path.AHIBaseDir'] if defaults_dict['path.AHIBaseDir'] != "BLANK" else None
        if AHIBaseDir is not None:
            folder_location = AHIBaseDir
        else:
            raise Exception(
                'path.AHIBaseDir is not set.\n' +\
                'Please set path to default AHI directory in:\n' +\
                '%s' % os.path.join(main, 'defaults.json')
            )
    else:
        raise Exception(
            'Full path to AHI folder has not been given and defaults have not been used. If stored locally, use ./YYYYmmdd_HHMM as input.'
        )
    # Set the correct path to the save directory
    if use_defaults and save_dir == os.path.join(main_dir, 'Example'):
        SaveDir = defaults_dict['path.SaveDir'] if defaults_dict['path.SaveDir'] != "BLANK" else None
        if SaveDir is not None:
            save_dir = SaveDir
        else:
            raise Exception(
                'path.SaveDir is not set.\n' +\
                'Please set path to default save directory in:\n' +\
                '%s' % os.path.join(main, 'defaults.json')
            )
    # Set path to ERA5 data (37 pressure level and single level)
    if use_defaults and era_dir == os.path.join(main_dir, 'Example', 'ERA5'):
        ERA5Dir = defaults_dict['path.ERA5BaseDir'] if defaults_dict['path.ERA5BaseDir'] != "BLANK" else None
        if ERA5Dir is not None:
            era_dir = ERA5Dir
        else:
            raise Exception(
                'path.ERA5Dir is not set.\n' +\
                'Please set path to default ERA5 data directory in:\n' +\
                '%s' % os.path.join(main, 'defaults.json')
            )
    # Check if the code is to be run interactively and run the full_analysis code
    era_msg_a = 'using' if use_era else 'without using'
    era_msg_b = 'w/' if use_era else 'w/o'   
    if interactive.lower() in ['true', '1', 't', 'y', 'yes']:
        if click.confirm(f'Want to carry out analysis for {folder_name} located at {folder_location} {era_msg_a} ERA5 data?', default=True):
            pad_print(f'Analysing {folder_name} {era_msg_b} ERA5 data...', r=False)
            full_analysis(  # Carry out full analysis
                full_path_to_scene=path_to_scene,
                save_directory=save_dir,
                era_directory=era_dir,
                use_era=use_era,
                night_mode=night_mode
                all=True
            )
            pad_print(f'Saving ahi_nn_analysis_{folder_name}.nc in {save_dir}', r=False) 
    else:
        print(f'Analysing {folder_name} {era_msg_b} ERA5 data...')
        full_analysis(  # Carry out full analysis
            full_path_to_scene=path_to_scene,
            save_directory=save_dir,
            era_directory=era_dir,
            use_era=use_era,
            night_mode=night_mode
            all=True
        )
        print(f'Saving ahi_nn_analysis_{folder_name}.nc in {save_dir}')
    print(f'Took {round((time() - start)/60., 2)}mins')
    return None

if __name__ == '__main__':
    main()


