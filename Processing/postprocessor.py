'''
Contains the code necessary to post-process the outputs
of neural networks used to analyse AHI scenes.
'''


import os
import sys
import json
import numpy as np
import netCDF4 as nc
main_dir = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])


def postprocess_analysed_data(proc_dict, use_era):
    '''
    Takes a dictionary of processed data and converts
    the raw predictions into continuous and/or binary
    masks with the correct shape (5500, 5500) at 2km
    resolution.

    :param proc_dict:
    :return:
    '''
    postproc_dict = {}
    model_tail = 'w-era' if use_era else 'wo-era'
    ### Load in thresholds ###
    with open(os.path.join(main_dir, 'Models', 'optimal_thresholds.json'), 'r') as f:
        opt_thresholds = json.load(f)
    ### Process each nn output into a scene-like format ###
    for key in proc_dict.keys():
        if '_nn' in key: # Make sure to only look at nn outputs
            model_dict = {}
            continuous_labels = np.full(proc_dict['NaN Mask'].shape, -9999.)
            flat_continuous_labels = continuous_labels[~proc_dict['NaN Mask']]
            for Day_or_Night_Twilight, specific_continuous_labels in proc_dict[key].items(): # Make sure to join algorithm outputs
                flat_continuous_labels[proc_dict[Day_or_Night_Twilight]['Mask']] = specific_continuous_labels
            continuous_labels[~proc_dict['NaN Mask']] = flat_continuous_labels
            if 'height' in key:
                model_dict['Continuous Labels'] = (continuous_labels * 30.6) - 0.5
            else:
                model_dict['Continuous Labels'] = continuous_labels
            model_in_thresholds = any([True if '_'.join(key.split('_')[:-2]) in model_name else False for model_name in opt_thresholds.keys()])
            if model_in_thresholds: # If the output was from a binary nn, apply threshold
                binary_labels = np.full(proc_dict['NaN Mask'].shape, -9999)
                flat_binary_labels = binary_labels[~proc_dict['NaN Mask']]
                for Day_or_Night_Twilight, continuous_labels in proc_dict[key].items(): # Make sure to join algorithm outputs
                    flat_continuous_labels[proc_dict[Day_or_Night_Twilight]['Mask']] = continuous_labels
                    model_name = '_'.join(  # Make sure to put the day/night back into the model name
                        key.split('_')[:-1] + [Day_or_Night_Twilight.lower(), 'nn', model_tail]
                    )
                    model_binary_labels = (continuous_labels > opt_thresholds[model_name]).astype('int') # Apply specific threshold
                    flat_binary_labels[proc_dict[Day_or_Night_Twilight]['Mask']] = model_binary_labels
                binary_labels[~proc_dict['NaN Mask']] = flat_binary_labels
                model_dict['Binary Labels'] = binary_labels.astype('int')
                ### Clean up RAM ###
                del flat_binary_labels
                del binary_labels
                del model_binary_labels
            postproc_dict[key] = model_dict
            ### Clean up RAM ###
            del continuous_labels
            del flat_continuous_labels
            del model_dict
    return postproc_dict

def postprocessed_scene_to_nc(scn, postproc_dict, save_directory, use_era):
    '''
    Will take a satpy Scene and the post-processed data from that
    scene and convert it into a .nc file. The file will be saved
    to the <save_directory> directory with the name:
        ahi_nn_analysis_<scene_datetime>.nc
    where <scene_datetime> has the format:
        YYYYmmdd_HHMM

    :param scn:
    :param postproc_dict:
    :param save_directory:
    :return:
    '''
    ### Define the filename and open a new .nc file in <save_directory> ###
    start_string = scn.start_time.strftime('%Y%m%d_%H%M')
    fname = f'ahi_nn_analysis_{start_string}_{'w-era' if use_era else 'wo-era'}.nc'
    fullname = os.path.join(save_directory, fname)
    dst = nc.Dataset(fullname, 'w', format='NETCDF4')
    ### Add basic dimensions to file ###
    dst.createDimension('x', 5500)
    dst.createDimension('y', 5500)
    ### Load in and add lats and lons for full-disk data ###
    lons, lats = scn['B16'].area.get_lonlats()
    longitudes = dst.createVariable('longitudes', 'f8', ('x', 'y',))
    longitudes[:, :] = lons[:, :]
    latitudes = dst.createVariable('latitudes', 'f8', ('x', 'y',))
    latitudes[:, :] = lats[:, :]
    ### Add in data from post-processed dictionary ###
    for model_name in postproc_dict.keys():
        for label_type, label_data in postproc_dict[model_name].items():
            ### If statements ensure that the label data is given the correct name according to CF conventions ###
            if 'cloud_id' in model_name:
                if 'Binary' in label_type:
                    variable_name = 'cloud_binary_mask'
                    v_unit = 'None'
                    v_min = 0
                    v_max = 1
                elif 'Continuous' in label_type:
                    variable_name = 'cloud_continuous_mask'
                    v_unit = 'None'
                    v_min = 0.
                    v_max = 1.
            elif 'cloud_phase' in model_name:
                if 'Binary' in label_type:
                    variable_name = 'thermodynamic_phase_of_cloud_water_particles_at_cloud_top'
                    v_unit = 'None'
                    v_min = 0
                    v_max = 1
                elif 'Continuous' in label_type:
                    variable_name = 'predicted_thermodynamic_phase_of_cloud_water_particles_at_cloud_top'
                    v_unit = 'None'
                    v_min = 0.
                    v_max = 1.
            elif 'cloud_top_height' in model_name:
                variable_name = 'cloud_top_altitude'
                v_unit = 'km'
                v_min = -0.5
                v_max = 30.1
            else:
                raise Exception(
                    'Model has not been added to into post-processing. \n' +
                    'Please update in postprocessor.'
                )
            variable = dst.createVariable(
                variable_name,
                'f8',
                ('x', 'y',),
                fill_value=-9999 if label_type == 'Binary' else -9999.
            )
            variable[:, :] = label_data[:, :]
            variable.units = v_unit
            variable.valid_min = v_min
            variable.valid_max = v_max
            variable.scale_factor = 1.0
    dst.close()


