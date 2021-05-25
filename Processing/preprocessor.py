"""
Contains the code need to convert raw scene
or collocated training data into a standard
format that the neural networks can use, as
well as the structure of a scene.
"""

import os
import numpy as np
import pandas as pd
import dask.array as da
import h5py
from glob import glob
from scipy.interpolate import griddata
from satpy import Scene, find_files_and_readers
from pyorbital.orbital import get_observer_look
from pyorbital.astronomy import sun_zenith_angle
from numba import jit, prange
from random import sample
import time

normalisation_values = {  # Input: [offset, denominator]
    'Himawari Band 1 Mean at 2km Resolution': [50., 50.],
    'Himawari Band 1 Sigma at 2km Resolution': [50., 50.],
    'Himawari Band 2 Mean at 2km Resolution': [50., 50.],
    'Himawari Band 2 Sigma at 2km Resolution': [50., 50.],
    'Himawari Band 3 Mean at 2km Resolution': [50., 50.],
    'Himawari Band 3 Sigma at 2km Resolution': [50., 50.],
    'Himawari Band 4 Mean at 2km Resolution': [50., 50.],
    'Himawari Band 4 Sigma at 2km Resolution': [50., 50.],
    'Himawari Band 5 Value at 2km Resolution': [50., 50.],
    'Himawari Band 6 Value at 2km Resolution': [50., 50.],
    'Himawari Band 7 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 8 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 9 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 10 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 11 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 12 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 13 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 14 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 15 Value at 2km Resolution': [273.15, 150.],
    'Himawari Band 16 Value at 2km Resolution': [273.15, 150.],
    'Himawari Solar Zenith Angle': [0., 90.],
    'Himawari Observation Elevation Angle': [0., 90.],
    'Himawari Latitude': [0., 90.],
    'ERA5 Atmosphere Temperature Profile': [273.15, 150.],
    'ERA5 Skin Temperature': [273.15, 50.]
}


# For collocated training data ###

def convert_binary_format_to_bits(array_of_integers):
    """
    Converts 16 bit binary integers into their bit representation.

    :param array_of_integers:
    :return:
    """
    return np.binary_repr(array_of_integers, width=16)


convert_binary_format_to_bits = np.vectorize(convert_binary_format_to_bits)


def convert_binary_to_useful_columns(dataframe):
    """
    Converts the column of 16 bit binary integers into the constituent
    columns described in the CALIOP documentation.

    :param dataframe:
    :return:
    """
    df = dataframe.copy()
    integer_numbers = list(df['CALIOP Vertical Feature Mask (Binary Format)'])
    full_bits = [convert_binary_format_to_bits(integer_number) for integer_number in integer_numbers]
    columns_to_add = {
        'CALIOP Feature Type': [None, -3],
        'CALIOP Feature Type QA': [-3, -5],
        'CALIOP Ice/Water Phase': [-5, -7],
        'CALIOP Ice/Water Phase QA': [-7, -9],
        'CALIOP Feature Sub-Type': [-9, -12],
        'CALIOP Cloud/Aerosol/PSC Type QA': [-12, -13],
        'CALIOP Horizontal Averaging Required': [-13, -16]
    }
    for column, arr_bounds in columns_to_add.items():
        column_data = [[int(bits[arr_bounds[-1]:arr_bounds[0]], 2) for bits in arr] for arr in full_bits]
        df[column] = column_data
    df = df.drop(columns='CALIOP Vertical Feature Mask (Binary Format)')
    return df


def extract_inputs_from_dataframe(dataframe, include_era=True):
    """
    Extracts the inputs assigned in the global dictionary
    "normalisation_values" from the given dataframe.

    :param dataframe:
    :return:
    """
    inputs = []
    if include_era:  # If ERA data is to be included, use all inputs
        norm_vals = normalisation_values
    else:  # If ERA5 data isn't to be used, remove from inputs
        norm_vals = {
            key: item
            for key, item
            in normalisation_values.items()
            if 'ERA5 ' not in key
        }
    for input_var, values in norm_vals.items():
        data = dataframe.copy()[input_var].to_numpy()
        if np.ma.isMA(data[0]):
            data = np.asarray([i.data[0] for i in data])
        inputs.append((data - values[0]) / values[1])
    inputs = np.dstack(tuple(inputs))[0]
    return inputs


def extract_labels_from_dataframe(dataframe, label='binary_id', qa_filter=True, qa_limit=0.7):
    """
    Extracts the labels set by "label" from the given dataframe.
    Will give a quality filter at the value assigned by "qa_limit"
    if "qa_filter" is on. Otherwise, the quality filter will be all
    True values.

    :param dataframe:
    :param label:
    :return:
    """

    labels, qa_mask = None, None

    acceptable_labels = [
        'binary_id',
        'binary_phase',
        'regression_top_heights'
    ]
    df = convert_binary_to_useful_columns(dataframe)
    if label in acceptable_labels:
        if label == 'binary_id':
            labels = np.array([profile[0] for profile in list(df['CALIOP Feature Type'].copy())])
            labels = (labels == 2.).astype('int')  # 0 -> Non-cloud, 1 -> Cloud
            qa_mask = np.full(labels.shape, True)
            if qa_filter:
                qa_mask = np.array([
                    np.all((np.array(profile) > qa_limit) | (np.array(profile) == -9999.))
                    for profile
                    in list(df['CALIOP QA Scores'].copy())
                ])
        elif label == 'binary_phase':
            cloud_mask = np.array([profile[0] for profile in list(df['CALIOP Feature Type'].copy())]) == 2.
            labels = np.array([profile[0] for profile in list(df['CALIOP Ice/Water Phase'].copy())])
            labels = (labels == 2.).astype('int')  # 0 -> Ice, 1 -> Water
            qa_mask = np.full(labels.shape, True)
            if qa_filter:
                qa_mask = np.array([
                    np.all((np.array(profile) > qa_limit) | (np.array(profile) == -9999.))
                    for profile
                    in list(df['CALIOP QA Scores'].copy())
                ])
            qa_mask = qa_mask & cloud_mask
        elif label == 'regression_top_heights':
            cloud_mask = np.array([profile[0] for profile in list(df['CALIOP Feature Type'].copy())]) == 2.
            labels = np.array([profile[0] for profile in list(df['CALIOP Feature Top Altitudes'].copy())])
            labels = (labels + 0.5) / 30.6  # normalised st 0. -> -0.5km, 1. -> 30.1km due to the 5km product
            qa_mask = np.full(labels.shape, True)
            if qa_filter:
                qa_mask = np.array([
                    np.all((np.array(profile) > qa_limit) | (np.array(profile) == -9999.))
                    for profile
                    in list(df['CALIOP QA Scores'].copy())
                ])
            qa_mask = qa_mask & cloud_mask
        return labels, qa_mask
    else:
        raise Exception(f'Label is not in in available labels.\nAvailable labels are: {acceptable_labels}')


def extract_inputs_and_labels_from_dataframe(dataframe, label='binary_id', include_era=True, qa_filter=True,
                                             qa_limit=0.7):
    """
    Extracts both the inputs assigned in the global dictionary
    "normalisation_values" and the labels set by "label" from
    the given dataframe. Will apply a quality filter at the value
    assigned by "qa_limit" if "qa_filter" is on.

    :param dataframe:
    :return:
    """
    data_dict = {}
    inputs = extract_inputs_from_dataframe(
        dataframe=dataframe,
        include_era=include_era
    )
    labels, qa_mask = extract_labels_from_dataframe(
        dataframe=dataframe,
        label=label,
        qa_filter=qa_filter,
        qa_limit=qa_limit
    )
    day_mask = dataframe['Himawari Solar Zenith Angle'].copy()[qa_mask] < 80.
    data_dict['Day'] = {
        'Inputs': inputs[qa_mask][day_mask],
        'Labels': labels[qa_mask][day_mask]
    }
    night_mask = dataframe['Himawari Solar Zenith Angle'].copy()[qa_mask] >= 90.
    data_dict['Night'] = {
        'Inputs': inputs[qa_mask][night_mask],
        'Labels': labels[qa_mask][night_mask]
    }
    twilight_mask = (~day_mask) & (~night_mask)
    data_dict['Twilight'] = {
        'Inputs': inputs[qa_mask][twilight_mask],
        'Labels': labels[qa_mask][twilight_mask]
    }
    return data_dict


def load_collocated_dataframes(target_dir, label='binary_id', include_era=True, qa_filter=True, qa_limit=0.7):
    """
    Load in .h5 collocated files from the target directory and
    return a dictionary of inputs and labels, with label settings
    dictated by the "label", "qa_filter" and "qa_limit" params.

    :param target_dir:
    :param label:
    :param qa_filter:
    :param qa_limit:
    :return:
    """

    def load_and_print(n, m, df_path):
        print(f'Loading Dataframe {n + 1}/{m}                         ', end='\r')
        try:
            df_in = pd.read_hdf(df_path)
            print(f'Dataframe {n + 1}/{m}: Data Read                       ', end='\r')
            data_dict = extract_inputs_and_labels_from_dataframe(
                df_in,
                label,
                include_era,
                qa_filter,
                qa_limit
            )
            print(f'Dataframe {n + 1}/{m}: Data Extracted                          ', end='\r')
            return data_dict
        except Exception as e:
            print(f'Loading Dataframe {n + 1}/{m} failed                   ', end='\r')
            time.sleep(0.5)
            print(e, '                               ', end='\r')
            time.sleep(0.5)
            return None

    available_dfs = glob(os.path.join(target_dir, '**', '*.h5'), recursive=True)
    # TO BE REMOVED: Will cap the load in at 1200 dataframes for testing ###
    max_load = 1200
    if len(available_dfs) > max_load:
        available_dfs = sample(available_dfs, max_load)
    day_inputs = []
    night_inputs = []
    twilight_inputs = []
    day_labels = []
    night_labels = []
    twilight_labels = []
    successful_loads = 0
    for n, df in enumerate(available_dfs):
        sing = load_and_print(n, len(available_dfs), df)
        if sing is not None:
            successful_loads += 1
            day_inputs.append(sing['Day']['Inputs'])
            night_inputs.append(sing['Night']['Inputs'])
            twilight_inputs.append(sing['Twilight']['Inputs'])
            day_labels.append(sing['Day']['Labels'])
            night_labels.append(sing['Night']['Labels'])
            twilight_labels.append(sing['Twilight']['Labels'])
    print(f'{successful_loads} Dataframes Loaded             ')
    data_dict = {}
    data_dict['Day'] = {'Inputs': np.vstack(tuple(day_inputs)), 'Labels': np.hstack(tuple(day_labels))}
    data_dict['Night'] = {'Inputs': np.vstack(tuple(night_inputs)), 'Labels': np.hstack(tuple(night_labels))}
    data_dict['Twilight'] = {'Inputs': np.vstack(tuple(twilight_inputs)), 'Labels': np.hstack(tuple(twilight_labels))}
    return data_dict  # complete dataframe


def split_into_training_and_validation(data_dict, training_frac=0.8):
    """
    Takes a dictionary of inputs and labels and splits it into
    training and validation datasets.

    :param data_dict:
    :return:
    """
    split_data_dict = {}
    if data_dict['Labels'].dtype == type(int()):
        # Ensure an equal number of each class is stored ###
        class_dict = {}
        train_inputs = []
        train_labels = []
        val_inputs = []
        val_labels = []
        min_class_length = min([  # Find the minimum number of entries for the classes
            sum(data_dict['Labels'] == class_number)
            for class_number
            in range(np.max(data_dict['Labels']) + 1)
        ])
        for class_number in range(np.max(data_dict['Labels']) + 1):
            class_mask = (data_dict['Labels'] == class_number)  # Look at only this class
            class_labels = data_dict['Labels'][class_mask]  # Take only the data for this class
            class_inputs = data_dict['Inputs'][class_mask]
            even_prob = min_class_length / len(
                class_labels)  # Assign the probability of values being True in the even masking
            if even_prob >= 1:
                even_prob = 0.9999
            even_mask = np.random.choice(  # Create a mask to take approximately the same number of each class
                a=[True, False],
                size=class_labels.shape,
                p=[even_prob, 1 - even_prob]
            )
            class_dict[str(class_number)] = {  # Add data to class_dict for further processing
                'Inputs': class_inputs[even_mask],
                'Labels': class_labels[even_mask]
            }
        # Randomly select the training data ###
        for class_number, class_data in class_dict.items():
            training_mask = np.random.choice(
                a=[True, False],
                size=class_data['Labels'].shape,
                p=[training_frac, 1 - training_frac]
            )
            train_inputs.append(class_data['Inputs'][training_mask])
            train_labels.append(class_data['Labels'][training_mask])
            val_inputs.append(class_data['Inputs'][~training_mask])
            val_labels.append(class_data['Labels'][~training_mask])
        class_dict = None
        # Split the data into training and validation sets ###
        split_data_dict['Training'] = {
            'Inputs': np.vstack(tuple(train_inputs)),
            'Labels': np.hstack(tuple(train_labels))
        }
        split_data_dict['Validation'] = {
            'Inputs': np.vstack(tuple(val_inputs)),
            'Labels': np.hstack(tuple(val_labels))
        }
    else:
        # Randomly select the training data ###
        training_mask = np.random.choice(
            a=[True, False],
            size=data_dict['Labels'].shape,
            p=[training_frac, 1 - training_frac]
        )
        # Split the data into training and validation sets ###
        split_data_dict['Training'] = {
            'Inputs': data_dict['Inputs'][training_mask],
            'Labels': data_dict['Labels'][training_mask]
        }
        split_data_dict['Validation'] = {
            'Inputs': data_dict['Inputs'][~training_mask],
            'Labels': data_dict['Labels'][~training_mask]
        }
    return split_data_dict


# For scenes ###
def read_h8_folder(fn, verbose=False):
    """
    Converts a folder of raw .hsd data into a satpy Scene, as well as
    load in available bands.

    :param fn: str type. Path to the folder containing
               the .hsd files to be read.
    :param verbose: boolean type. If True, will print out
                    list of available dataset names.
    :return: Satpy Scene of the input data.
    """
    # Find all the .hsd files to be read
    files = find_files_and_readers(
        reader='ahi_hsd',
        base_dir=fn
    )
    # Put all the .hsd files into a satpy Scene
    scn = Scene(
        reader='ahi_hsd',
        filenames=files
    )
    # Find available bands
    bands = scn.available_dataset_names()
    # Load each band
    for band in bands:
        scn.load([band])
    # Print out loaded bands
    if verbose:
        print("Available Band: %s" % bands)
    return scn


def load_jma_cloud_mask(fn):
    """
    Loads the JMA operational binary cloud mask as a numpy array.

    :param fn: str type. Full/path/to/filename of cloud mask file.
    :return: numpy array of binary cloud mask.
    """
    from netCDF4 import Dataset
    # Load netcdf file
    dst = Dataset(fn)
    cloud_mask = np.array(dst['CloudMaskBinary'])
    cloud_mask = cloud_mask.astype('float')
    cloud_mask[cloud_mask == -128] = np.nan
    return cloud_mask


def load_era_dataset(him_name, era_base_dir, var_name='t', multilevel=True):
    """
    Finds and loads the corresponding era5 data for the Himawari-8 scene.

    :param him_name: str type. Himawari-8 scene name.
    :return:
    """
    from glob import glob
    from netCDF4 import Dataset
    if multilevel:
        level_marker = 'pl'
    else:
        level_marker = 'sfc'
    path_to_data = os.path.join(
        era_base_dir,
        '**',
        f'{var_name}_era5_mnth_{level_marker}_{him_name[:6]}01-{him_name[:6]}??.nc'
    )
    # print(path_to_data)
    fname = glob(path_to_data, recursive=True)
    # print(fname)
    dst = None
    for f in fname:
        if dst is None:
            try:
                dst = Dataset(f)
            except:
                pass
        else:
            break
    if dst is None:
        raise Exception(
            'Failed to load ERA5 data. Check if dataset is available in %s' % era_base_dir
        )
    # print(dst)
    time_stamp = int(him_name[-4:-2]) - 1
    if var_name == '2t':
        var_name_mod = 't2m'
    elif var_name == 'ci':
        var_name_mod = 'siconc'
    else:
        var_name_mod = var_name
    if multilevel:
        data_arr_l = dst[var_name_mod][time_stamp, :, 35:686, 958:].data
        data_arr_r = dst[var_name_mod][time_stamp, :, 35:686, :168].data
        data_arr = np.dstack((data_arr_l, data_arr_r))
        data_arr = np.dstack(tuple([data_arr[n, :, :] for n in range(37)]))
    else:
        data_arr_l = dst[var_name_mod][time_stamp, 35:686, 958:].data
        data_arr_r = dst[var_name_mod][time_stamp, 35:686, :168].data
        data_arr = np.hstack((data_arr_l, data_arr_r))
        data_arr[data_arr < 0] = np.nan
    lons, lats = np.meshgrid(
        np.concatenate((dst['longitude'][958:], dst['longitude'][:168])),
        dst['latitude'][35:686],
    )
    return data_arr, lats, lons


def regrid_era_data(era_data, era_lats, era_lons, him_lats, him_lons):
    # Shift longitudes to prevent dateline problems ###
    shifted_era_lons = era_lons.copy() - 140.7
    shifted_era_lons[shifted_era_lons < -180.] += 360.
    shifted_era_lons[shifted_era_lons > 180.] -= 360.
    shifted_him_lons = him_lons.copy() - 140.7
    shifted_him_lons[shifted_him_lons < -180.] += 360.
    shifted_him_lons[shifted_him_lons > 180.] -= 360.
    era_flattened_coords = (shifted_era_lons.flatten(), era_lats.copy().flatten())
    era_flattened_data = era_data.copy().flatten()
    him_coords = (shifted_him_lons, him_lats.copy())
    return griddata(era_flattened_coords, era_flattened_data, him_coords, method='linear')


def downsample_array(array, divide_by):
    """
    Decreases the resolution of the input 2D array to the
    resolution specified by divide_by, e.g. divide_by=2 will
    half the resolution of the array by averaging every 2x2
    grid of pixels into 1 pixel.

    :param array: 2D numpy array of data.
    :param divide_by: int type. The number by which the resolution will be divided by.
    :return: 2D numpy array with 1/(divide_by) the resolution of the input array.
    """
    # Generate the array of mean values ###
    mean_arr = np.zeros((int(array.shape[0] / divide_by), int(array.shape[-1] / divide_by)))
    count_arr = np.zeros((int(array.shape[0] / divide_by), int(array.shape[-1] / divide_by)))
    for i in range(divide_by):
        for j in range(divide_by):
            arr = array[i::divide_by, j::divide_by].copy()
            arr[np.isnan(arr)] = 0.
            mean_arr = mean_arr + arr
            count_arr = count_arr + ~np.isnan(array[i::divide_by, j::divide_by])
    mean_arr = mean_arr / count_arr
    # Generate the array of standard deviation values ###
    std_arr = np.zeros((int(array.shape[0] / divide_by), int(array.shape[-1] / divide_by)))
    for i in range(divide_by):
        for j in range(divide_by):
            arr = array[i::divide_by, j::divide_by].copy()
            arr = np.square(arr - mean_arr)
            arr[np.isnan(arr)] = 0.
            std_arr = std_arr + arr
    std_arr = std_arr / count_arr  # Variance
    std_arr = np.sqrt(std_arr)
    # Return the mean and standard deviation arrays ###
    return mean_arr, std_arr


def upsample_array(array, times_by):
    """
    Increases the resolution of the 2D input array to an
    equivalent 2D array with the resolution specified by
    times_by, e.g. if times_by=2, each value is doubled
    into a 2x2 grid.
    NB//
        This does NOT give extra information. This simply
        allows lower res data to be used w/ higher res data,
        e.g. for RGB false colours.

    :param array: 2D numpy array of data.
    :return: 2D numpy array regridded to (times_by)x the resolution of
             the input array.
    """
    # Define original no. rows
    row_init = array.shape[0]
    # Define final no. rows
    row_fin = times_by * row_init
    # Define initial no. columns
    col_init = array.shape[1]
    # Define final np. columns
    col_fin = times_by * col_init
    # Repeat the columns
    array = np.repeat(array, times_by)
    # Reshape the flattened output
    array = np.reshape(array, [row_init, col_fin])
    # Repeat the rows
    array = np.tile(array, (1, times_by))
    # Return array in its final shape
    return np.reshape(array, [row_fin, col_fin])


def generate_band_arrays(scn):
    """
    Creates a dictionary of arrays for the scene band
    data at 2km resolution.

    :param scn: satpy Scene of Himawari-8 data.
    :return: dictionary of masked arrays
    """
    print('Generating array of all 16 Himawari bands')
    if len(scn.available_dataset_names()) != 16:
        raise Exception('Himawari scene is missing data.\nBands in scene:\n%s'
                        % scn.available_dataset_names)
    half_km = [
        'B03'
    ]
    one_km = [
        'B01',
        'B02',
        'B04'
    ]
    band_data = {}
    for band in scn.available_dataset_names():
        print(f'Loading {band}...')
        band_array = scn[band].data
        if band in half_km or band in one_km:
            if band in half_km:
                band_mean, band_std = downsample_array(
                    array=band_array,
                    divide_by=4
                )
            else:
                band_mean, band_std = downsample_array(
                    array=band_array,
                    divide_by=2
                )
            band_data[f'Himawari Band {str(int(band[1:]))} Mean at 2km Resolution'] = band_mean
            band_data[f'Himawari Band {str(int(band[1:]))} Sigma at 2km Resolution'] = band_std
            # Clean up RAM
            del band_mean
            del band_std
        else:
            band_data[f'Himawari Band {str(int(band[1:]))} Value at 2km Resolution'] = band_array
        # Clean up RAM
        del band_array
    print('All bands loaded')
    return band_data


@jit()
def get_scene_angles(scn,
                     use_driver=False):
    """
    Creates a dictionary containing the scene coordinates,
    observation and solar angles.

    :param scn:
    :param use_driver:
    :return:
    """
    angle_dict = {}
    print('Loading fixed scene angles...')
    # Set the scene time to be halfway through the scan
    scn_time = scn.start_time + (scn.end_time - scn.start_time) / 2
    # Make sure the scene times are the same shape as the lats and lons  
    # scn_time = np.full(scn['B16'].data.shape, scn_time)
    # Driver file name and location
    proc_dir = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
    angles_file = os.path.join(
        proc_dir,
        'Processing',
        'fixed_angles.h5'
    )
    angles_file_exists = os.path.isfile(angles_file)
    if use_driver and angles_file_exists:
        # Load from angles driver file
        fixed_angles = h5py.File(angles_file)
        # Load and assign fixed angles
        lats = fixed_angles['/lats']
        lons = fixed_angles['/lons']
        # sat_azi = fixed_angles['/sat_azi']
        sat_elv = fixed_angles['/sat_elv']
        # Assign as dask arrays
        lats = da.from_array(lats, chunks=scn['B16'].data.chunksize)
        lons = da.from_array(lons, chunks=scn['B16'].data.chunksize)
        # sat_azi = da.from_array(sat_azi, chunks=scn['B16'].data.chunksize)
        sat_elv = da.from_array(sat_elv, chunks=scn['B16'].data.chunksize)
    else:
        # Calculate angles
        # Get the longitudes and latitudes of the scene pixels
        lons, lats = scn['B16'].attrs['area'].get_lonlats(chunks=scn['B16'].data.chunks)
        lons = da.where(lons >= 1e30, np.nan, lons)
        lats = da.where(lats >= 1e30, np.nan, lats)

        sat_azi, sat_elv = get_observer_look(
            sat_lon=scn['B16'].attrs['satellite_longitude'],
            sat_lat=scn['B16'].attrs['satellite_latitude'],
            sat_alt=scn['B16'].attrs['satellite_altitude'] / 1000.,
            utc_time=scn_time,
            lon=lons,
            lat=lats,
            alt=0.
        )
        # sat_azi = da.where(sat_azi >= 1e30, np.nan, sat_azi)
        sat_elv = da.where(sat_elv >= 1e30, np.nan, sat_elv)
    # Store fixed angles
    angle_dict['Himawari Latitude'] = lats
    angle_dict['Himawari Longtiude'] = lons
    angle_dict['Himawari Observation Elevation Angle'] = sat_elv
    # angle_dict['Himawari Observation Azimuth Angle'] = sat_azi
    # Solar Angles #
    print('Loading solar angles...')
    # Get the solar angles
    sunz = sun_zenith_angle(scn_time, lons, lats)
    sunz = da.where(sunz >= 1e30, np.nan, sunz)
    sunz = 90. - sunz

    # solar_elv, solar_azi = get_alt_az(
    #     utc_time=scn_time,
    #    lon=lons,
    #     lat=lats
    # )
    # solar_azi = np.rad2deg(solar_azi)
    #  solar_elv = np.rad2deg(solar_elv)
    #  solar_azi = da.where(solar_azi >= 1e30, np.nan, solar_azi)
    #  solar_elv = da.where(solar_elv >= 1e30, np.nan, solar_elv)

    angle_dict['Himawari Solar Zenith Angle'] = 90. - sunz
    #  angle_dict['Himawari Solar Azimuth Angle'] = solar_azi
    print('All angles loaded')
    return angle_dict



def get_era_data(scn, era_dir):
    era_dict = {}
    # Latitudes and Longitudes ###
    print('Loading ERA Data...')
    lons, lats = scn['B16'].area.get_lonlats()  # Get the longitudes and latitudes of the scene pixels
    # lons[lons == np.inf] = np.nan
    # lats[lats == np.inf] = np.nan
    # Set which ERA5 datasets to load in ###
    multi_level_era_datasets = {
        't': 'ERA5 Atmosphere Temperature Profile'
    }
    single_level_era_datasets = {
        'skt': 'ERA5 Skin Temperature'
    }
    for era_dataset in list(single_level_era_datasets.keys()) + list(multi_level_era_datasets.keys()):
        print(f'Loading ERA dataset: {era_dataset}')
        if era_dataset in multi_level_era_datasets.keys():
            era_data, era_lats, era_lons = load_era_dataset(
                him_name=scn.start_time.strftime('%Y%m%d_%H%M'),
                era_base_dir=era_dir,
                var_name=era_dataset,
                multilevel=True
            )
            ahi_scale_era_data = []
            for i in range(era_data.shape[-1]):
                ahi_scale_era_data.append(regrid_era_data(
                    era_data=era_data[:, :, i],
                    era_lats=era_lats,
                    era_lons=era_lons,
                    him_lats=lats,
                    him_lons=lons,
                ))
            ahi_scale_era_data = np.dstack(tuple(ahi_scale_era_data))
            era_name = multi_level_era_datasets[era_dataset]
        else:
            era_data, era_lats, era_lons = load_era_dataset(
                him_name=scn.start_time.strftime('%Y%m%d_%H%M'),
                era_base_dir=era_dir,
                var_name=era_dataset,
                multilevel=False
            )
            ahi_scale_era_data = regrid_era_data(
                era_data=era_data,
                era_lats=era_lats,
                era_lons=era_lons,
                him_lats=lats,
                him_lons=lons,
            )
            era_name = single_level_era_datasets[era_dataset]
        ahi_scale_era_data[ahi_scale_era_data == np.inf] = np.nan
        era_dict[era_name] = ahi_scale_era_data
    return era_dict


def preprocess_scene(scn, include_era=True, era_dir=None, use_driver=False):
    """
    Takes a satpy Scene and processes into inputs for a
    neural network to analyse, along with the original
    structure of the scene.

    :param scn:
    :return:
    """
    proc_dict = {}
    band_dict = generate_band_arrays(scn)
    angles_dict = get_scene_angles(scn, use_driver)
    if include_era:  # If ERA data is to be included, get the data and use all inputs
        all_data = {  # Expand all input dictionaries into a single dictionary
            **band_dict,
            **angles_dict,
            **get_era_data(scn, era_dir)
        }
        norm_vals = normalisation_values
    else:  # If ERA5 data isn't to be used, don't load it in and remove from inputs
        all_data = {  # Expand all input dictionaries into a single dictionary
            **band_dict,
            **angles_dict
        }
        norm_vals = {
            key: item
            for key, item
            in normalisation_values.items()
            if 'ERA5 ' not in key
        }
    szas = all_data['Himawari Solar Zenith Angle'].compute()
    # Clean up RAM ###
    del band_dict
    del angles_dict
    # Normalise data and put it into standard format ###
    norm_data = [  # Select inputs from normalisation_values dictionary and apply shifts
        (all_data[key] - values[0]) / values[1] for key, values in norm_vals.items()
    ]
    # Clean up RAM 
    del all_data
    print('Combining datasets...')
    norm_data = np.dstack(tuple(norm_data))  # Convert into 5500 x 5500 x n_inputs size array
    norm_data = norm_data.compute() # Force to numpy array for final boolean slicing
    # print(type(norm_data))
    print('Datasets Combined')
    print('Splitting Data by Solar Zenith Angle...')
    norm_data[norm_data == np.inf] = np.nan
    # Check for where nans are and remove and entries containing nans ###
    nan_mask = np.any(np.isnan(norm_data), axis=2)
    norm_data = norm_data[~nan_mask]
    # Apply day mask for separate sza algorithms ###
    day_mask = szas[~nan_mask] < 80.
    night_mask = szas[~nan_mask] >= 90.
    twilight_mask = (~day_mask) & (~night_mask)
    # Add data to dictionary ###
    proc_dict['Day'] = {
        'Inputs': norm_data[day_mask] if sum(day_mask) > 0 else None,
        'Mask': day_mask
    }
    proc_dict['Night'] = {
        'Inputs': norm_data[night_mask] if sum(night_mask) > 0 else None,
        'Mask': night_mask
    }
    proc_dict['Twilight'] = {
        'Inputs': norm_data[twilight_mask] if sum(twilight_mask) > 0 else None,
        'Mask': twilight_mask
    }
    proc_dict['NaN Mask'] = nan_mask
    # Clean up RAM ###
    del szas
    del norm_data
    del day_mask
    del night_mask
    del twilight_mask
    print('Preprocessing of scene complete')
    return proc_dict


if __name__ == '__main__':
    main_dir = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
    example_scene = os.path.join(main_dir, 'Example', '20200105_0500')
    example_scene = read_h8_folder(example_scene)
    example_scene = preprocess_scene(example_scene)
    for sza_region, sza_dict in example_scene.items():
        in_shape = sza_dict['Inputs'].shape
        print(f'For {sza_region} data: {in_shape}')
        in_means = np.mean(sza_dict['Inputs'], axis=1)
        print(f'Input means: {in_means}')
        in_maxes = np.nanmax(sza_dict['Inputs'], axis=1)
        print(f'Input maximums: {in_maxes}')
        in_mins = np.nanmin(sza_dict['Inputs'], axis=1)
        print(f'Input minimums: {in_mins}')
