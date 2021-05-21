"""
Contains the code to plot the outputs of the neural networks and
RGB images of a scene.
"""

import sys
import os
import click
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as feature
import numpy as np
from datetime import datetime as dt
import netCDF4 as nc

main_dir = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
sys.path.append(main_dir)
from Processing.preprocessor import read_h8_folder, downsample_array


def plot_nn_output(dataset, variable_name, dataset_datetime, apply_mask=False):
    """
    Takes a neural network output as a 2d array and plots the
    values in a standard format map over the AHI observable area
    or over Australia.

    :param dataset:
    :param variable_name:
    :param dataset_datetime:
    :param apply_mask:
    :return:
    """
    cmap = plt.get_cmap('bwr')
    cmap.set_bad(color='k', alpha=1.)
    v_data = np.array(dataset[variable_name])
    v_data[v_data == dataset[variable_name]._FillValue] = np.nan
    if 'cloud' in variable_name and 'mask' not in variable_name:
        cloud_mask = np.array(dataset['cloud_binary_mask']) == 1
        v_data[~cloud_mask] = np.nan
    if apply_mask:
        fsize = (15, 10)
        xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
        ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
        mask = np.full(v_data.shape, False)
        min_y, max_y = int(6500 / 11000 * v_data.shape[0]), int(9750 / 11000 * v_data.shape[0])
        min_x, max_x = int(2900 / 11000 * v_data.shape[0]), int(8250 / 11000 * v_data.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        v_data = v_data[mask].reshape(new_shape)
    else:
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    im = ax.imshow(
        v_data,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        vmin=dataset[variable_name].valid_min,
        vmax=dataset[variable_name].valid_max
    )
    ax.set_title(
        ' '.join([
            a[0].upper() + a[1:]
            if len(a) > 2
            else a.upper()
            if len(a) == 1
            else a
            for a
            in variable_name.split('_')
        ]),
        fontweight='bold',
        loc='left',
        fontsize=12
    )
    string_time = dataset_datetime.strftime('%d %B %Y %H:%M UTC')
    ax.set_title(
        string_time,
        loc='right',
        fontsize=12
    )
    ax.text(
        0.5,
        -0.05,
        ' ',
        size=12,
        ha='center',
        transform=ax.transAxes
    )
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    plt.colorbar(
        im,
        cax=cbaxes,
        orientation='vertical',
        label=r'%s [%s]' % (variable_name, dataset[variable_name].units)
    )
    return fig


def plot_full_analysis(path_to_dataset, save_dir, aus_only=False):
    """
    Takes a .nc file of NN outputs from the full analysis of
    an AHI scene and plots them in standard format. Can be
    applied only over Australia.

    :param path_to_dataset:
    :param save_dir:
    :param aus_only:
    :return:
    """
    fname_stringtime = os.path.basename(path_to_dataset)[-16:-3]
    dst = nc.Dataset(path_to_dataset)
    plottables = [
        vname
        for vname
        in dst.variables.keys()
        if vname not in ['longitudes', 'latitudes']
    ]
    for plottable in plottables:
        print(plottable)
        fig = plot_nn_output(
            dataset=dst,
            variable_name=plottable,
            dataset_datetime=dt.strptime(fname_stringtime, '%Y%m%d_%H%M'),
            apply_mask=aus_only
        )
        fig_name = '_'.join([plottable, fname_stringtime])
        if aus_only:
            fig_name = '_'.join([fig_name, 'australia_only'])
        fig_name = fig_name + '.png'
        plt.savefig(
            os.path.join(save_dir, fig_name),
            bbox_inches='tight'
        )
    return None


def plot_single_channel(ahi_scn, band_number=3, apply_mask=False):
    """
    Plots a single

    :param ahi_scn:
    :param band_number:
    :param apply_mask:
    :return:
    """
    band_wavelengths = [
        '0.455',
        '0.510',
        '0.645',
        '0.860',
        '1.61',
        '2.26',
        '3.85',
        '6.25',
        '6.95',
        '7.35',
        '8.60',
        '9.63',
        '10.45',
        '11.2',
        '12.35',
        '13.3'
    ]
    band_indicator = f"B{int(band_number):02d}"
    band_values = np.array(ahi_scn[band_indicator])
    if apply_mask:
        fsize = (15, 10)
        xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
        ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
        mask = np.full(band_values.shape, False)
        min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
        min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        band_values = band_values[mask]
        if band_values.shape != new_shape:
            band_values = band_values.reshape(new_shape)
    else:
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    im = ax.imshow(
        band_values,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap='bone',
        vmin=0 if int(band_number) < 7 else 273.15 - 150,
        vmax=100 if int(band_number) < 7 else 273.15 + 150
    )
    ax.set_title(
        f'Band {str(int(band_number))}',
        fontweight='bold',
        loc='left',
        fontsize=12
    )
    string_time = ahi_scn.start_time.strftime('%d %B %Y %H:%M UTC')
    ax.set_title(
        string_time,
        loc='right',
        fontsize=12
    )
    ax.text(
        0.5,
        -0.05,
        'Band Central Wavelength: ' + str(band_wavelengths[int(int(band_number) - 1)]) + r'$\mu m$',
        size=12,
        ha='center',
        transform=ax.transAxes
    )
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    plt.colorbar(
        im,
        cax=cbaxes,
        orientation='vertical',
        label=r'%s [%s]' % ('Reflectance' if int(band_number) < 7 else 'BT', '%' if int(band_number) < 7 else 'K')
    )
    return fig


def true_colour_rgb(ahi_scn, apply_mask=False):
    gamma = 2.
    bands = {
        'r': downsample_array(np.array(ahi_scn['B03']), 4)[0],
        'g': downsample_array(np.array(ahi_scn['B02']), 2)[0],
        'b': downsample_array(np.array(ahi_scn['B01']), 2)[0],
    }
    if apply_mask:
        band_values = bands['r']
        mask = np.full(band_values.shape, False)
        min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
        min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        for band, band_values in bands.items():
            bands[band] = band_values[mask].reshape(new_shape)
    for band, band_values in bands.items():
        band_values[band_values > 100.] = 100.
        band_values[band_values < 0.] = 0.
        band_values = band_values / 100.
        band_values = band_values ** (1 / gamma)
        bands[band] = band_values
    rgb_array = np.dstack((
        bands['r'],
        bands['g'],
        bands['b']
    ))
    return rgb_array


def natural_colour_rgb(ahi_scn, apply_mask=False):
    gamma = 2.
    bands = {
        'r': np.array(ahi_scn['B05']),
        'g': downsample_array(np.array(ahi_scn['B04']), 2)[0],
        'b': downsample_array(np.array(ahi_scn['B03']), 4)[0],
    }
    if apply_mask:
        band_values = bands['r']
        mask = np.full(band_values.shape, False)
        min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
        min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        for band, band_values in bands.items():
            bands[band] = band_values[mask].reshape(new_shape)
    for band, band_values in bands.items():
        band_values[band_values > 100.] = 100.
        band_values[band_values < 0.] = 0.
        band_values = band_values / 100.
        band_values = band_values ** (1 / gamma)
        bands[band] = band_values
    rgb_array = np.dstack((
        bands['r'],
        bands['g'],
        bands['b']
    ))
    return rgb_array


def eumetsat_dust_rgb(ahi_scn, apply_mask=False):
    bands = {
        'r': np.array(ahi_scn['B15']),
        'g': np.array(ahi_scn['B13']),
        'b': np.array(ahi_scn['B11'])
    }
    if apply_mask:
        band_values = bands['r']
        mask = np.full(band_values.shape, False)
        min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
        min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        for band, band_values in bands.items():
            bands[band] = band_values[mask].reshape(new_shape)
    # Make r = limited B15 - B13 #
    pseudo_r = bands['r'] - bands['g']
    pseudo_r[pseudo_r > 2.] = 2.
    pseudo_r[pseudo_r < -4.] = -4.
    pseudo_r = (pseudo_r + 4.) / 6.
    # Make g = limited B13 - B11 #
    pseudo_g = bands['g'] - bands['b']
    pseudo_g[pseudo_g > 15.] = 15.
    pseudo_g[pseudo_g < 0.] = 0.
    pseudo_g = pseudo_g / 15.
    pseudo_g = pseudo_g ** (1 / 2.5)
    # Make b = limited B11 ###
    pseudo_b = bands['b']
    pseudo_b[pseudo_b > 289.] = 289.
    pseudo_b[pseudo_b < 261.] = 261.
    pseudo_b = (pseudo_b - 261.) / (289. - 261.)
    # Correct rgb arrays in dict #
    bands['r'] = pseudo_r
    bands['g'] = pseudo_g
    bands['b'] = pseudo_b
    rgb_array = np.dstack((
        bands['r'],
        bands['g'],
        bands['b']
    ))
    return rgb_array


def plot_rgb(ahi_scn, rgb_name='true_colour', apply_mask=False):
    available_rgbs = {  # 'RGB Name': ['Red', 'Green', 'Blue']
        'true_colour': ['R: ' + r'$0.645 \mu m$', 'G: ' + r'$0.510 \mu m$', 'B: ' + r'$0.455 \mu m$'],
        'EUMetSat_dust': ['R: ' + r'$12.35 \mu m$' + ' - ' + r'$10.45 \mu m$',
                          'G: ' + r'$10.45 \mu m$' + ' - ' + r'$8.60 \mu m$', 'B: ' + r'$10.45 \mu m$'],
        'natural_colour': ['R: ' + r'$1.61 \mu m$', 'G: ' + r'$0.860 \mu m$', 'B: ' + r'$0.645 \mu m$']
    }
    rgb_array = None
    if rgb_name not in available_rgbs.keys():
        raise Exception(
            f'Given rgb_name {rgb_name} is not available. \n' +
            f'Available RGBs are: \n' +
            '\n'.join([
                '\t' + key + ': ' + str(item)
                for key, item
                in available_rgbs.items()
            ])
        )
    elif rgb_name == 'true_colour':
        rgb_array = true_colour_rgb(
            ahi_scn=ahi_scn,
            apply_mask=apply_mask
        )
    elif rgb_name == 'EUMetSat_dust':
        rgb_array = eumetsat_dust_rgb(
            ahi_scn=ahi_scn,
            apply_mask=apply_mask
        )
    elif rgb_name == 'natural_colour':
        rgb_array = natural_colour_rgb(
            ahi_scn=ahi_scn,
            apply_mask=apply_mask
        )
    if apply_mask:
        fsize = (15, 10)
        xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
        ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
    else:
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    im = ax.imshow(
        rgb_array,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax)
    )
    ax.set_title(
        ' '.join([
            a[0].upper() + a[1:]
            if len(a) > 2
            else a.upper()
            if len(a) == 1
            else a
            for a
            in rgb_name.split('_')
        ]) + ' RGB',
        fontweight='bold',
        loc='left',
        fontsize=12
    )
    string_time = ahi_scn.start_time.strftime('%d %B %Y %H:%M UTC')
    ax.set_title(
        string_time,
        loc='right',
        fontsize=12
    )
    ax.text(
        0.5,
        -0.05,
        ' | '.join(available_rgbs[rgb_name]),
        size=12,
        ha='center',
        transform=ax.transAxes
    )
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    cbaxes.axis('off')
    return fig


@click.command()
@click.option(
    '--path_to_nc_file', '-pnc',
    default=os.path.join(main_dir, 'Example', 'ahi_nn_analysis_20200105_0500.nc'),
    help='The full/path/to/nc_file, where the .nc file has the format ' +
         'ahi_nn_annalysis_<YYYYmmdd>_<HHMM>.nc'
)
@click.option(
    '--save_dir', '-s',
    default=os.path.join(main_dir, 'Example', 'Images'),
    help='The directory where the images will be saved.'
)
@click.option(
    '--aus_only', '-a',
    default='True',
    help='Boolean indicating whether to produce graphs covering Australia only.'
)
@click.option(
    '--with_rgbs_and_channels', '-w',
    default='False',
    help='Boolean indicating whether to produce RGB images of the specified region.'
)
@click.option(
    '--path_to_scene', '-ps',
    default=os.path.join(main_dir, 'Example', '20200105_0500'),
    help='The full/path/to/scene, including the name of the folder ' +
         'that contains the scene data to be analysed. The scene folder ' +
         'has the format YYYYmmdd_HHMM.'
)
def main(path_to_nc_file, save_dir, aus_only, with_rgbs_and_channels, path_to_scene):
    if aus_only not in ['True', 'true', '1', 'yes', 'Yes']:
        plot_full_analysis(
            path_to_nc_file,
            save_dir
        )
    else:
        plot_full_analysis(
            path_to_nc_file,
            save_dir,
            aus_only=True
        )
    if with_rgbs_and_channels in ['True', 'true', '1', 'yes', 'Yes']:
        ahi_scn = read_h8_folder(path_to_scene)
        scene_stringtime = ahi_scn.start_time.strftime('%Y%m%d_%H%M')
        for rgb_name in ['true_colour', 'natural_colour', 'EUMetSat_dust']:
            if aus_only not in ['True', 'true', '1', 'yes', 'Yes']:
                fig = plot_rgb(
                    ahi_scn=ahi_scn,
                    rgb_name=rgb_name,
                    apply_mask=False
                )
                fig_name = '_'.join([rgb_name, scene_stringtime]) + '.png'
            else:
                fig = plot_rgb(
                    ahi_scn=ahi_scn,
                    rgb_name=rgb_name,
                    apply_mask=True
                )
                fig_name = '_'.join([rgb_name, scene_stringtime, 'australia_only']) + '.png'
            fig.savefig(
                os.path.join(save_dir, fig_name),
                bbox_inches='tight'
            )
        for bnum in range(1, 16 + 1):
            if aus_only not in ['True', 'true', '1', 'yes', 'Yes']:
                fig = plot_single_channel(
                    ahi_scn=ahi_scn,
                    band_number=bnum,
                    apply_mask=False
                )
                fig_name = '_'.join([f"B{int(bnum):02d}", scene_stringtime]) + '.png'
            else:
                fig = plot_single_channel(
                    ahi_scn=ahi_scn,
                    band_number=bnum,
                    apply_mask=True
                )
                fig_name = '_'.join([f"B{int(bnum):02d}", scene_stringtime, 'australia_only']) + '.png'
            fig.savefig(
                os.path.join(save_dir, fig_name),
                bbox_inches='tight'
            )


if __name__ == '__main__':
    main()
