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
from matplotlib.colors import LinearSegmentedColormap
main_dir = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
sys.path.append(main_dir)
from Processing.preprocessor import read_h8_folder, downsample_array

plt.rcParams.update({'font.size': 22})


def plot_nn_output(dataset, variable_name, dataset_datetime, apply_mask=False, mask_area='Australia'):
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
    if 'binary' in variable_name:
        colors = [
            (0., 0., 0.8),
            (0.8, 0., 0.)
        ]
        cmap = LinearSegmentedColormap.from_list('nn_rgb', colors, N=2)
    else:
        cmap = plt.get_cmap('bwr')
    cmap.set_bad(color='k', alpha=1.)
    # if variable_name == 'cloud_binary_mask':
    #     v_data = np.array(dataset['cloud_continuous_mask'])
    #     v_data[v_data == dataset[variable_name]._FillValue] = np.nan
    #     mask = v_data > 0.8
    #     v_data[mask] = 1.
    #     v_data[~mask] = 0.
    # else:
    #     v_data = np.array(dataset[variable_name])
    #     v_data[v_data == dataset[variable_name]._FillValue] = np.nan
    v_data = np.array(dataset[variable_name])
    v_data[v_data == dataset[variable_name]._FillValue] = np.nan
    if 'cloud' in variable_name and 'mask' not in variable_name:
        cloud_mask = np.array(dataset['cloud_binary_mask']) == 1
        v_data[~cloud_mask] = np.nan
    if 'aerosol' in variable_name:
        cloud_mask = np.array(dataset['cloud_binary_mask']) == 1
        if 'mask' in variable_name:
            v_data[cloud_mask] = np.nan
        else:
            aerosol_mask = np.array(dataset['aerosol_binary_mask']) == 1
            v_data[cloud_mask & ~aerosol_mask] = np.nan
    if apply_mask:
        if mask_area == 'Australia':
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
            mask = np.full(v_data.shape, False)
            min_y, max_y = int(6500 / 11000 * v_data.shape[0]), int(9750 / 11000 * v_data.shape[0])
            min_x, max_x = int(2900 / 11000 * v_data.shape[0]), int(8250 / 11000 * v_data.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            v_data = v_data[mask].reshape(new_shape)
        elif mask_area == 'Cape York':
            fsize = (10, 10)
            xmin, xmax = 5000 * 1000 - 5500000, 6000 * 1000 - 5500000
            ymin, ymax = 5500000 - 7500 * 1000, 5500000 - 6500 * 1000
            mask = np.full(v_data.shape, False)
            min_y, max_y = int(6500 / 11000 * v_data.shape[0]), int(7500 / 11000 * v_data.shape[0])
            min_x, max_x = int(5000 / 11000 * v_data.shape[0]), int(6000 / 11000 * v_data.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            v_data = v_data[mask].reshape(new_shape)
        elif mask_area == 'SE Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 5750 * 1000 - 5500000, 6750 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 8750 * 1000
            mask = np.full(v_data.shape, False)
            min_y, max_y = int(8750 / 11000 * v_data.shape[0]), int(9750 / 11000 * v_data.shape[0])
            min_x, max_x = int(5750 / 11000 * v_data.shape[0]), int(6750 / 11000 * v_data.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            v_data = v_data[mask].reshape(new_shape)
        elif mask_area == 'East Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 6500 * 1000 - 5500000, 7500 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(v_data.shape, False)
            min_y, max_y = int(7750 / 11000 * v_data.shape[0]), int(8750 / 11000 * v_data.shape[0])
            min_x, max_x = int(6500 / 11000 * v_data.shape[0]), int(7500 / 11000 * v_data.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            v_data = v_data[mask].reshape(new_shape)
        elif mask_area == 'Central Australia':
            fsize = (10, 10)
            xmin, xmax = 4750 * 1000 - 5500000, 5750 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(v_data.shape, False)
            min_y, max_y = int(7750 / 11000 * v_data.shape[0]), int(8750 / 11000 * v_data.shape[0])
            min_x, max_x = int(4750 / 11000 * v_data.shape[0]), int(5750 / 11000 * v_data.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            v_data = v_data[mask].reshape(new_shape)
        elif mask_area == 'China':
            fsize = (10, 10)
            xmin, xmax = 750, 4750 
            ymin, ymax = 4000, 0
            mask = np.full(v_data.shape, False)
            min_y, max_y = int(ymax / 11000 * v_data.shape[0]), int(ymin / 11000 * v_data.shape[0])
            min_x, max_x = int(xmin / 11000 * v_data.shape[0]), int(xmax / 11000 * v_data.shape[0])
            xmin, xmax = xmin * 1000 - 5500000, xmax * 1000 - 5500000
            ymin, ymax = 5500000 - ymin * 1000, 5500000 - ymax * 1000
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            v_data = v_data[mask].reshape(new_shape)
    else:
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='black')
    im = ax.imshow(
        v_data,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        vmin=dataset[variable_name].valid_min,
        vmax=dataset[variable_name].valid_max if dataset[variable_name].valid_max < 31 else 10
    )
    gl = ax.gridlines(
        draw_labels=True,
        color='black',
        linestyle='--',
        alpha=0.7
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'rotation': 0}
    gl.ylabel_style = {'rotation': 0}
    ax.set_title(
        'NN ' + ' '.join([
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
        fontsize=22
    )
    string_time = dataset_datetime.strftime('%d %B %Y %H:%M UTC')
    ax.set_title(
        string_time,
        loc='right',
        fontsize=22
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
    vlabel = ' '.join([
        a[0].upper() + a[1:]
        if len(a) > 2
        else a.upper()
        if len(a) == 1
        else a
        for a
        in variable_name.split('_')
    ])
    if dataset[variable_name].units != 'None':
        cbar_label = r'%s [%s]' % (vlabel, dataset[variable_name].units)
    else:
        cbar_label = None
    cbar = plt.colorbar(
        im,
        cax=cbaxes,
        orientation='vertical',
        label=cbar_label
    )
    if 'binary' in variable_name and 'cloud' in variable_name:
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels([
            '\nNon-Cloud',
            '\nCloud'
            ],
            rotation=90,
            va='center',
            ha='center'
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
    fname_stringtime = os.path.basename(path_to_dataset)
    fname_stringtime = '_'.join(fname_stringtime.split('_')[-3:-1])
    dst = nc.Dataset(path_to_dataset)
    plottables = [
        vname
        for vname
        in dst.variables.keys()
        if vname not in ['longitudes', 'latitudes']
    ]
    for plottable in plottables:
        print(plottable)
        fig_name = '_'.join([plottable, fname_stringtime])
        if aus_only:
            for region in ['Australia','Cape York','SE Coast of Australia','East Coast of Australia','Central Australia', 'China']:
                fig = plot_nn_output(
                    dataset=dst,
                    variable_name=plottable,
                    dataset_datetime=dt.strptime(fname_stringtime, '%Y%m%d_%H%M'),
                    apply_mask=aus_only,
                    mask_area=region
                )
                region_fig_name = '_'.join([fig_name, f'{"-".join(region.split(" "))}_only'])
                region_fig_name = region_fig_name + '.png'
                plt.savefig(
                    os.path.join(save_dir, region_fig_name),
                    format='png',
                    bbox_inches='tight',
                    dpi=300
                )
                plt.clf()
        else:
            fig = plot_nn_output(
                dataset=dst,
                variable_name=plottable,
                dataset_datetime=dt.strptime(fname_stringtime, '%Y%m%d_%H%M'),
                apply_mask=aus_only
            )
            fig_name = fig_name + '.png'
            plt.savefig(
                os.path.join(save_dir, fig_name),
                format='png',
                bbox_inches='tight',
                dpi=300
            )
    return None


def plot_single_channel(ahi_scn, band_number=3, apply_mask=False, mask_area='Australia'):
    """
    Plots a single channel's data

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
        if mask_area == 'Australia':
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'Cape York':
            fsize = (10, 10)
            xmin, xmax = 5000 * 1000 - 5500000, 6000 * 1000 - 5500000
            ymin, ymax = 5500000 - 7500 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            min_x, max_x = int(5000 / 11000 * band_values.shape[0]), int(6000 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'SE Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 5750 * 1000 - 5500000, 6750 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 8750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(8750 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(5750 / 11000 * band_values.shape[0]), int(6750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'East Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 6500 * 1000 - 5500000, 7500 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'Central Australia':
            fsize = (10, 10)
            xmin, xmax = 4750 * 1000 - 5500000, 5750 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(4750 / 11000 * band_values.shape[0]), int(5750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'China':
            fsize = (10, 10)
            xmin, xmax = 750, 4750 
            ymin, ymax = 4000, 0
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(ymax / 11000 * band_values.shape[0]), int(ymin / 11000 * band_values.shape[0])
            min_x, max_x = int(xmin / 11000 * band_values.shape[0]), int(xmax / 11000 * band_values.shape[0])
            xmin, xmax = xmin * 1000 - 5500000, xmax * 1000 - 5500000
            ymin, ymax = 5500000 - ymin * 1000, 5500000 - ymax * 1000
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        if band_values.shape != new_shape:
            band_values = band_values.reshape(new_shape)
    else:
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    low_perc = np.nanpercentile(band_values.flatten(), 5)
    high_perc = np.nanpercentile(band_values.flatten(), 95)
    im = ax.imshow(
        band_values,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap='magma',
        vmin=low_perc,
        vmax=high_perc
    )
    ax.set_title(
        f'Band {str(int(band_number))}',
        fontweight='bold',
        loc='left',
        fontsize=22
    )
    string_time = ahi_scn.start_time.strftime('%d %B %Y %H:%M UTC')
    ax.set_title(
        string_time,
        loc='right',
        fontsize=22
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
        extend='both',
        orientation='vertical',
        label=r'%s [%s]' % ('Reflectance' if int(band_number) < 7 else 'BT', '%' if int(band_number) < 7 else 'K')
    )
    return fig


def plot_channel_difference(ahi_scn, primary_band_number=14, secondary_band_number=15, apply_mask=False, mask_area='Australia'):
    """
    Plots the difference between 2 channels:
        primary_band - secondary_band

    :param ahi_scn:
    :param primary_band_number:
    :param secondary_band_number:
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
    primary_band_indicator = f"B{int(primary_band_number):02d}"
    primary_band = np.array(ahi_scn[primary_band_indicator])
    secondary_band_indicator = f"B{int(secondary_band_number):02d}"
    secondary_band = np.array(ahi_scn[secondary_band_indicator])
    if primary_band.shape != secondary_band.shape:
        primary_to_secondary = primary_band.shape[0] / secondary_band.shape[0]
        if primary_to_secondary < 1.:
            secondary_band = downsample_array(secondary_band, int(1/primary_to_secondary))[0]
        else:
            primary_band = downsample_array(primary_band, int(primary_to_secondary))[0]
    band_values = primary_band - secondary_band
    if apply_mask:
        if mask_area == 'Australia':
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'Cape York':
            fsize = (10, 10)
            xmin, xmax = 5000 * 1000 - 5500000, 6000 * 1000 - 5500000
            ymin, ymax = 5500000 - 7500 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            min_x, max_x = int(5000 / 11000 * band_values.shape[0]), int(6000 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'SE Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 5750 * 1000 - 5500000, 6750 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 8750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(8750 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(5750 / 11000 * band_values.shape[0]), int(6750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'East Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 6500 * 1000 - 5500000, 7500 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'Central Australia':
            fsize = (10, 10)
            xmin, xmax = 4750 * 1000 - 5500000, 5750 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(4750 / 11000 * band_values.shape[0]), int(5750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
            band_values = band_values[mask].reshape(new_shape)
        elif mask_area == 'China':
            fsize = (10, 10)
            xmin, xmax = 750, 4750 
            ymin, ymax = 4000, 0
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(ymax / 11000 * band_values.shape[0]), int(ymin / 11000 * band_values.shape[0])
            min_x, max_x = int(xmin / 11000 * band_values.shape[0]), int(xmax / 11000 * band_values.shape[0])
            xmin, xmax = xmin * 1000 - 5500000, xmax * 1000 - 5500000
            ymin, ymax = 5500000 - ymin * 1000, 5500000 - ymax * 1000
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        if band_values.shape != new_shape:
            band_values = band_values.reshape(new_shape)
    else:
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    mod_max = max([abs(np.nanmin(band_values)), abs(np.nanmax(band_values))])
    im = ax.imshow(
        band_values,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap='bwr',
        vmin=-mod_max,
        vmax=mod_max
    )
    ax.set_title(
        f'Band Difference: {str(int(primary_band_number))} - {str(int(secondary_band_number))}',
        fontweight='bold',
        loc='left',
        fontsize=22
    )
    string_time = ahi_scn.start_time.strftime('%d %B %Y %H:%M UTC')
    ax.set_title(
        string_time,
        loc='right',
        fontsize=22
    )
    ax.text(
        0.5,
        -0.05,
        'Band Central Wavelengths: ' + \
        str(band_wavelengths[int(int(primary_band_number) - 1)]) + r'$\mu m$' + \
        ' - ' + str(band_wavelengths[int(int(secondary_band_number) - 1)]) + r'$\mu m$',
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
        label=r'%s [%s]' % ('Difference in Reflectance' if int(primary_band_number) < 7 else 'Difference in BT', '%' if int(primary_band_number) < 7 else 'K')
    )
    return fig


def true_colour_rgb(ahi_scn, apply_mask=False, mask_area='Australia'):
    gamma = 2.
    bands = {
        'r': downsample_array(np.array(ahi_scn['B03']), 4)[0],
        'g': downsample_array(np.array(ahi_scn['B02']), 2)[0],
        'b': downsample_array(np.array(ahi_scn['B01']), 2)[0],
    }
    if apply_mask:
        band_values = bands['r']
        if mask_area == 'Australia':
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'Cape York':
            fsize = (10, 10)
            xmin, xmax = 5000 * 1000 - 5500000, 6000 * 1000 - 5500000
            ymin, ymax = 5500000 - 7500 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            min_x, max_x = int(5000 / 11000 * band_values.shape[0]), int(6000 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'SE Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 5750 * 1000 - 5500000, 6750 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 8750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(8750 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(5750 / 11000 * band_values.shape[0]), int(6750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'East Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 6500 * 1000 - 5500000, 7500 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'Central Australia':
            fsize = (10, 10)
            xmin, xmax = 4750 * 1000 - 5500000, 5750 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(4750 / 11000 * band_values.shape[0]), int(5750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'China':
            fsize = (10, 10)
            xmin, xmax = 750, 4750 
            ymin, ymax = 4000, 0
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(ymax / 11000 * band_values.shape[0]), int(ymin / 11000 * band_values.shape[0])
            min_x, max_x = int(xmin / 11000 * band_values.shape[0]), int(xmax / 11000 * band_values.shape[0])
            xmin, xmax = xmin * 1000 - 5500000, xmax * 1000 - 5500000
            ymin, ymax = 5500000 - ymin * 1000, 5500000 - ymax * 1000
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


def natural_colour_rgb(ahi_scn, apply_mask=False, mask_area='Australia'):
    gamma = 2.
    bands = {
        'r': np.array(ahi_scn['B05']),
        'g': downsample_array(np.array(ahi_scn['B04']), 2)[0],
        'b': downsample_array(np.array(ahi_scn['B03']), 4)[0],
    }
    if apply_mask:
        band_values = bands['r']
        if mask_area == 'Australia':
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'Cape York':
            fsize = (10, 10)
            xmin, xmax = 5000 * 1000 - 5500000, 6000 * 1000 - 5500000
            ymin, ymax = 5500000 - 7500 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            min_x, max_x = int(5000 / 11000 * band_values.shape[0]), int(6000 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'SE Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 5750 * 1000 - 5500000, 6750 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 8750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(8750 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(5750 / 11000 * band_values.shape[0]), int(6750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'East Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 6500 * 1000 - 5500000, 7500 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'Central Australia':
            fsize = (10, 10)
            xmin, xmax = 4750 * 1000 - 5500000, 5750 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(4750 / 11000 * band_values.shape[0]), int(5750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'China':
            fsize = (10, 10)
            xmin, xmax = 750, 4750 
            ymin, ymax = 4000, 0
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(ymax / 11000 * band_values.shape[0]), int(ymin / 11000 * band_values.shape[0])
            min_x, max_x = int(xmin / 11000 * band_values.shape[0]), int(xmax / 11000 * band_values.shape[0])
            xmin, xmax = xmin * 1000 - 5500000, xmax * 1000 - 5500000
            ymin, ymax = 5500000 - ymin * 1000, 5500000 - ymax * 1000
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


def eumetsat_dust_rgb(ahi_scn, apply_mask=False, mask_area='Australia'):
    bands = {
        'r': np.array(ahi_scn['B15']),
        'g': np.array(ahi_scn['B13']),
        'b': np.array(ahi_scn['B11'])
    }
    if apply_mask:
        band_values = bands['r']
        if mask_area == 'Australia':
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(2900 / 11000 * band_values.shape[0]), int(8250 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'Cape York':
            fsize = (10, 10)
            xmin, xmax = 5000 * 1000 - 5500000, 6000 * 1000 - 5500000
            ymin, ymax = 5500000 - 7500 * 1000, 5500000 - 6500 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            min_x, max_x = int(5000 / 11000 * band_values.shape[0]), int(6000 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'SE Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 5750 * 1000 - 5500000, 6750 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 8750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(8750 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
            min_x, max_x = int(5750 / 11000 * band_values.shape[0]), int(6750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'East Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 6500 * 1000 - 5500000, 7500 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(6500 / 11000 * band_values.shape[0]), int(7500 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'Central Australia':
            fsize = (10, 10)
            xmin, xmax = 4750 * 1000 - 5500000, 5750 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(7750 / 11000 * band_values.shape[0]), int(8750 / 11000 * band_values.shape[0])
            min_x, max_x = int(4750 / 11000 * band_values.shape[0]), int(5750 / 11000 * band_values.shape[0])
            mask[min_y:max_y, min_x:max_x] = True
            new_shape = (int(max_y - min_y), int(max_x - min_x))
        elif mask_area == 'China':
            fsize = (10, 10)
            xmin, xmax = 750, 4750 
            ymin, ymax = 4000, 0
            mask = np.full(band_values.shape, False)
            min_y, max_y = int(ymax / 11000 * band_values.shape[0]), int(ymin / 11000 * band_values.shape[0])
            min_x, max_x = int(xmin / 11000 * band_values.shape[0]), int(xmax / 11000 * band_values.shape[0])
            xmin, xmax = xmin * 1000 - 5500000, xmax * 1000 - 5500000
            ymin, ymax = 5500000 - ymin * 1000, 5500000 - ymax * 1000
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


def plot_rgb(ahi_scn, rgb_name='true_colour', apply_mask=False, mask_area='Australia'):
    available_rgbs = {  # 'RGB Name': ['Red', 'Green', 'Blue']
        'true_colour': [
            'R: ' + r'$0.64 \mu m$',
            'G: ' + r'$0.51 \mu m$',
            'B: ' + r'$0.47 \mu m$'
        ],
        'EUMetSat_dust': [
            'R: ' + r'$12.4 \mu m$' + ' - ' + r'$10.4 \mu m$',
            'G: ' + r'$10.4 \mu m$' + ' - ' + r'$8.6 \mu m$', 
            'B: ' + r'$10.4 \mu m$'
        ],
        'natural_colour': [
            'R: ' + r'$1.6 \mu m$', 
            'G: ' + r'$0.86 \mu m$',
            'B: ' + r'$0.64 \mu m$'
        ]
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
            apply_mask=apply_mask,
            mask_area=mask_area
        )
    elif rgb_name == 'EUMetSat_dust':
        rgb_array = eumetsat_dust_rgb(
            ahi_scn=ahi_scn,
            apply_mask=apply_mask,
            mask_area=mask_area
        )
    elif rgb_name == 'natural_colour':
        rgb_array = natural_colour_rgb(
            ahi_scn=ahi_scn,
            apply_mask=apply_mask,
            mask_area=mask_area
        )
    if apply_mask:
        if mask_area == 'Australia':
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
        elif mask_area == 'Cape York':
            fsize = (10, 10)
            xmin, xmax = 5000 * 1000 - 5500000, 6000 * 1000 - 5500000
            ymin, ymax = 5500000 - 7500 * 1000, 5500000 - 6500 * 1000
        elif mask_area == 'SE Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 5750 * 1000 - 5500000, 6750 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 8750 * 1000
        elif mask_area == 'East Coast of Australia':
            fsize = (10, 10)
            xmin, xmax = 6500 * 1000 - 5500000, 7500 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
        elif mask_area == 'Central Australia':
            fsize = (10, 10)
            xmin, xmax = 4750 * 1000 - 5500000, 5750 * 1000 - 5500000
            ymin, ymax = 5500000 - 8750 * 1000, 5500000 - 7750 * 1000
        elif mask_area == 'China':
            fsize = (10, 10)
            xmin, xmax = 750 * 1000 - 5500000, 4750 * 1000 - 5500000
            ymin, ymax = 5500000 - 4000 * 1000, 5500000 - 0 * 1000
    else:
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='black')
    im = ax.imshow(
        rgb_array,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax)
    )
    gl = ax.gridlines(
        draw_labels=True,
        color='black',
        linestyle='--',
        alpha=0.7
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'rotation': 0}
    gl.ylabel_style = {'rotation': 0}
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
        fontsize=22
    )
    string_time = ahi_scn.start_time.strftime('%d %B %Y %H:%M UTC')
    ax.set_title(
        string_time,
        loc='right',
        fontsize=22
    )
    ax.text(
        0.5,
        -0.05,
        ' | '.join(available_rgbs[rgb_name]),
        fontsize=22,
        ha='center',
        transform=ax.transAxes
    )
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    cbaxes.axis('off')
    return fig


def plot_smoke_analysis(ahi_scn, dataset):
    """
    Plots a 6 panel figure loking at the SE coast of Australia.
    """
    fig, axs = plt.subplots(
        2, 
        3, 
        subplot_kw={'projection': ccrs.Geostationary(140.735785863)},
        figsize=(16,9)
    )
    string_time = ahi_scn.start_time.strftime('%d %B %Y %H:%M UTC')
    fig.suptitle(string_time)
    mask_area = 'SE Coast of Australia'
    xmin, xmax = 5750 * 1000 - 5500000, 6750 * 1000 - 5500000
    ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 8750 * 1000
    # Top left panel
    variable_name = 'cloud_continuous_mask'
    cmap = plt.get_cmap('bwr')
    cmap.set_bad(color='k', alpha=1.)
    v_data = np.array(dataset[variable_name])
    v_data[v_data == dataset[variable_name]._FillValue] = np.nan
    mask = np.full(v_data.shape, False)
    min_y, max_y = int(8750 / 11000 * v_data.shape[0]), int(9750 / 11000 * v_data.shape[0])
    min_x, max_x = int(5750 / 11000 * v_data.shape[0]), int(6750 / 11000 * v_data.shape[0])
    mask[min_y:max_y, min_x:max_x] = True
    new_shape = (int(max_y - min_y), int(max_x - min_x))
    v_data = v_data[mask].reshape(new_shape)
    im1 = axs[0,0].imshow(
        v_data,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        vmin=dataset[variable_name].valid_min,
        vmax=dataset[variable_name].valid_max
    )
    axs[0,0].coastlines()
    divider = make_axes_locatable(axs[0,0])
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    vlabel = ' '.join([
        a[0].upper() + a[1:]
        if len(a) > 2
        else a.upper()
        if len(a) == 1
        else a
        for a
        in variable_name.split('_')
    ])
    cbar1 = plt.colorbar(
        im1,
        cax=cbaxes,
        orientation='vertical',
        label=r'%s [%s]' % (vlabel, dataset[variable_name].units)
    )
    # Top centre panel
    variable_name = 'cloud_binary_mask'
    colors = [
        (0., 0., 1.),
        (1., 0., 0.)
    ]
    cmap = LinearSegmentedColormap.from_list('nn_rgb', colors, N=2)
    cmap.set_bad(color='k', alpha=1.)
    v_data = np.array(dataset[variable_name])
    v_data[v_data == dataset[variable_name]._FillValue] = np.nan
    mask = np.full(v_data.shape, False)
    min_y, max_y = int(8750 / 11000 * v_data.shape[0]), int(9750 / 11000 * v_data.shape[0])
    min_x, max_x = int(5750 / 11000 * v_data.shape[0]), int(6750 / 11000 * v_data.shape[0])
    mask[min_y:max_y, min_x:max_x] = True
    new_shape = (int(max_y - min_y), int(max_x - min_x))
    v_data = v_data[mask].reshape(new_shape)
    im2 = axs[0,1].imshow(
        v_data,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        vmin=dataset[variable_name].valid_min,
        vmax=dataset[variable_name].valid_max
    )
    axs[0,1].coastlines()
    divider = make_axes_locatable(axs[0,1])
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    vlabel = ' '.join([
        a[0].upper() + a[1:]
        if len(a) > 2
        else a.upper()
        if len(a) == 1
        else a
        for a
        in variable_name.split('_')
    ])
    cbar2 = plt.colorbar(
        im2,
        cax=cbaxes,
        orientation='vertical',
        label=r'%s [%s]' % (vlabel, dataset[variable_name].units)
    )
    cbar2.set_ticks([0.25, 0.75])
    cbar2.ax.set_yticklabels([
        'Non-Cloud',
        'Cloud'
        ],
        rotation=90,
        verticalalignment='center'
    )
    # Top right panel
    band_values = np.array(ahi_scn['B13'])
    mask = np.full(band_values.shape, False)
    min_y, max_y = int(8750 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
    min_x, max_x = int(5750 / 11000 * band_values.shape[0]), int(6750 / 11000 * band_values.shape[0])
    mask[min_y:max_y, min_x:max_x] = True
    new_shape = (int(max_y - min_y), int(max_x - min_x))
    band_values = band_values[mask].reshape(new_shape)
    im3 = axs[0,2].imshow(
        band_values,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap='binary',
        vmin=250,
        vmax=300
    )
    axs[0,2].imshow(
        v_data,
        alpha=0.5*v_data,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap='bwr'
    )
    axs[0,2].coastlines()
    divider = make_axes_locatable(axs[0,2])
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    cbar3 = plt.colorbar(
        im3,
        extend='both',
        cax=cbaxes,
        orientation='vertical',
        label='%s [%s]' % (r'10.45$\mu m$ BT', 'K')
    )
    # Mid left panel
    img_arr = natural_colour_rgb(ahi_scn, apply_mask=True, mask_area=mask_area)
    axs[1,0].imshow(
        img_arr,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax)
    )
    axs[1,0].coastlines()
    divider = make_axes_locatable(axs[1,0])
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    cbaxes.axis('off')
    # Mid centre panel
    img_arr = eumetsat_dust_rgb(ahi_scn, apply_mask=True, mask_area=mask_area)
    axs[1,1].imshow(
        img_arr,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax)
    )
    axs[1,1].coastlines()
    divider = make_axes_locatable(axs[1,1])
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    cbaxes.axis('off')
    # Mid right panel
    img_arr = true_colour_rgb(ahi_scn, apply_mask=True, mask_area=mask_area)
    axs[1,2].imshow(
        img_arr,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax)
    )
    axs[1,2].coastlines()
    divider = make_axes_locatable(axs[1,2])
    cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    cbaxes.axis('off')
    # # Bottom left panel
    # variable_name = 'aerosol_continuous_mask'
    # cmap = plt.get_cmap('bwr')
    # cmap.set_bad(color='k', alpha=1.)
    # v_data = np.array(dataset[variable_name])
    # v_data[v_data == dataset[variable_name]._FillValue] = np.nan
    # mask = np.full(v_data.shape, False)
    # min_y, max_y = int(8750 / 11000 * v_data.shape[0]), int(9750 / 11000 * v_data.shape[0])
    # min_x, max_x = int(5750 / 11000 * v_data.shape[0]), int(6750 / 11000 * v_data.shape[0])
    # mask[min_y:max_y, min_x:max_x] = True
    # new_shape = (int(max_y - min_y), int(max_x - min_x))
    # v_data = v_data[mask].reshape(new_shape)
    # im1 = axs[2,0].imshow(
    #     v_data,
    #     origin='upper',
    #     transform=ccrs.Geostationary(140.735785863),
    #     extent=(xmin, xmax, ymin, ymax),
    #     cmap=cmap,
    #     vmin=dataset[variable_name].valid_min,
    #     vmax=dataset[variable_name].valid_max
    # )
    # axs[2,0].coastlines()
    # divider = make_axes_locatable(axs[2,0])
    # cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    # vlabel = ' '.join([
    #     a[0].upper() + a[1:]
    #     if len(a) > 2
    #     else a.upper()
    #     if len(a) == 1
    #     else a
    #     for a
    #     in variable_name.split('_')
    # ])
    # cbar1 = plt.colorbar(
    #     im1,
    #     cax=cbaxes,
    #     orientation='vertical',
    #     label=r'%s [%s]' % (vlabel, dataset[variable_name].units)
    # )
    # # Bottom centre panel
    # variable_name = 'aerosol_binary_mask'
    # colors = [
    #     (0., 0., 1.),
    #     (1., 0., 0.)
    # ]
    # cmap = LinearSegmentedColormap.from_list('nn_rgb', colors, N=2)
    # cmap.set_bad(color='k', alpha=1.)
    # v_data = np.array(dataset[variable_name])
    # v_data[v_data == dataset[variable_name]._FillValue] = np.nan
    # mask = np.full(v_data.shape, False)
    # min_y, max_y = int(8750 / 11000 * v_data.shape[0]), int(9750 / 11000 * v_data.shape[0])
    # min_x, max_x = int(5750 / 11000 * v_data.shape[0]), int(6750 / 11000 * v_data.shape[0])
    # mask[min_y:max_y, min_x:max_x] = True
    # new_shape = (int(max_y - min_y), int(max_x - min_x))
    # v_data = v_data[mask].reshape(new_shape)
    # im2 = axs[2,1].imshow(
    #     v_data,
    #     origin='upper',
    #     transform=ccrs.Geostationary(140.735785863),
    #     extent=(xmin, xmax, ymin, ymax),
    #     cmap=cmap,
    #     vmin=dataset[variable_name].valid_min,
    #     vmax=dataset[variable_name].valid_max
    # )
    # axs[2,1].coastlines()
    # divider = make_axes_locatable(axs[2,1])
    # cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    # vlabel = ' '.join([
    #     a[0].upper() + a[1:]
    #     if len(a) > 2
    #     else a.upper()
    #     if len(a) == 1
    #     else a
    #     for a
    #     in variable_name.split('_')
    # ])
    # cbar2 = plt.colorbar(
    #     im2,
    #     cax=cbaxes,
    #     orientation='vertical',
    #     label=r'%s [%s]' % (vlabel, dataset[variable_name].units)
    # )
    # cbar2.set_ticks([0.25, 0.75])
    # cbar2.ax.set_yticklabels([
    #     'Non Thick Aerosol',
    #     'Thick Aerosol'
    #     ],
    #     rotation=90,
    #     verticalalignment='center'
    # )
    # # Bottom right panel
    # band_values = np.array(ahi_scn['B13'])
    # mask = np.full(band_values.shape, False)
    # min_y, max_y = int(8750 / 11000 * band_values.shape[0]), int(9750 / 11000 * band_values.shape[0])
    # min_x, max_x = int(5750 / 11000 * band_values.shape[0]), int(6750 / 11000 * band_values.shape[0])
    # mask[min_y:max_y, min_x:max_x] = True
    # new_shape = (int(max_y - min_y), int(max_x - min_x))
    # band_values = band_values[mask].reshape(new_shape)
    # im3 = axs[2,2].imshow(
    #     band_values,
    #     origin='upper',
    #     transform=ccrs.Geostationary(140.735785863),
    #     extent=(xmin, xmax, ymin, ymax),
    #     cmap='binary',
    #     vmin=250,
    #     vmax=300
    # )
    # axs[2,2].imshow(
    #     v_data,
    #     alpha=0.5*v_data,
    #     origin='upper',
    #     transform=ccrs.Geostationary(140.735785863),
    #     extent=(xmin, xmax, ymin, ymax),
    #     cmap='bwr'
    # )
    # axs[2,2].coastlines()
    # divider = make_axes_locatable(axs[2,2])
    # cbaxes = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    # cbar3 = plt.colorbar(
    #     im3,
    #     extend='both',
    #     cax=cbaxes,
    #     orientation='vertical',
    #     label='%s [%s]' % (r'10.45$\mu m$ BT', 'K')
    # )
    fig.tight_layout()
    return fig


@click.command()
@click.option(
    '--path_to_nc_file', '-pnc',
    default=os.path.join(main_dir, 'Example', 'ahi_nn_analysis_20200105_0500_wo-era.nc'),
    help='The full/path/to/nc_file, where the .nc file has the format ' +
         'ahi_nn_annalysis_<YYYYmmdd>_<HHMM>_<w/wo-era>.nc'
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
@click.option(
    '--smoke_analysis_only', '-sao',
    default='False',
    help='Plots smoke analysis figure if True'
)
def main(path_to_nc_file, save_dir, aus_only, with_rgbs_and_channels, path_to_scene, smoke_analysis_only):
    if smoke_analysis_only in ['True', 'true', '1', 'yes', 'Yes']:
        dst = nc.Dataset(path_to_nc_file)
        ahi_scn = read_h8_folder(path_to_scene)
        scene_stringtime = ahi_scn.start_time.strftime('%Y%m%d_%H%M')
        fig = plot_smoke_analysis(
            ahi_scn=ahi_scn,
            dataset=dst
        )
        fig_name = '_'.join(['smoke_analysis_panels', scene_stringtime]) + '.png'
        fig.savefig(
            os.path.join(save_dir, fig_name),
            format='png',
            bbox_inches='tight'
        )
        plt.clf()
        plt.close(fig)
    else:
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
            print('Plotting RGBs...')
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
                    fig.savefig(
                            os.path.join(save_dir, fig_name),
                            format='png',
                            bbox_inches='tight',
                            dpi=300
                        )
                    plt.clf()
                    plt.close(fig)
                else:
                    for region in ['Australia','Cape York','SE Coast of Australia','East Coast of Australia','Central Australia', 'China']:
                        fig = plot_rgb(
                            ahi_scn=ahi_scn,
                            rgb_name=rgb_name,
                            apply_mask=True,
                            mask_area=region
                        )
                        fig_name = '_'.join([rgb_name, scene_stringtime, f'{"-".join(region.split(" "))}_only']) + '.png'
                        fig.savefig(
                            os.path.join(save_dir, fig_name),
                            format='png',
                            bbox_inches='tight',
                            dpi=300
                        )
                        plt.clf()
                        plt.close(fig)
            #             fig = plot_channel_difference(
            #                 ahi_scn=ahi_scn,
            #                 apply_mask=True,
            #                 mask_area=region
            #             )
            #             fig_name = '_'.join(['11-12micron_diff', scene_stringtime, f'{"-".join(region.split(" "))}_only']) + '.png'
            #             fig.savefig(
            #                 os.path.join(save_dir, fig_name),
            #                 format='png',
            #                 bbox_inches='tight'
            #             )
            #             plt.clf()
            #             plt.close(fig)
            #             fig = plot_channel_difference(
            #                 ahi_scn=ahi_scn,
            #                 primary_band_number=4,
            #                 secondary_band_number=5,
            #                 apply_mask=True,
            #                 mask_area=region
            #             )
            #             fig_name = '_'.join(['B04-B05_diff', scene_stringtime, f'{"-".join(region.split(" "))}_only']) + '.png'
            #             fig.savefig(
            #                 os.path.join(save_dir, fig_name),
            #                 format='png',
            #                 bbox_inches='tight'
            #             )
            #             plt.clf()
            #             plt.close(fig)
            # for bnum in range(1, 16 + 1):
            #     if aus_only not in ['True', 'true', '1', 'yes', 'Yes']:
            #         fig = plot_single_channel(
            #             ahi_scn=ahi_scn,
            #             band_number=bnum,
            #             apply_mask=False
            #         )
            #         fig_name = '_'.join([f"B{int(bnum):02d}", scene_stringtime]) + '.png'
            #         fig.savefig(
            #                 os.path.join(save_dir, fig_name),
            #                 format='png',
            #                 bbox_inches='tight'
            #             )
            #     else:
            #         for region in ['Australia','Cape York','SE Coast of Australia','East Coast of Australia','Central Australia']:
            #             fig = plot_single_channel(
            #                 ahi_scn=ahi_scn,
            #                 band_number=bnum,
            #                 apply_mask=True,
            #                 mask_area=region
            #             )
            #             fig_name = '_'.join([f"B{int(bnum):02d}", scene_stringtime, f'{"-".join(region.split(" "))}_only']) + '.png'
            #             fig.savefig(
            #                 os.path.join(save_dir, fig_name),
            #                 format='png',
            #                 bbox_inches='tight'
            #             )


if __name__ == '__main__':
    main()
