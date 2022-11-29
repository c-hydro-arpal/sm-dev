"""
Class Features

Name:          driver_data_io_static
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20200515'
Version:       '1.0.0'
"""

######################################################################################
# Library
import logging
import numpy as np
import os
from copy import deepcopy

import pandas as pd

from lib_data_io_ascii import read_file_raster, read_file_point
from lib_data_io_pickle import read_obj, write_obj

from lib_utils_geo import get_grid_value_from_xy, get_grid_idx_from_xy, get_idx_by_win
from lib_utils_system import fill_tags2string, make_folder

from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)

# Debug
# import matplotlib.pylab as plt
######################################################################################


# -------------------------------------------------------------------------------------
# Class DriverData
class DriverData:

    # -------------------------------------------------------------------------------------
    # Initialize class
    def __init__(self, src_dict, anc_dict, alg_dict, tmp_dict=None,
                 template_tags_dict=None,
                 flag_geo_data_grid='grid', flag_geo_data_point='point',
                 flag_data_updating=True):

        self.flag_geo_data_grid = flag_geo_data_grid
        self.flag_geo_data_point = flag_geo_data_point
        self.template_tags_dict = template_tags_dict

        self.src_dict_grid = src_dict[flag_geo_data_grid]
        self.src_dict_point = src_dict[flag_geo_data_point]
        self.anc_dict = anc_dict
        self.alg_dict = alg_dict
        self.tmp_dict = tmp_dict

        self.file_name_tag = 'file_name'
        self.folder_name_tag = 'folder_name'
        self.file_fields_tag = 'file_fields'

        self.grid_terrain_tag = 'terrain'
        self.grid_cn_tag = 'cn'
        self.grid_cnet_tag = 'channels_network'
        self.points_registry_tag = 'stations_registry'

        self.geo_x_tag = 'Longitude'
        self.geo_y_tag = 'Latitude'

        self.alg_catchment_name = alg_dict['catchment_name']
        self.alg_point_geo_method_search = alg_dict['geo_method_search']
        self.alg_point_geo_radius_influence = alg_dict['geo_radius_influence']
        self.alg_point_geo_neighbours = alg_dict['geo_neighbours']
        self.alg_point_geo_spatial_window = alg_dict['geo_spatial_window']

        # source object(s)
        folder_name_tmp = self.src_dict_grid[self.grid_terrain_tag][self.folder_name_tag]
        file_name_tmp = self.src_dict_grid[self.grid_terrain_tag][self.file_name_tag]
        file_path_tmp = os.path.join(folder_name_tmp, file_name_tmp)

        self.file_path_src_grid_terrain = fill_tags2string(
            file_path_tmp,
            tags_format=template_tags_dict, tags_filling={'catchment_name': self.alg_catchment_name})[0]

        folder_name_tmp = self.src_dict_grid[self.grid_cn_tag][self.folder_name_tag]
        file_name_tmp = self.src_dict_grid[self.grid_cn_tag][self.file_name_tag]
        file_path_tmp = os.path.join(folder_name_tmp, file_name_tmp)

        self.file_path_src_grid_cn = fill_tags2string(
            file_path_tmp,
            tags_format=template_tags_dict, tags_filling={'catchment_name': self.alg_catchment_name})[0]

        folder_name_tmp = self.src_dict_grid[self.grid_cnet_tag][self.folder_name_tag]
        file_name_tmp = self.src_dict_grid[self.grid_cnet_tag][self.file_name_tag]
        file_path_tmp = os.path.join(folder_name_tmp, file_name_tmp)

        self.file_path_src_grid_cnet = fill_tags2string(
            file_path_tmp,
            tags_format=template_tags_dict, tags_filling={'catchment_name': self.alg_catchment_name})[0]

        self.file_fields_longitude_grid_tag = 'grid_longitude'
        self.file_fields_latitude_grid_tag = 'grid_latitude'
        self.file_fields_distance_grid_tag = 'grid_distance'

        folder_name_tmp = self.src_dict_point[self.folder_name_tag]
        file_name_tmp = self.src_dict_point[self.file_name_tag]
        file_path_tmp = os.path.join(folder_name_tmp, file_name_tmp)

        self.file_path_src_point = fill_tags2string(
            file_path_tmp,
            tags_format=template_tags_dict, tags_filling={'catchment_name': self.alg_catchment_name})[0]

        self.file_fields_src_point = self.src_dict_point[self.file_fields_tag]
        self.file_fields_terrain_point_tag = 'point_terrain'
        self.file_fields_cn_point_tag = 'point_cn'
        self.file_fields_cnet_point_tag = 'point_cnet'
        self.file_fields_idx_1d_point_tag = 'point_idx_1d'
        self.file_fields_idx_2d_x_point_tag = 'point_idx_2d_x'
        self.file_fields_idx_2d_y_point_tag = 'point_idx_2d_y'

        # destination object(s)
        folder_name_tmp = self.anc_dict[self.folder_name_tag]
        file_name_tmp = self.anc_dict[self.file_name_tag]
        file_path_tmp = os.path.join(folder_name_tmp, file_name_tmp)

        self.file_path_anc = fill_tags2string(
            file_path_tmp,
            tags_format=template_tags_dict, tags_filling={'catchment_name': self.alg_catchment_name})[0]

        # tmp object(s)
        self.folder_name_tmp_raw = tmp_dict[self.folder_name_tag]
        self.file_name_tmp_raw = tmp_dict[self.file_name_tag]

        # flags for updating dataset(s)
        self.flag_data_updating = flag_data_updating

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to join geo obj
    def join_geo_obj(self, geo_grid_darray_terrain, geo_grid_darray_cn, geo_grid_darray_cnet, geo_point_dframe_in):

        log_stream.info(' -----> Join the point and the grid information ... ')

        geo_grid_x_1d = geo_grid_darray_terrain[self.geo_x_tag].values
        geo_grid_y_1d = geo_grid_darray_terrain[self.geo_y_tag].values
        geo_grid_x_2d, geo_grid_y_2d = np.meshgrid(geo_grid_x_1d, geo_grid_y_1d)

        geo_point_dframe_out = pd.DataFrame()
        for geo_point_id, geo_point_row_in in geo_point_dframe_in.iterrows():

            geo_point_name = geo_point_row_in['point_name']

            log_stream.info(' ------> Point "' + geo_point_name + '" ... ')

            geo_point_row_out = deepcopy(geo_point_row_in)

            # Get the center point
            geo_point_x_center = geo_point_row_in['point_longitude']
            geo_point_y_center = geo_point_row_in['point_latitude']

            geo_grid_value_terrain_center, geo_grid_x_center, geo_grid_y_center = get_grid_value_from_xy(
                geo_grid_darray_terrain, geo_point_x_center, geo_point_y_center,
                select_method=self.alg_point_geo_method_search)

            geo_idx_cont_center, geo_idx_x_center, geo_idx_y_center, geo_distance_center = get_grid_idx_from_xy(
                geo_grid_x_center, geo_grid_y_center, geo_grid_x_2d, geo_grid_y_2d,
                geo_radius_influence=self.alg_point_geo_radius_influence, geo_neighbours=self.alg_point_geo_neighbours)

            geo_points_idx_window, geo_points_xy_window = get_idx_by_win(
                geo_idx_x_center, geo_idx_y_center, geo_grid_x_2d, geo_grid_y_2d,
                geo_win_x=self.alg_point_geo_spatial_window, geo_win_y=self.alg_point_geo_spatial_window)

            # Iterate over a window around the center point
            geo_grid_list_dem, geo_grid_list_cn, geo_grid_list_cnet = [], [], []
            geo_grid_list_x, geo_grid_list_y = [], []
            geo_idx_list_cont, geo_idx_list_x, geo_idx_list_y, geo_distance_list = [], [], [], []
            for geo_point_idx_window, geo_point_xy_window in zip(geo_points_idx_window, geo_points_xy_window):

                geo_point_x_window, geo_point_y_window = geo_point_xy_window[0], geo_point_xy_window[1]

                geo_grid_value_terrain_window, geo_grid_x_window, geo_grid_y_window = get_grid_value_from_xy(
                    geo_grid_darray_terrain, geo_point_x_window, geo_point_y_window,
                    select_method=self.alg_point_geo_method_search)

                geo_idx_cont_window, geo_idx_x_window, geo_idx_y_window, geo_distance_window = get_grid_idx_from_xy(
                    geo_grid_x_window, geo_grid_y_window, geo_grid_x_2d, geo_grid_y_2d,
                    geo_radius_influence=self.alg_point_geo_radius_influence,
                    geo_neighbours=self.alg_point_geo_neighbours)

                geo_grid_value_cn, _, _ = get_grid_value_from_xy(
                    geo_grid_darray_cn, geo_point_x_window, geo_grid_y_window,
                    select_method=self.alg_point_geo_method_search)
                geo_grid_value_cnet, _, _ = get_grid_value_from_xy(
                    geo_grid_darray_cnet, geo_point_x_window, geo_grid_y_window,
                    select_method=self.alg_point_geo_method_search)

                geo_grid_list_dem.append(float(geo_grid_value_terrain_window))
                geo_grid_list_cn.append(float(geo_grid_value_cn))
                geo_grid_list_cnet.append(int(geo_grid_value_cnet))
                geo_idx_list_cont.append(int(geo_idx_cont_window))
                geo_idx_list_x.append(int(geo_idx_x_window))
                geo_idx_list_y.append(int(geo_idx_y_window))
                geo_distance_list.append(float(geo_distance_window))
                geo_grid_list_x.append(float(geo_grid_x_window))
                geo_grid_list_y.append(float(geo_grid_y_window))

            assert geo_point_row_in['point_longitude'] == geo_point_x_center, \
                'coordinate x of geo point "' + geo_point_name + '" is not the same found in the source file'
            assert geo_point_row_in['point_latitude'] == geo_point_y_center, \
                'coordinate y of geo point "' + geo_point_name + '" is not the same found in the source file'

            geo_point_row_out[self.file_fields_terrain_point_tag] = geo_grid_list_dem
            geo_point_row_out[self.file_fields_cn_point_tag] = geo_grid_list_cn
            geo_point_row_out[self.file_fields_cnet_point_tag] = geo_grid_list_cnet
            geo_point_row_out[self.file_fields_idx_1d_point_tag] = geo_idx_list_cont
            geo_point_row_out[self.file_fields_idx_2d_x_point_tag] = geo_idx_list_x
            geo_point_row_out[self.file_fields_idx_2d_y_point_tag] = geo_idx_list_y
            geo_point_row_out[self.file_fields_distance_grid_tag] = geo_distance_list
            geo_point_row_out[self.file_fields_longitude_grid_tag] = geo_grid_list_x
            geo_point_row_out[self.file_fields_latitude_grid_tag] = geo_grid_list_y

            geo_point_dframe_out = geo_point_dframe_out.append(geo_point_row_out, ignore_index=True)

            log_stream.info(' ------> Point "' + geo_point_name + '" ... DONE')

        # geo_point_dframe_out = geo_point_dframe_out.set_index('point_name')

        log_stream.info(' -----> Join the point and the grid information ... DONE')

        return geo_point_dframe_out

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to get geo grid file
    @staticmethod
    def get_geo_grid(file_name):

        log_stream.info(' -----> Read file grid "' + file_name + '" ... ')
        if file_name.endswith('txt') or file_name.endswith('asc'):
            da_grid = read_file_raster(file_name, output_format='data_array')
            log_stream.info(' -----> Read file grid "' + file_name + '" ... DONE')
        else:
            log_stream.info(' -----> Read file grid "' + file_name + '" ... FAILED')
            log_stream.error(' ===> File is mandatory to run the application')
            raise FileNotFoundError('File not found')

        return da_grid
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to get geo point file
    def geo_geo_point(self, file_name):

        log_stream.info(' -----> Read file point "' + file_name + '" ... ')

        if file_name.endswith('csv') or file_name.endswith('txt'):
            df_point = read_file_point(file_name, file_columns_remap=self.file_fields_src_point)
            log_stream.info(' -----> Read file point "' + file_name + '" ... DONE')
        else:
            log_stream.info(' -----> Read file point "' + file_name + '" ... FAILED')
            log_stream.error(' ===> File is mandatory to run the application')
            raise FileNotFoundError('File not found')

        return df_point
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to organize geographical data
    def organize_data(self):

        # Starting info
        log_stream.info(' ----> Organize geographical information ... ')

        file_path_src_grid_terrain = self.file_path_src_grid_terrain
        file_path_src_grid_cn = self.file_path_src_grid_cn
        file_path_src_grid_cnet = self.file_path_src_grid_cnet
        file_path_src_point = self.file_path_src_point

        file_path_anc = self.file_path_anc

        flag_data_updating = self.flag_data_updating

        if flag_data_updating:
            if os.path.exists(file_path_anc):
                os.remove(file_path_anc)

        if not os.path.exists(file_path_anc):

            # Read geo terrain
            da_terrain = self.get_geo_grid(file_path_src_grid_terrain)
            # Read geo cn
            da_cn = self.get_geo_grid(file_path_src_grid_cn)
            # Read geo cnet
            da_cnet = self.get_geo_grid(file_path_src_grid_cnet)

            # Read section registry
            df_point_registry = self.geo_geo_point(file_path_src_point)

            # Join grid and point datasets
            df_point_joined = self.join_geo_obj(da_terrain, da_cn, da_cnet, df_point_registry)

            # Create geo data collections
            geo_data_collections = {self.grid_terrain_tag: da_terrain,
                                    self.grid_cn_tag: da_cn,
                                    self.grid_cnet_tag: da_cnet,
                                    self.points_registry_tag: df_point_joined}

            # Dump geo collections to destination file
            folder_name_anc, file_name_anc = os.path.split(file_path_anc)
            make_folder(folder_name_anc)

            write_obj(file_path_anc, geo_data_collections)

        else:

            # Read geo collections
            geo_data_collections = read_obj(file_path_anc)

        # Ending info
        log_stream.info(' ----> Organize geographical information ... DONE')

        return geo_data_collections
    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
