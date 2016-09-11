import os
import pdb
import gdal
import glob
import gdalconst
import pandas as pd
import numpy as np
import sys

from pygeoprocessing import geoprocessing as gp
from geopy.geocoders import Nominatim
import geocoder
from geopy.exc import GeocoderTimedOut
from shutil import copyfile
from functools32 import lru_cache
import shapefile
import rasterio

# TODO zonal statistics: https://github.com/perrygeo/python-rasterstats
# resize and resample:  http://data.naturalcapitalproject.org/pygeoprocessing/api/latest/api/geoprocessing.html
# temporary_filename
# temporary_folder
# unique_raster_values, unique_raster_values_count
# vectorize_datasets
# assert_datasets_in_same_projection
# calculate_raster_stats_uri
# clip_dataset_uri
# create_rat_uri
# sieve: http://pktools.nongnu.org/html/pksieve.html
# composite/mosaic: http://pktools.nongnu.org/html/pkcomposite.html
# mosaic


def get_dataset_type(path_ds):
    """
    Return dataset type e.g. GeoTiff
    Args:
        path_ds:

    Returns:

    """
    dataset = gdal.Open(path_ds, gdalconst.GA_ReadOnly)
    dataset_type = dataset.GetDriver().LongName

    dataset = None  # Close dataset

    return dataset_type


def get_dataset_datatype(path_ds):
    """
    Return datatype of dataset e.g. GDT_UInt32
    Args:
        path_ds:

    Returns:

    """
    dataset = gdal.Open(path_ds, gdalconst.GA_ReadOnly)

    band = dataset.GetRasterBand(1)
    bandtype = gdal.GetDataTypeName(band.DataType)  # UInt32

    dataset = None  # Close dataset

    if bandtype == 'UInt32':
        return gdalconst.GDT_UInt32
    elif bandtype == 'UInt16':
        return gdalconst.GDT_UInt16
    elif bandtype == 'Float32':
        return gdalconst.GDT_Float32
    elif bandtype == 'Float64':
        return gdalconst.GDT_Float64
    elif bandtype == 'Int16':
        return gdalconst.GDT_Int16
    elif bandtype == 'Int32':
        return gdalconst.GDT_Int32
    elif bandtype == 'Unknown':
        return gdalconst.GDT_Unknown
    else:
        return gdalconst.GDT_UInt32


def get_properties(path_ds, name_property):
    """

    Args:
        path_ds:
        name_property:

    Returns:

    """
    dict_properties = gp.get_raster_properties_uri(path_ds)

    return dict_properties[name_property]


def get_values_rat_column(path_ds, name_col='Value'):
    """

    Args:
        path_ds:
        name_col:

    Returns:

    """
    dict_values = gp.get_rat_as_dictionary_uri(path_ds)

    name_key = [s for s in dict_values.keys() if '.' + name_col in s]

    return dict_values.get(name_key[0], None)


def lookup(path_ds, path_out_ds, from_field='Value', to_field='', overwrite=True):
    """

    Args:
        path_ds:
        path_out_ds:
        from_field:
        to_field:
        overwrite:

    Returns:

    """
    val_from = get_values_rat_column(path_ds, name_col=from_field)
    val_to = get_values_rat_column(path_ds, name_col=to_field)

    dict_reclass = dict(zip(val_from, val_to))
    gp.reclassify_dataset_uri(path_ds, dict_reclass, path_out_ds, out_datatype=get_dataset_datatype(path_ds),
                              out_nodata=gp.get_nodata_from_uri(path_ds))


def convert_raster_to_ascii(path_input_raster, path_ascii_output, overwrite=True):
    """
    Convert input raster to ascii format
    Args:
        path_input_raster:
        path_ascii_output:
        overwrite:

    Returns:

    """
    if overwrite and os.path.isfile(path_ascii_output):
        os.remove(path_ascii_output)

    # Open existing dataset
    path_inp_ds = gdal.Open(path_input_raster)

    # Open output format driver, gdal_translate --formats lists all of them
    format_file = 'AAIGrid'
    driver = gdal.GetDriverByName(format_file)

    # Output to new format
    path_dest_ds = driver.CreateCopy(path_ascii_output, path_inp_ds, 0)

    # Close the datasets to flush to disk
    path_dest_ds = None
    path_inp_ds = None


@lru_cache(maxsize=1024)
def get_latlon_location(loc):
    """
    Get latitude/longitude of location
    Args:
        loc:

    Returns:

    """
    try:
        # Geopy
        geolocator = Nominatim()
        location = geolocator.geocode(loc, timeout=5)
        lat = location.latitude
        lon = location.longitude
    except Exception, e:
        # TODO Be smarter about catching exceptions here, needs internet connection to work
        # Default behaviour is to exit
        print 'Geolocator not working: ' + loc + ' ' + str(e)
        return 0.0, 0.0  # Equator is default

    return lat, lon


@lru_cache(maxsize=1024)
def get_hemisphere(loc, boundary=0.0):
    """
    Get hemisphere in which a location lies (northern/southern)
    Args:
        loc: Name of country/region to use to get latitude
        boundary: Latitude above which it is N hemisphere

    Returns:

    """
    lat, _ = get_latlon_location(loc)

    if lat >= boundary:
        return 'N'
    else:
        return 'S'


@lru_cache(maxsize=1024)
def is_temperate(loc, n_boundary=23.5, s_boundary=-23.5):
    """

    Args:
        loc:
        n_boundary:
        s_boundary:

    Returns:

    """
    lat, _ = get_latlon_location(loc)

    if lat >= n_boundary or lat <= s_boundary:
        return 'Temperate'
    else:
        return 'Tropical'


# Vector (shapefile data)
def get_att_table_shpfile(path_shpfile):
    """

    Args:
        path_shpfile:

    Returns:

    """
    # Read shapefile data into dataframe
    hndl_shp = shapefile.Reader(path_shpfile)

    fields = hndl_shp.fields[1:]
    field_names = [field[0] for field in fields]

    # construction of a dictionary field_name:value
    df_shp = pd.DataFrame(columns=field_names)
    for rec in hndl_shp.shapeRecords():
        df_shp.loc[len(df_shp)] = rec.record

    return df_shp


def copy_shpfile(path_inp_shp, path_out_shp):
    """

    Args:
        path_inp_shp:
        path_out_shp:

    Returns:

    """
    files_to_copy = glob.glob(os.path.dirname(path_inp_shp) + os.sep +
                              os.path.splitext(os.path.basename(path_inp_shp))[0] + '*')

    name_new_file = os.path.splitext(os.path.basename(path_out_shp))[0]

    path_inp = os.path.dirname(path_inp_shp)
    path_out = os.path.dirname(path_out_shp)

    for fl in files_to_copy:
        ext = os.path.splitext(fl)[1]
        copyfile(fl, path_out + os.sep + name_new_file + ext)


def extract_at_point_from_ras(path_ras, lon, lat):
    """
    Extract value from raster at given longitude and latitude
    East and North are positive, West and South are negative
    Args:
        path_ras:
        lon: 27.8 (E)
        lat: -13.13 (S)

    Returns:

    """
    with rasterio.drivers():
        # Read raster bands directly to numpy arrays
        with rasterio.open(path_ras) as src:
            return np.asarray(src.sample([(lon, lat)]))


def get_grid_cell_area(nrows, ncols):
    """

    :param nrows: Number of rows
    :param ncols: Number of columns
    :return:
    """
    R = 6371.0  # radius of earth
    csize = 180. / nrows

    cell_area = np.zeros(shape=(nrows, ncols))
    lat = np.zeros(shape=(nrows,))

    for i in range(nrows):
        lat[i] = i * csize

    lat = lat * np.pi / 180.

    sarea = np.pi / 180. * csize * np.power(R, 2.) * (np.cos(lat) - np.cos(lat + np.pi * csize / 180))
    cell_area = np.zeros((nrows, ncols))

    for i in range(nrows):
        cell_area[i, :] = sarea[i]

    return cell_area


if __name__ == '__main__':
    pass
