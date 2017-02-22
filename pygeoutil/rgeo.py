import os
import pdb
import re
import rasterio
import gdal
import glob
import json
import subprocess

import urllib.request
import pandas as pd
import numpy as np
import multiprocessing
import sys
from . import ggeo

from tsgettoolbox import tsgettoolbox
from geopy.geocoders import Nominatim
import geocoder
from geopy.exc import GeocoderTimedOut
from shutil import copyfile
from functools import lru_cache
import shapefile
import fiona
from shapely.geometry import mapping, shape

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


def dms2dd(degrees, minutes, seconds, direction):
    """
    http://en.proft.me/2015/09/20/converting-latitude-and-longitude-decimal-values-p/
    Args:
        degrees:
        minutes:
        seconds:
        direction:

    Returns:

    """
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)

    if direction == 'S' or direction == 'W':
        dd *= -1

    return dd


def dd2dms(deg):
    """
    http://en.proft.me/2015/09/20/converting-latitude-and-longitude-decimal-values-p/
    Args:
        deg:

    Returns:

    """
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60

    return [d, m, sd]


def parse_dms(dms):
    """
    http://en.proft.me/2015/09/20/converting-latitude-and-longitude-decimal-values-p/
    example: parse_dms("36°57'9' N 110°4'21' W")
    Args:
        dms:

    Returns:

    """
    parts = re.split('[^\d\w]+', dms)
    lat = dms2dd(parts[0], parts[1], parts[2], parts[3])
    lng = dms2dd(parts[4], parts[5], parts[6], parts[7])

    return lat, lng


def get_met(lat, lon, start_date=None, end_date=None):
    """

    :param lat: 
    :param lon: 
    :param start_date: 
    :param end_date: 
    :return: 
    """
    from pcse.db import NASAPowerWeatherDataProvider
    wdp = NASAPowerWeatherDataProvider(lat, lon, start_date, end_date)
    _df = wdp._query_NASAPower_server(lat, lon, start_date, end_date)

    _df = [x.decode("utf-8") for x in _df]  # Decode from bytes to strings
    _ix = [i for i, s in enumerate(_df) if 'END HEADER' in s][0]  # Find end of header
    _header = [x for x in _df[_ix - 1].split()]  # Header is one line before end of header
    _df = _df[_ix + 1:]  # All data (except after END HEADER)

    # Convert data to dataframe
    df = pd.DataFrame(columns=_header, data=[row.split() for row in _df])
    df.replace('-', np.NaN, inplace=True)
    df = pd.to_numeric(df.stack(), 'coerce').unstack()
    df['datetime'] = pd.to_datetime(df['YEAR'].astype(int), format='%Y') + pd.to_timedelta(df['DOY'] - 1, unit='d')

    # Add datetime column and set it as index
    if 'datetime' in df and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('datetime')

    return df


def get_ts(lat, lon, name_var, var=None, sub_var=None, start_date=None, end_date=None):
    """

    :param lat: 
    :param lon: 
    :param name_var: e.g. 'modis'    
    :param var: e.g. 
    :param sub_var: e.g. MYD17A2
    :param start_date: 
    :param end_date: 
    :return: 
    """
    if name_var == 'modis':
        df = tsgettoolbox.modis(lat=lat,
                                lon=lon,
                                product=var,  # e.g. MOD13Q1
                                band=sub_var,  # e.g. 250m_16_days_NDVI
                                startdate=start_date,
                                enddate=end_date)
    elif name_var == 'ldas':
        # NOTE: Only works for lat/lon within U.S.A (GLDAS gives HTTP 404 error)
        df = tsgettoolbox.ldas(lat=lat,
                               lon=lon,
                               variable=var,  # e.g. 'GLDAS:GLDAS_NOAH025_3H.001:TSOIL0-10cm'
                               startDate=start_date,
                               endDate=end_date)
    elif name_var == 'power':  # NASA POWER
        df = get_met(lat,
                     lon,
                     start_date=start_date,
                     end_date=end_date)
    else:
        raise ValueError

    return df


def get_elev(lat, lon, name_var):
    """
    Get elevation from GOOGLE eleeation API    
    :param lat: 
    :param lon: 
    :param name_var: 
    :return: 
    """
    if name_var == 'elev':
        base_url = 'http://maps.googleapis.com/maps/api/elevation/json?'
        params_url = "locations=%s,%s&sensor=%s" % (lat, lon, 'false')
        url = base_url + params_url

        with urllib.request.urlopen(url) as f:
            response = json.loads(f.read().decode())

        # status = response['status']  # TODO Make sure status is correct
        result = response['results'][0]

        # Returns height above sea level in metres
        return float(result['elevation'])
    else:
        raise ValueError


def get_properties(path_ds, name_property):
    """

    Args:
        path_ds:
        name_property:

    Returns:

    """
    dict_properties = ggeo.get_raster_properties_uri(path_ds)

    return dict_properties[name_property]


def get_values_rat_column(path_ds, name_col='Value'):
    """

    Args:
        path_ds:
        name_col:

    Returns:

    """
    _rat = subprocess.check_output('gdalinfo -json ' + path_ds, shell=True)
    data = json.loads(_rat)  # load json string into dictionary

    # dict_values = ggeo.get_rat_as_dictionary_uri(path_ds)
    _col_names = [x['name'] for x in data['rat']['fieldDefn']]
    df = pd.DataFrame(columns=_col_names)
    for _row in data['rat']['row']:
         df.loc[len(df)] = _row['f']

    return df[df.columns[df.columns.to_series().str.contains(name_col)]].values.T[0]
    #name_key = [s for s in dict_values.keys() if '.' + name_col in s]

    #return dict_values.get(name_key[0], None)


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

    ggeo.reclassify_dataset_uri(path_ds,
                                dict_reclass,
                                path_out_ds,
                                out_datatype=ggeo.get_dataset_datatype(path_ds),
                                out_nodata=ggeo.get_nodata_from_uri(path_ds))


def get_arr_res(lats, lons):
    """
    Get resolution of array from lat lon data
    Args:
        lats:
        lons:

    Returns:

    """
    # Check if both lat and lon have same shape
    if np.isclose(np.abs(lats[1] - lats[0]),  np.abs(lons[1] - lons[0])):
        return np.abs(lats[1] - lats[0])
    else:
        raise ValueError('lat and lon do not have same shape')


def get_geo_idx(val_dd, array_dd):
    """
    Get the index of the nearest decimal degree in an array of decimal degrees
    Args:
        val_dd:
        array_dd:

    Returns:

    """
    return (np.abs(array_dd - val_dd)).argmin()


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
    except Exception as e:
        # TODO Be smarter about catching exceptions here, needs internet connection to work
        # Default behaviour is to exit
        print('Geolocator not working: ' + loc + ' ' + str(e))
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


def get_country_lat_lon_extent(alpha2):
    """

    Args:
        alpha2:

    Returns: longitude(left) longitude(right), latitude (bottom), latitude(top)

    """
    #
    # 'mexico', 'south_africa', 'spain', 'australia', 'ukraine', 'uk_of_great_britain_and_northern_ireland',
    # 'germany','spain', 'kazakhstan', 'hungary', 'italy','indonesia'
    if alpha2 == 'united_states_of_america':
        return [-130, -60, 25, 48]
    elif alpha2 == 'russian_federation':
        return [22, 179, 35, 83]
    elif alpha2 == 'china':
        return [70, 138, 12, 55]
    elif alpha2 == 'india':
        return [64, 100, 4, 37]
    elif alpha2 == 'argentina':
        return [-80, -50, -16, -64]
    elif alpha2 == 'brazil':
        return [-75, -35, 5, -35]
    elif alpha2 == 'canada':
        return [-140, -50, 40, 70]
    elif alpha2 == 'egypt':
        return [-80, -50, -16, -64]
    elif alpha2 == 'france':
        return [-80, -50, -16, -64]
    elif alpha2 == 'mexico':
        return [-120, -85, 15, 35]
    elif alpha2 == 'south_africa':
        return [10, 35, -20, -35]
    elif alpha2 == 'spain':
        return [-80, -50, -16, -64]
    elif alpha2 == 'australia':
        return [-80, -50, -16, -64]
    elif alpha2 == 'ukraine':
        return [-80, -50, -16, -64]
    elif alpha2 == 'uk_of_great_britain_and_northern_ireland':
        return [-80, -50, -16, -64]
    elif alpha2 == 'germany':
        return [-80, -50, -16, -64]
    elif alpha2 == 'spain':
        return [-80, -50, -16, -64]
    elif alpha2 == 'kazakhstan':
        return [-80, -50, -16, -64]
    elif alpha2 == 'hungary':
        return [-80, -50, -16, -64]
    elif alpha2 == 'italy':
        return [-80, -50, -16, -64]
    elif alpha2 == 'indonesia':
        return [-80, -50, -16, -64]
    else:
        return [-180, 180, -90, 90]


def clip_raster(path_raster, path_mask, path_out_ras):
    """

    :param path_raster: Raster
    :param path_mask: Shapefile
    :param path_out_ras: Raster
    :param process_pool: Parallel or not
    :return:
    """
    import subprocess

    if os.path.splitext(path_mask)[1] != '.shp':
        _cmd = 'rio clip ' + path_raster + ' ' + path_out_ras + ' --like ' + path_mask
    else:
        _cmd = 'rio clip ' + path_raster + ' ' + path_out_ras + ' --bounds $(fio info ' + path_mask + ' --bounds)'

    try:
        subprocess.check_output(_cmd, stderr=subprocess.STDOUT, shell=True)
    except:
        raise ValueError('clip_raster encountered error')


def select(path_inp, name_col, val, path_out):
    """
    Select from shapefile by attribute
    Replacement of ArcGIS: http://pro.arcgis.com/en/pro-app/tool-reference/data-management/select-layer-by-attribute.htm
    Args:
        path_inp:
        name_col:
        val:
        path_out:

    Returns:

    """

    with fiona.open(path_inp) as src:
        with fiona.open(path_out, 'w', **src.meta) as sink:
            for feature in src:
                if feature['properties'][name_col] == val:
                    sink.write(feature)

            # filtered = filter(lambda f: f['properties'][name_col] == val, src)
            # pdb.set_trace()
            # geom = shape(filtered[0]['geometry'])
            # filtered[0]['geometry'] = mapping(geom)
            #
            # sink.write(filtered[0])

if __name__ == '__main__':
    pass
