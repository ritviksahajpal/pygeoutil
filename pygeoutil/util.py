import logging
import math
import os
import errno
import click
import pdb
import sys

import numpy
import netCDF4
import pandas
from skimage.measure import block_reduce
from pandas.core.common import array_equivalent
from tqdm import tqdm

import warnings
from tempfile import mkdtemp

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from joblib import Memory
    memory = Memory(cachedir=mkdtemp(), verbose=0)

# Ignore divide by 0 errors
numpy.seterr(divide='ignore', invalid='ignore')


@click.group()
def glm():
    pass


def make_dir_if_missing(d):
    """
    Create directory if not present, else do nothing
    :param d: Path of directory to create
    :return: Nothing, side-effect: create directory
    """
    try:
        os.makedirs(d)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    """
    Fast version of numpy genfromtxt
    code from here: http://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy/8964779#8964779
    """
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.lstrip().rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = numpy.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def roundup(x, near):
    """
    Round x to nearest number e.g. roundup(76, 5) gives 80
    :param x: Number to round-up
    :param near: Number to which roundup to
    :return: rounded up number
    """
    return int(math.ceil(x / near)) * near


def round_closest(x, base=10):
    return int(base * round(float(x)/base))


def delete_files(list_file_paths):
    """

    :param list_file_paths:
    :return:
    """
    logging.info('delete_files')
    for fl in list_file_paths:
        if os.path.isfile(fl):
            try:
                os.remove(fl)
            except:
                logging.info('Not able to delete ' + fl)
        else:
            logging.info('Already deleted ' + fl)


def get_ascii_header(path_file, getrows=0):
    """
    http://stackoverflow.com/questions/1767513/read-first-n-lines-of-a-file-in-python
    :param path_file:
    :param getrows:
    :return:
    """
    from itertools import islice
    with open(path_file) as inp_file:
        hdr = list(islice(inp_file, getrows))

    return hdr


def open_or_die(path_file, perm='r', csv_header=True, skiprows=0, delimiter=' ', mask_val=-9999.0, format=''):
    """
    Open file or quit gracefully
    :param path_file: Path of file to open
    :param perm: Permissions with which to open file. Default is read-only
    :param format: Special code for some file openings
    :return: Handle to file (netCDF), or dataframe (csv) or numpy array
    """

    try:
        if os.path.splitext(path_file)[1] == '.nc':
            hndl = netCDF4.Dataset(path_file, perm, format='NETCDF4')
            return hndl
        elif os.path.splitext(path_file)[1] == '.csv':
            df = pandas.read_csv(path_file, header=csv_header)
            return df
        elif os.path.splitext(path_file)[1] == '.xlsx' or os.path.splitext(path_file)[1] == '.xls':
            df = pandas.ExcelFile(path_file, header=csv_header)
            return df
        elif os.path.splitext(path_file)[1] == '.asc':
            data = iter_loadtxt(path_file, delimiter=delimiter, skiprows=skiprows)
            data = numpy.ma.masked_values(data, mask_val)
            return data
        elif os.path.splitext(path_file)[1] == '.txt':
            data = iter_loadtxt(path_file, delimiter=delimiter, skiprows=skiprows)
            data = numpy.ma.masked_values(data, mask_val)
            return data
        else:
            logging.error('Invalid file type ' + os.path.splitext(path_file)[1])
            sys.exit(0)
    except:
        logging.error('Error opening file ' + path_file)
        sys.exit(0)


def get_ascii_plot_parameters(asc, step_length=10.0):
    """
    Get min, max and step value for ascii file
    :param asc:
    :param step_length:
    :return:
    """
    if asc.max() > 0.0:
        max = math.ceil(asc.max())
    else:
        max = math.floor(asc.max())

    if asc.min() > 0.0:
        min = math.ceil(asc.min())
    else:
        min = math.floor(asc.min())

    step = (max - min)/step_length

    return min, max, step


def get_nc_var3d(hndl_nc, var, year, subset_arr=None):
    """
    Get value from netcdf for variable var for year
    :param hndl_nc:
    :param var:
    :param year:
    :param subset_arr:
    :return:
    """
    # TODO: Function assumes that subset_arr is boolean i.e. 1 or 0 (if not, errors can happen)
    use_subset_arr = subset_arr is None

    # If subset arr exists, then it should have 2 dimensions (x and y)
    if not use_subset_arr:
        ndim = subset_arr.ndim
        if ndim != 2:
            logging.error('Incorrect dimensions of subset array (should be 2): ' + str(ndim))
            sys.exit(0)

    try:
        val = hndl_nc.variables[var][year, :, :]

        if not use_subset_arr:
            # Shapes should match for netCDF slice and subset array
            if val.shape != subset_arr.shape:
                logging.error('Shapes do not match for netCDF slice and subset array')
            else:
                val = val * subset_arr
    except:
        val = numpy.nan
        logging.error('Error in getting var ' + var + ' for year ' + str(year) + ' from netcdf ')

    return val


def get_nc_var2d(hndl_nc, var, subset_arr=None):
    """
    Get value from netcdf for var
    :param hndl_nc:
    :param var:
    :param subset_arr:
    :return:
    """
    # TODO: Function assumes that subset_arr is boolean i.e. 1 or 0 (if not, errors can happen)
    use_subset_arr = subset_arr is None

    # If subset arr exists, then it should have 2 dimensions (x and y)
    if not use_subset_arr:
        ndim = subset_arr.ndim
        if ndim != 2:
            logging.error('Incorrect dimensions of subset array (should be 2): ' + str(ndim))
            sys.exit(0)

    try:
        val = hndl_nc.variables[var][:, :]

        if not use_subset_arr:
            # Shapes should match for netCDF slice and subset array
            if val.shape != subset_arr.shape:
                logging.error('Shapes do not match for netCDF slice and subset array')
            else:
                val = val * subset_arr
    except:
        val = numpy.nan
        logging.info('Error in getting var ' + var + ' from netcdf ')

    return val


@memory.cache
def get_nc_var1d(hndl_nc, var):
    """
    Get value from netcdf for var
    :param hndl_nc:
    :param var:
    :return:
    """
    try:
        val = hndl_nc.variables[var][:]
    except:
        val = numpy.nan
        logging.info('Error in getting var ' + var + ' from netcdf ')

    return val


def sum_area_nc(path_nc, var_name, carea, year):
    """

    :param path_nc:
    :param var_name:
    :param carea:
    :param year:
    :return:
    """
    hndl_nc = open_or_die(path_nc)

    return numpy.ma.sum(open_or_die(carea) * (get_nc_var3d(hndl_nc, var_name, year)))


@memory.cache
def transitions_to_matrix(flat_matrix):
    """

    :param flat_matrix:
    :return:
    """
    df_trans = flat_matrix.transpose().reset_index()

    # Determine row and column names
    df_trans['col_name'] = df_trans['index'].str.split('_to_').map(lambda x: x[1])
    df_trans['row_name'] = df_trans['index'].str.split('_to_').map(lambda x: x[0])

    # Delete duplicate rows which might exist if CFTs are not tracked
    # In that case c3ann_to_urban and c4ann_to_urban will both be cropland_to_urban
    df_trans.drop_duplicates(inplace=True)
    # Create matrix
    df_trans = df_trans.pivot(index='row_name', columns='col_name', values=0)

    # Drop names
    df_trans.index.name = None
    df_trans.columns.name = None

    # Fill Nan's by 0
    df_trans.fillna(0, inplace=True)

    return df_trans


def convert_arr_to_nc(arr, var_name, lat, lon, out_nc_path, tme=''):
    """
    :param arr: Array to convert to netCDF
    :param var_name: Name of give the variable
    :param lat:
    :param lon:
    :param out_nc_path: Output path including file name
    :param tme: Array of time values (can be empty)
    :return:
    """
    onc = open_or_die(out_nc_path, 'w')

    # dimensions
    onc.createDimension('lat', numpy.shape(lat)[0])
    onc.createDimension('lon', numpy.shape(lon)[0])
    if len(tme) > 1:
        onc.createDimension('time', numpy.shape(tme)[0])
        time = onc.createVariable('time', 'i4', ('time',))
        # Assign time
        time[:] = tme

    # variables
    latitudes = onc.createVariable('lat', 'f4', ('lat',))
    longitudes = onc.createVariable('lon', 'f4', ('lon',))

    # Metadata
    latitudes.units = 'degrees_north'
    latitudes.standard_name = 'latitude'
    longitudes.units = 'degrees_east'
    longitudes.standard_name = 'longitude'

    # Assign lats/lons
    latitudes[:] = lat
    longitudes[:] = lon

    # Assign data
    if len(tme) > 1:
        onc_var = onc.createVariable(var_name, 'f4', ('time', 'lat', 'lon',), fill_value=numpy.nan)
        # Iterate over all years
        for j in numpy.arange(tme):
            onc_var[j, :, :] = arr[j, :, :]
    else:
        # Only single year data
        onc_var = onc.createVariable(var_name, 'f4', ('lat', 'lon',), fill_value=numpy.nan)
        onc_var[:, :] = arr[:, :]

    onc.close()


@memory.cache
def convert_ascii_nc(asc_data, out_path, num_lats, num_lons, skiprows=0, var_name='data', desc='netCDF'):
    """
    Convert input ascii file to netCDF file. Compute shape from ascii file
    Assumes 2D file, no time dimension
    :param asc_data: Path to ascii file to be converted to NC
    :param out_path:
    :param skiprows:
    :param num_lats:
    :param num_lons:
    :param var_name:
    :param desc: Description of data
    :return: Path of netCDF file that was created, side-effect: create netCDF file
    """
    # Compute dimensions of nc file based on # rows/cols in ascii file
    fl_res = num_lats/asc_data.shape[0]
    if fl_res != num_lons/asc_data.shape[1]:
        # Incorrect dimensions in ascii data
        sys.exit(0)

    # Initialize nc file
    path = os.path.dirname(out_path)
    fl_name = os.path.basename(out_path).split('.')[0]
    out_nc = path + os.sep + fl_name + '.nc'

    nc_data = netCDF4.Dataset(out_nc, 'w', format='NETCDF4')
    nc_data.description = desc

    # dimensions
    nc_data.createDimension('lat', asc_data.shape[0])
    nc_data.createDimension('lon', asc_data.shape[1])

    # Populate and output nc file
    latitudes = nc_data.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_data.createVariable('longitude', 'f4', ('lon',))
    data = nc_data.createVariable(var_name, 'f4', ('lat', 'lon',), fill_value=numpy.nan)

    data.units = ''

    # set the variables we know first
    latitudes[:] = numpy.arange(90.0 - fl_res/2.0, -90.0, -fl_res)
    longitudes[:] = numpy.arange(-180.0 + fl_res/2.0, 180.0,  fl_res)
    data[:, :] = asc_data[:, :]

    nc_data.close()
    return out_nc


@memory.cache
def convert_nc_to_csv(path_nc, var_name='data', csv_out='output', do_time=False, time_var='time'):
    """
    Convert netCDF file to csv. If netCDF has a time dimension, then select last year for output
    :param path_nc: Path of netCDF file to convert to csv file
    :param var_name: Variable whose data is to be extracted
    :param csv_out: Output csv file name
    :param do_time: Is there a time dimension involved
    :param time_var: Name of time dimension
    :return: Nothing. Side-effect: Save csv file
    """
    hndl_nc = open_or_die(path_nc)

    if do_time:
        ts = get_nc_var1d(hndl_nc, var=time_var)
        nc_data = get_nc_var3d(hndl_nc, var=var_name, year=len(ts)-1)
    else:
        nc_data = get_nc_var2d(hndl_nc, var=var_name)

    # Save the data
    numpy.savetxt(csv_out+'.csv', nc_data, delimiter=', ')

    hndl_nc.close()


@memory.cache
def subtract_netcdf(left_nc, right_nc, left_var, right_var='', date=-1, tme_name='time'):
    """
    Subtract right_nc from left_nc and return numpy array
    :param left_nc: netCDF file to subtract from
    :param right_nc: netCDF file getting subtracted
    :param left_var: Variable to extract from left_nc
    :param right_var: Variable to extract from right_nc
    :param date: Which year to extract (or last year)
    :param tme_name:
    :return: numpy array (left_nc - right_nc)
    """
    hndl_left = open_or_die(left_nc)
    hndl_right = open_or_die(right_nc)

    # If right_var is not specified then assume it to be same as left_var
    # Useful to find out if netCDF is staying constant
    if len(right_var) == 0:
        right_var = left_var

    ts = get_nc_var1d(hndl_left, var=tme_name)  # time-series
    ts_r = get_nc_var1d(hndl_right, var=tme_name)  # time-series
    ts = ts if len(ts) < len(ts_r) else ts_r
    if date == -1:  # Plot either the last year {len(ts)-1} or whatever year the user wants
        yr = len(ts) - 1
    else:
        yr = date - ts[0]

    left_data = get_nc_var3d(hndl_left, var=left_var, year=yr)
    right_data = get_nc_var3d(hndl_right, var=right_var, year=yr)

    diff_data = left_data - right_data

    hndl_left.close()
    hndl_right.close()

    return diff_data


@memory.cache
def avg_netcdf(path_nc, var, do_area_wt=False, area_data='', date=-1, tme_name='time'):
    """
    Average across netCDF, can also do area based weighted average
    :param path_nc: path to netCDF file
    :param var: variable in netCDF file to average
    :param do_area_wt:
    :param area_data:
    :param date: Do it for specific date or entire time range (if date == -1)
    :param tme_name: Time!
    :return: List of sum values (could be single value if date == -1)
    """
    arr_avg = []
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, tme_name)  # time-series
    if date == -1:  # Plot either the last year {len(ts)-1} or whatever year the user wants
        iyr = ts - ts[0]
    else:
        iyr = date - ts[0]

    if do_area_wt:
        max_ar = numpy.ma.max(area_data)

        if date == -1:
            for yr in iyr:
                arr_avg.append(numpy.ma.sum(get_nc_var3d(hndl_nc, var=var, year=yr) * area_data) / max_ar)
        else:
            arr_avg.append((get_nc_var3d(hndl_nc, var=var, year=iyr) * area_data) / max_ar)
    else:
        if date == -1:
            for yr in iyr:
                arr_avg.append(numpy.ma.mean(get_nc_var3d(hndl_nc, var=var, year=yr)))
        else:
            arr_avg.append(get_nc_var3d(hndl_nc, var=var, year=yr))

    hndl_nc.close()
    return arr_avg


def avg_hist_asc(asc_data, bins=[], use_pos_vals=True, subset_asc=None, do_area_wt=False, area_data=''):
    """
    Create a weighted average array, return histogram and bin edge values
    :param asc_data: ascii data
    :param bins: Optional parameter: bins for computing histogram
    :param use_pos_vals: Use +ve values only
    :param subset_asc:
    :param do_area_wt:
    :param area_data:
    :return: The values of the histogram, bin edges
    """
    if subset_asc is not None:
        asc_data = numpy.ma.masked_where(subset_asc > 0.0, asc_data, 0.0)

    if do_area_wt:
        # Multiply fraction of grid cell by area
        ar = open_or_die(area_data)
        arr_avg = asc_data * ar
    else:
        arr_avg = asc_data

    # Select values > 0.0 since -ve values are coming from non-land areas
    if use_pos_vals:
        arr_avg = numpy.ma.masked_where(arr_avg >= 0.0, arr_avg, 0.0)

    if len(bins):
        return numpy.histogram(arr_avg, bins=bins)
    else:
        return numpy.histogram(arr_avg)


def avg_hist_netcdf(path_nc, var, bins=[], use_pos_vals=True, subset_asc=None, do_area_wt=False, area_data='', date=2015,
                    tme_name='time'):
    """
    Create a weighted average array, return histogram and bin edge values
    :param path_nc: netCDF file URI
    :param var: variable in netCDF file to average
    :param use_pos_vals: Use +ve values only
    :param subset_asc:
    :param do_area_wt:
    :param area_data:
    :param date: average variable for specific date (year)
    :param tme_name: name of time dimension in netCDF file
    :return: The values of the histogram, bin edges
    """
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, var=tme_name)  # time-series
    iyr = date - ts[0]
    hndl_nc = open_or_die(path_nc)

    return avg_hist_asc(get_nc_var3d(hndl_nc, var=var, year=iyr), bins=bins, subset_asc=subset_asc,
                        do_area_wt=do_area_wt, area_data=area_data)


def sum_netcdf(path_nc, var, do_area_wt=False, arr_area=None, precompute_area=None, date=-1, tme_name='time',
               subset_arr=None):
    """
    Sum across netCDF, can also do area based weighted sum
    :param path_nc: netCDF file
    :param var: variable in netCDF file to sum
    :param do_area_wt: Should we multiply grid cell area with fraction of grid cell
    :param arr_area: Array specifying area of each cell
    :param precompute_area
    :param date: Do it for specific date or entire time range (if date == -1)
    :param tme_name: Time!
    :param subset_arr: Subset the netCDF based on this 2D array (assuming it has 1's and 0's)
    :return: List of sum values (could be single value if date == -1)
    """
    arr_sum = []
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, var=tme_name)  # time-series
    if date == -1:  # Plot either the last year {len(ts)-1} or whatever year the user wants
        iyr = ts - ts[0]
    else:
        iyr = [date - ts[0]]

    if precompute_area == 'secondary':
        for yr in tqdm(iyr, desc='sum netcdf', disable=(len(iyr) < 2)):
            sum_secd = get_nc_var3d(hndl_nc, var='secdf', year=yr, subset_arr=subset_arr) + \
                       get_nc_var3d(hndl_nc, var='secdn', year=yr, subset_arr=subset_arr)
            nc_area = sum_secd * arr_area
            arr_sum.append(numpy.ma.sum(get_nc_var3d(hndl_nc, var=var, year=yr, subset_arr=subset_arr) * nc_area) /
                           numpy.ma.sum(nc_area))
    elif precompute_area == 'primary':
        for yr in tqdm(iyr, desc='sum netcdf', disable=(len(iyr) < 2)):
            sum_prim = get_nc_var3d(hndl_nc, var='primf', year=yr, subset_arr=subset_arr) + \
                       get_nc_var3d(hndl_nc, var='primn', year=yr, subset_arr=subset_arr)
            nc_area = sum_prim * arr_area
            arr_sum.append(numpy.ma.sum(get_nc_var3d(hndl_nc, var=var, year=yr, subset_arr=subset_arr) * nc_area) /
                           numpy.ma.sum(nc_area))
    else:
        # Multiply fraction of grid cell by area
        if do_area_wt:
            for yr in tqdm(iyr, desc='sum netcdf', disable=(len(iyr) < 2)):
                arr_sum.append(numpy.ma.sum(get_nc_var3d(hndl_nc, var=var, year=yr, subset_arr=subset_arr) * arr_area))
        else:
            for yr in tqdm(iyr, desc='sum netcdf', disable=(len(iyr) < 2)):
                arr_sum.append(numpy.ma.sum(get_nc_var3d(hndl_nc, var=var, year=yr, subset_arr=subset_arr)))

    return arr_sum


@memory.cache
def max_diff_netcdf(path_nc, var, fill_mask=False, tme_name='time'):
    """
    :param path_nc: netCDF file
    :param var: variable in netCDF file to sum
    :param fill_mask:
    :param tme_name: Time!
    :return: List of sum values (could be single value if date == -1)
    """
    logging.info('max_diff_netcdf ' + var)
    arr_diff = []
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, var=tme_name)  # time-series
    iyr = ts - ts[0]

    for yr in tqdm(iyr, desc='max_diff_netcdf', disable=(len(iyr) < 2)):
        if yr == iyr.max():
            break
        if fill_mask:
            left_yr = numpy.ma.filled(get_nc_var3d(hndl_nc, var=var, year=yr + 1), fill_value=numpy.nan)
            right_yr = numpy.ma.filled(get_nc_var3d(hndl_nc, var=var, year=yr), fill_value=numpy.nan)
        else:
            left_yr = get_nc_var3d(hndl_nc, var=var, year=yr + 1)
            right_yr = get_nc_var3d(hndl_nc, var=var, year=yr)
        arr_diff.append(numpy.max(left_yr - right_yr))

    hndl_nc.close()
    return arr_diff


@memory.cache
def avg_np_arr(data, block_size=1):
    """
    COARSENS: Takes data, and averages all positive (only numerical) numbers in blocks
    :param data: numpy array (2D)
    :param block_size:
    :return:
    """
    dimensions = data.shape

    if len(dimensions) > 2:
        logging.info("Error: Cannot handle greater than 2D numpy array")
        sys.exit(0)

    # Down-sample image by applying function to local blocks.
    # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce
    avrgd = block_reduce(data, block_size=(block_size, block_size), func=numpy.ma.mean)

    return avrgd


@memory.cache
def upscale_np_arr(data, block_size=2):
    """
    Performs interpolation to up-size or down-size images
    :param data:
    :param block_size:
    :return:
    """
    dimensions = data.shape

    if len(dimensions) > 2:
        logging.info("Error: Cannot handle greater than 2D numpy array")
        sys.exit(0)

    # http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
    from skimage.transform import resize
    # Divide data by block_size ^ 2 so that data values are right
    avrgd = resize(data/(block_size*block_size),
                   output_shape=(dimensions[0]*block_size, dimensions[1]*block_size))

    return avrgd


def downscale_nc(path_nc, var_name, out_nc_name, scale=1.0, area_name='cell_area', lat_name='lat', lon_name='lon',
                 tme_name='time'):
    """

    :param path_nc:
    :param var_name:
    :param out_nc_name:
    :param scale:
    :param area_name:
    :param lat_name:
    :param lon_name:
    :param tme_name:
    :return:
    """
    # @TODO: Fix ice-water fraction in grid cell area calculations
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, var=tme_name)  # time-series
    num_lats = len(get_nc_var2d(hndl_nc, var=lat_name))  # Number of lats in original netCDF file
    num_lons = len(get_nc_var2d(hndl_nc, var=lon_name))
    # Output netCDF file will have dimensions: (scale *num_lats , scale * num_lons)

    # Create output netCDF file
    onc = open_or_die(os.path.dirname(path_nc) + os.sep + out_nc_name, 'w')

    # dimensions
    onc.createDimension('lat', int(num_lats * scale))
    onc.createDimension('lon', int(num_lons * scale))
    if len(ts) > 1:
        onc.createDimension('time', numpy.shape(ts)[0])
        time = onc.createVariable('time', 'i4', ('time',))
        # Assign time
        time[:] = ts

    # variables
    latitudes = onc.createVariable('lat', 'f4', ('lat',))
    longitudes = onc.createVariable('lon', 'f4', ('lon',))
    cell_area = onc.createVariable(area_name, 'f4', ('lat', 'lon',), fill_value=numpy.nan)

    # Metadata
    latitudes.units = 'degrees_north'
    latitudes.standard_name = 'latitude'
    longitudes.units = 'degrees_east'
    longitudes.standard_name = 'longitude'

    # Assign lats/lons
    # CF conventions - cell boundaries essentially stating that lat/lon are in the center of the grid cell
    latitudes[:] = numpy.arange(90.0 - fl_res/2.0, -90.0, -fl_res)
    longitudes[:] = numpy.arange(-180.0 + fl_res/2.0, 180.0,  fl_res)

    # Assign data
    if len(ts) > 1:
        onc_var = onc.createVariable(var_name, 'f4', ('time', 'lat', 'lon',), fill_value=numpy.nan)
        # Iterate over all years
        for j in numpy.arange(len(ts)):
            # Get data from coarse resolution netCDF
            coarse_arr = get_nc_var3d(hndl_nc, var=var_name, year=j)
            # Create finer resolution numpy array
            finer_arr = coarse_arr.repeat(scale, 0).repeat(scale, 1)
            onc_var[j, :, :] = finer_arr[:, :]
    else:
        # Only single year data
        onc_var = onc.createVariable(var_name, 'f4', ('lat', 'lon',), fill_value=numpy.nan)
        # Get data from coarse resolution netCDF
        coarse_arr = get_nc_var2d(hndl_nc, var=var_name)
        # Create finer resolution numpy array
        finer_arr = coarse_arr.repeat(scale, 0).repeat(scale, 1)
        onc_var[:, :] = finer_arr[:, :]

    # Assign area
    # Get data from coarse resolution netCDF
    coarse_arr = get_nc_var2d(hndl_nc, var=area_name)
    # Create finer resolution numpy array
    finer_arr = coarse_arr.repeat(scale, 0).repeat(scale, 1)
    # cell_area[:, :] = (finer_arr[:, :] * constants.M2_TO_KM2)/(scale * scale)  # convert from m^2 to km^2 and
    # downscale
    cell_area[:, :] = (finer_arr[:, :])/(scale * scale)

    onc.close()


def extract_from_ascii(asc, ulat=90.0, llat=-90.0, llon=-180.0, rlon=180.0, res=1.0, subset_arr=None):
    """
    Extract from ascii, a window defined by ulat (top), llat(bottom), llon(left), rlon(right)
    :param asc:
    :param ulat:
    :param llat:
    :param llon:
    :param rlon:
    :param res:
    :param subset_arr:
    :return:
    """
    top_row = (90.0 - ulat)/res
    bottom_row = top_row + abs(llat - ulat)/res

    left_column = abs(-180.0 - llon)/res
    right_column = left_column + abs(rlon - llon)/res

    if subset_arr is None:
        asc_subset = asc[top_row:bottom_row, left_column:right_column]
    else:
        asc_subset = asc[top_row:bottom_row, left_column:right_column] * \
                     subset_arr[top_row:bottom_row, left_column:right_column]

    return asc_subset


def duplicate_columns(frame):
    """
    http://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
    :param frame:
    :return:
    """
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:, i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:, j].values
                if array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break

    return dups


def send_email(to=[], subject='', contents=[]):
    """

    :param to:
    :param subject:
    :param contents:
    :return:
    """
    import yagmail

    try:
        yag = yagmail.SMTP('sahajpal.ritvik@gmail.com')
        yag.send(to=to, subject=subject, contents=contents)
        yag.close()
    except:
        logging.info('Error in sending email')

if __name__ == '__main__':
    glm()

