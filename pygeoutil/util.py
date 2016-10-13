import math
import os
import errno
import pdb
import datetime
import sys
import calendar

import rgeo

import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from skimage.measure import block_reduce
from tqdm import tqdm

# Ignore divide by 0 errors.
np.seterr(divide='ignore', invalid='ignore')


######################
# Miscellaneous
######################
def get_key_from_val(dicts, name_val):
    """
    Find if name_val is in one of the keys in dicts, return corresponding key
    Args:
        dicts:
        name_val:

    Returns:

    """
    for key, val in dicts.items():
        if name_val in val:
            return key
    else:
        raise ValueError(name_val + ' does not exist')


def transitions_to_matrix(flat_matrix):
    """

    Args:
        flat_matrix:

    Returns:

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


def duplicate_columns(frame):
    """
    http://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
    Args:
        frame:

    Returns:

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
                if pd.core.common.array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break

    return dups


def get_git_revision_hash():
    """

    Returns:

    """
    import subprocess

    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def send_email(to=[], subject='', contents=[]):
    """
    send email
    Args:
        to:
        subject:
        contents:

    Returns:

    """
    import yagmail

    try:
        yag = yagmail.SMTP('sahajpal.ritvik@gmail.com')
        yag.send(to=to, subject=subject, contents=contents)
        yag.close()
    except:
        raise ValueError('Error in sending email')


def compose_date(years, months=1, days=1, weeks=None, hours=None, minutes=None, seconds=None, milliseconds=None,
                 microseconds=None, nanoseconds=None):
    """
    From http://stackoverflow.com/questions/34258892/converting-year-and-day-of-year-into-datetime-index-in-pandas
    Args:
        years:
        months:
        days:
        weeks:
        hours:
        minutes:
        seconds:
        milliseconds:
        microseconds:
        nanoseconds:

    Returns:

    """
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1

    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')

    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)

    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals) if v is not None)


def get_month_names():
    """

    Returns:

    """
    list_mon_names = []
    for i in range(12):
        list_mon_names.append(calendar.month_abbr[i + 1].title())

    return list_mon_names


def nan_helper(y):
    """
    http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Helper to handle indices and logical indices of NaN
    Args:
        y: y, 1d numpy array with possible NaNs

    Returns:
     - nans, logical indices of NaNs
     - index, a function, with signature indices= index(logical_indices),
       to convert logical indices of NaNs to 'equivalent' indices

    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def sliding_mean(data_array, window=5):
    """
    This function takes an array of numbers and smoothes them out.
    Smoothing is useful for making plots a little easier to read.
    Args:
        data_array:
        window:

    Returns:

    """
    # Return without change if window size is zero
    if window == 0:
        return data_array

    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


######################
# numpy array ops
######################
def replace_subset_arr(lats, lons, cell_size, subset_arr):
    """

    Args:
        lats:
        lons:
        cell_size:
        subset_arr:

    Returns:

    """
    arr_global = np.zeros((int(180. // cell_size), int(360. // cell_size)))

    latitudes = np.arange(90.0 - cell_size / 2.0, -90.0, -cell_size)
    longitudes = np.arange(-180.0 + cell_size / 2.0, 180.0, cell_size)

    start_row = rgeo.get_geo_idx(lats[0], latitudes)
    end_row = rgeo.get_geo_idx(lats[-1], latitudes)

    start_col = rgeo.get_geo_idx(lons[0], longitudes)
    end_col = rgeo.get_geo_idx(lons[-1], longitudes)

    subset_arr = np.nan_to_num(subset_arr)
    arr_global[start_row:end_row + 1, start_col:end_col + 1] = subset_arr

    return arr_global


def avg_np_arr(data, area_cells=None, block_size=1, func=np.ma.mean):
    """
    COARSENS: Takes data, and averages all positive (only numerical) numbers in blocks
    E.g. with a block_size of 2, convert (720 x 1440) array into (360 x 720)
    Args:
        data: numpy array (2D)
        area_cells:
        block_size:

    Returns:

    """
    dimensions = data.shape

    if area_cells is not None and area_cells.shape != dimensions:
        raise AssertionError('Shape of input data array and area array should be the same but is not')

    if len(dimensions) > 2:
        raise AssertionError('Error: Cannot handle greater than 2D numpy array')

    # Down-sample image by applying function to local blocks.
    # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce
    if area_cells is not None:
        avrgd = block_reduce(data * area_cells, block_size=(block_size, block_size), func=func)
    else:
        avrgd = block_reduce(data, block_size=(block_size, block_size), func=func)

    return avrgd


def upscale_np_arr(data, area_cells=None, block_size=2):
    """
    Performs interpolation to up-size or down-size images
    Args:
        data:
        area_cells:
        block_size:

    Returns:

    """
    dimensions = data.shape

    if area_cells is not None and area_cells.shape != dimensions:
        raise AssertionError('Shape of input data array and area array should be the same but is not')

    if len(dimensions) > 2:
        raise AssertionError('Error: Cannot handle greater than 2D numpy array')

    # http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
    from skimage.transform import resize
    # Divide data by block_size ^ 2 so that data values are right
    if area_cells is not None:
        avrgd = resize((data * area_cells)/(block_size*block_size),
                       output_shape=(dimensions[0]*block_size, dimensions[1]*block_size))
    else:
        avrgd = resize(data/(block_size*block_size),
                       output_shape=(dimensions[0]*block_size, dimensions[1]*block_size))

    return avrgd


######################
# Ascii files
######################
def get_ascii_header(path_file, getrows=0):
    """
    http://stackoverflow.com/questions/1767513/read-first-n-lines-of-a-file-in-python
    Args:
        path_file:
        getrows:

    Returns:

    """
    from itertools import islice
    with open(path_file) as inp_file:
        hdr = list(islice(inp_file, getrows))

    return hdr


def extract_from_ascii(asc, ulat=90.0, llat=-90.0, llon=-180.0, rlon=180.0, res=1.0, subset_arr=None):
    """
    Extract from ascii, a window defined by ulat (top), llat(bottom), llon(left), rlon(right)
    Args:
        asc:
        ulat:
        llat:
        llon:
        rlon:
        res:
        subset_arr:

    Returns:

    """
    top_row = int((90.0 - ulat)/res)
    bottom_row = int(top_row + abs(llat - ulat)/res)

    left_column = int(abs(-180.0 - llon)/res)
    right_column = int(left_column + abs(rlon - llon)/res)

    if subset_arr is None:
        asc_subset = asc[top_row:bottom_row, left_column:right_column]
    else:
        asc_subset = asc[top_row:bottom_row, left_column:right_column] * \
                     subset_arr[top_row:bottom_row, left_column:right_column]

    return asc_subset


def avg_hist_asc(asc_data, bins=[], use_pos_vals=True, subset_asc=None, do_area_wt=False, area_data=''):
    """
    Create a weighted average array, return histogram and bin edge values
    Args:
        asc_data: ascii data
        bins: Optional parameter: bins for computing histogram
        use_pos_vals: Use +ve values only
        subset_asc:
        do_area_wt:
        area_data:

    Returns:
        The values of the histogram, bin edges

    """
    if subset_asc is not None:
        asc_data = np.ma.masked_where(subset_asc > 0.0, asc_data, 0.0)

    if do_area_wt:
        # Multiply fraction of grid cell by area
        arr_avg = asc_data * area_data
    else:
        arr_avg = asc_data

    # Select values > 0.0 since -ve values are coming from non-land areas
    if use_pos_vals:
        arr_avg = np.ma.masked_where(arr_avg >= 0.0, arr_avg, 0.0)

    if len(bins):
        return np.histogram(arr_avg, bins=bins)
    else:
        return np.histogram(arr_avg)


def write_ascii(arr, path_out, name_fl, ncols, nrows, cell_size, xllcorner=-180, yllcorner=-90, nodata_val=-9999):
    """
    Output numpy array (arr) to ascii file
    Args:
        arr:
        path_out:
        name_fl:
        ncols:
        nrows:
        cell_size:
        xllcorner:
        yllcorner:
        nodata_val:

    Returns:

    """
    make_dir_if_missing(path_out)

    asc_file = open(path_out + os.sep + name_fl, 'w+')

    asc_file.write('ncols         %s\n' % ncols)
    asc_file.write('nrows         %s\n' % nrows)
    asc_file.write('xllcorner     %s\n' % xllcorner)
    asc_file.write('yllcorner     %s\n' % yllcorner)
    asc_file.write('cellsize      %s\n' % cell_size)
    asc_file.write('NODATA_value  %s\n' % nodata_val)

    np.savetxt(asc_file, arr, fmt='%.6f', delimiter=' ')
    asc_file.close()


######################
# Managing file system
######################
def get_modification_date(filename):
    """
    http://stackoverflow.com/questions/237079/how-to-get-file-creation-modification-date-times-in-python
    E.g. print modification_date('/var/log/syslog')
    >>> 2009-10-06 10:50:01
    Args:
        filename:

    Returns:

    """
    t = os.path.getmtime(filename)

    return datetime.datetime.fromtimestamp(t)


def go_higher_dir_levels(path_to_dir, level=0):
    """
    Gien directory path, go up number of levels defined by level
    :param path_to_dir:
    :param level:
    :return:
    """
    up_dir = path_to_dir

    for lev in range(level):
        up_dir = os.path.dirname(path_to_dir)

    return up_dir


def make_dir_if_missing(d):
    """
    Create directory if not present, else do nothing
    Args:
        d: Path of directory to create

    Returns:
        Nothing, side-effect: create directory

    """
    try:
        os.makedirs(d)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    finally:
        return d


def delete_files(list_file_paths):
    """

    Args:
        list_file_paths:

    Returns:

    """
    for fl in list_file_paths:
        if os.path.isfile(fl):
            try:
                os.remove(fl)
            except:
                raise IOError('Not able to delete ' + fl)
        else:
            raise IOError('Already deleted ' + fl)


######################
# Mathematical ops
######################
def round_down(x, near):
    """
    Round x to nearest number e.g. round_down(79, 5) gives 75
    Args:
        x: Number to round-down
        near: Number to which round-down to

    Returns:
        rounded up number

    """
    return int(math.ceil(x // near)) * near


def round_up(x, near):
    """
    Round x to nearest number e.g. round_up(76, 5) gives 80
    Args:
        x: Number to round-up
        near: Number to which round-up to

    Returns:
        rounded up number

    """
    return int(math.ceil(float(x) / float(near))) * near


def round_closest(x, base=10):
    """

    Args:
        x:
        base:

    Returns:

    """
    return int(base * round(float(x)/base))


######################
# File handling
######################
def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    """
    Fast version of numpy genfromtxt
    code from here: http://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy/8964779#8964779
    Args:
        filename:
        delimiter:
        skiprows:
        dtype:

    Returns:

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

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def open_or_die(path_file, perm='r', csv_header=0, skiprows=0, delimiter=' ', mask_val=-9999.0, use_xarray=False,
                format=''):
    """
    Open file or quit gracefully
    Args:
        path_file: Path of file to open
        perm: Permissions with which to open file. Default is read-only
        csv_header: see http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
        skiprows:
        delimiter:
        mask_val:
        use_xarray:
        format: Special code for some file openings

    Returns:
        Handle to file (netCDF), or dataframe (csv) or numpy array

    """
    try:
        if os.path.splitext(path_file)[1] == '.nc' and not use_xarray:
            hndl = netCDF4.Dataset(path_file, perm, format='NETCDF4')
            return hndl
        elif os.path.splitext(path_file)[1] == '.csv':
            df = pd.read_csv(path_file, header=csv_header)
            return df
        elif os.path.splitext(path_file)[1] == '.xlsx' or os.path.splitext(path_file)[1] == '.xls':
            df = pd.ExcelFile(path_file, header=csv_header)
            return df
        elif os.path.splitext(path_file)[1] == '.asc':
            data = iter_loadtxt(path_file, delimiter=delimiter, skiprows=skiprows)
            data = np.ma.masked_values(data, mask_val)
            return data
        elif os.path.splitext(path_file)[1] == '.txt':
            data = iter_loadtxt(path_file, delimiter=delimiter, skiprows=skiprows)
            data = np.ma.masked_values(data, mask_val)
            return data
        elif os.path.splitext(path_file)[1] == '.nc' and use_xarray:
            merge_nc_files(path_file)
        else:
            raise IOError('Invalid file type ' + os.path.splitext(path_file)[1])
            sys.exit(0)
    except:
        raise IOError('Error opening file ' + path_file)
        sys.exit(0)


def get_ascii_plot_parameters(asc, step_length=10.0):
    """
    Get min, max and step value for ascii file
    Args:
        asc:
        step_length:

    Returns:

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


######################
# netCDF
######################
def get_nc_var4d(hndl_nc, var, year, pos=0, use_xarray=False):
    """

    Args:
        hndl_nc:
        var:
        year:
        pos:
        use_xarray:

    Returns:

    """
    # Return if number of dimensions is not 4
    # TODO: Assumes xarray, will not work for netCDF4
    if use_xarray:
        if len(hndl_nc.variables[var].dims) != 4:
            return np.nan
    else:
        if len(hndl_nc.variables[var].dimensions) != 4:
            return np.nan

    if pos == 0:
        if use_xarray:
            val = hndl_nc.variables[var][year, :, :, :].values
        else:
            val = hndl_nc.variables[var][year, :, :, :]
    elif pos == 1:
        if use_xarray:
            val = hndl_nc.variables[var][:, year, :, :].values
        else:
            val = hndl_nc.variables[var][:, year, :, :]
    elif pos == 2:
        if use_xarray:
            val = hndl_nc.variables[var][:, :, year, :].values
        else:
            val = hndl_nc.variables[var][:, :, year, :]
    else:
        if use_xarray:
            val = hndl_nc.variables[var][:, :, :, year].values
        else:
            val = hndl_nc.variables[var][:, :, :, year]

    return val


def get_nc_var3d(hndl_nc, var, year, subset_arr=None):
    """
    Get value from netcdf for variable var for year
    Args:
        hndl_nc:
        var:
        year:
        subset_arr:

    Returns:

    """
    # TODO: Function assumes that subset_arr is boolean i.e. 1 or 0 (if not, errors can happen)
    use_subset_arr = subset_arr is None

    # Return if number of dimensions is not 3
    if len(hndl_nc.variables[var].dimensions) != 3:
        return np.nan

    # If subset arr exists, then it should have 2 dimensions (x and y)
    if not use_subset_arr:
        ndim = subset_arr.ndim
        if ndim != 2:
            raise IOError('Incorrect dimensions of subset array (should be 2): ' + str(ndim))

    try:
        val = hndl_nc.variables[var][year, :, :]

        if not use_subset_arr:
            # Shapes should match for netCDF slice and subset array
            if val.shape != subset_arr.shape:
                raise AttributeError('Shapes do not match for netCDF slice and subset array')
            else:
                val = val * subset_arr
    except:
        val = np.nan
        raise AttributeError('Error in getting var ' + var + ' for year ' + str(year) + ' from netcdf ')

    return val


def get_nc_var2d(hndl_nc, var, subset_arr=None):
    """
    Get value from netcdf for var
    Args:
        hndl_nc:
        var:
        subset_arr:

    Returns:

    """
    # Return if number of dimensions is not 2
    if len(hndl_nc.variables[var].dimensions) != 2:
        return np.nan

    # TODO: Function assumes that subset_arr is boolean i.e. 1 or 0 (if not, errors can happen)
    use_subset_arr = subset_arr is None

    # If subset arr exists, then it should have 2 dimensions (x and y)
    if not use_subset_arr:
        ndim = subset_arr.ndim
        if ndim != 2:
            raise AttributeError('Incorrect dimensions of subset array (should be 2): ' + str(ndim))

    try:
        val = hndl_nc.variables[var][:, :]

        if not use_subset_arr:
            # Shapes should match for netCDF slice and subset array
            if val.shape != subset_arr.shape:
                raise AttributeError('Shapes do not match for netCDF slice and subset array')
            else:
                val = val * subset_arr
    except:
        val = np.nan
        raise AttributeError('Error in getting var ' + var + ' from netcdf ')

    return val


def get_nc_var1d(hndl_nc, var):
    """
    Get value from netcdf for var
    Args:
        hndl_nc:
        var:

    Returns:

    """
    # Return if number of dimensions is not 1
    if len(hndl_nc.variables[var].dimensions) != 1:
        return np.nan

    try:
        val = hndl_nc.variables[var][:]
    except:
        val = np.nan
        raise AttributeError('Error in getting var ' + var + ' from netcdf ')

    return val


def get_time_nc(path_nc, name_time_var='time'):
    """

    Args:
        path_nc:
        name_time_var:

    Returns:

    """
    hndl_nc = open_or_die(path_nc)

    var_time = get_nc_var1d(hndl_nc, var=name_time_var)

    return var_time


def get_dims_in_nc(path_nc):
    """

    Args:
        path_nc:

    Returns:

    """
    hndl_nc = open_or_die(path_nc)
    list_dims = []

    for name_dim, dim in hndl_nc.dimensions.items():
        # Append dimension names to list_dims
        list_dims.append(name_dim)

    return list_dims


def rename_vars_in_nc(path_nc, dict_rename):
    """

    Args:
        path_nc:
        dict_rename:

    Returns:

    """
    list_vars = get_vars_in_nc(path_nc)

    with open_or_die(path_nc, perm='r+') as hndl_nc:
        for var in list_vars:
            if var in dict_rename.keys() and var != dict_rename[var][0]:
                hndl_nc.renameVariable(var, dict_rename[var][0])


def get_vars_in_nc(path_nc, ignore_var=None):
    """
    Get list of variables in netCDF file, ignore variables in ignore_var list
    Args:
        path_nc:
        ignore_var:

    Returns:

    """
    list_vars = []

    with open_or_die(path_nc) as hndl_nc:
        for idx, (name_var, var) in enumerate(hndl_nc.variables.items()):
            if ignore_var is not None:
                if name_var in ignore_var:
                    continue

            # Append variable to list of variables
            list_vars.append(name_var)

    return list_vars


def create_nc_var(hndl_nc, var, name_var, dims):
    """

    Args:
        hndl_nc:
        var:
        name_var:
        dims:

    Returns:

    """
    dtype = 'f8' if var.dtype == 'datetime64[ns]' else var.dtype

    out_var_nc = hndl_nc.createVariable(name_var, dtype, dims, zlib=True)
    out_var_nc.setncatts({k: var.getncattr(k) for k in var.ncattrs()})

    return out_var_nc


def sum_area_nc(path_nc, var_name, carea, year):
    """

    Args:
        path_nc:
        var_name:
        carea:
        year:

    Returns:

    """
    hndl_nc = open_or_die(path_nc)

    return np.ma.sum(carea * (get_nc_var3d(hndl_nc, var_name, year)))


def convert_arr_to_nc(arr, var_name, lat, lon, out_nc_path, tme=''):
    """

    Args:
        arr: Array to convert to netCDF
        var_name: Name of give the variable
        lat:
        lon:
        out_nc_path: Output path including file name
        tme: Array of time values (can be empty)

    Returns:

    """
    onc = open_or_die(out_nc_path, 'w')

    # dimensions
    onc.createDimension('lat', np.shape(lat)[0])
    onc.createDimension('lon', np.shape(lon)[0])
    if len(tme) > 1:
        onc.createDimension('time', np.shape(tme)[0])
        time = onc.createVariable('time', 'i4', ('time',), zlib=True)
        # Assign time
        time[:] = tme

    # variables
    latitudes = onc.createVariable('lat', 'f4', ('lat',), zlib=True)
    longitudes = onc.createVariable('lon', 'f4', ('lon',), zlib=True)

    # Metadata
    latitudes.units = 'degrees_north'
    latitudes.standard_name = 'lat'
    longitudes.units = 'degrees_east'
    longitudes.standard_name = 'lon'

    # Assign lats/lons
    latitudes[:] = lat
    longitudes[:] = lon

    # Assign data
    if len(tme) > 1:
        onc_var = onc.createVariable(var_name, 'f4', ('time', 'lat', 'lon',), fill_value=np.nan, zlib=True)
        # Iterate over all years
        for j in np.arange(tme):
            onc_var[j, :, :] = arr[j, :, :]
    else:
        # Only single year data
        onc_var = onc.createVariable(var_name, 'f4', ('lat', 'lon',), fill_value=np.nan, zlib=True)
        onc_var[:, :] = arr[:, :]

    onc.close()


def convert_ascii_nc(asc_data, out_path, num_lats, num_lons, skiprows=0, var_name='data', desc='netCDF'):
    """
    Convert input ascii file to netCDF file. Compute shape from ascii file
    Assumes 2D file, no time dimension
    Args:
        asc_data: Path to ascii file to be converted to NC
        out_path:
        num_lats:
        num_lons:
        skiprows:
        var_name:
        desc: Description of data

    Returns:
        Path of netCDF file that was created, side-effect: create netCDF file

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
    latitudes = nc_data.createVariable('lat', 'f4', ('lat',), zlib=True)
    longitudes = nc_data.createVariable('lon', 'f4', ('lon',), zlib=True)
    data = nc_data.createVariable(var_name, 'f4', ('lat', 'lon',), fill_value=np.nan, zlib=True)

    data.units = ''

    # set the variables we know first
    latitudes[:] = np.arange(90.0 - fl_res/2.0, -90.0, -fl_res)
    longitudes[:] = np.arange(-180.0 + fl_res/2.0, 180.0,  fl_res)
    data[:, :] = asc_data[:, :]

    nc_data.close()
    return out_nc


def convert_nc_to_csv(path_nc, var_name='data', csv_out='output', do_time=False, time_var='time'):
    """
    Convert netCDF file to csv. If netCDF has a time dimension, then select last year for output
    Args:
        path_nc: Path of netCDF file to convert to csv file

        var_name: Variable whose data is to be extracted

        csv_out: Output csv file name

        do_time: Is there a time dimension involved

        time_var: Name of time dimension

    Returns:
        Nothing. Side-effect: Save csv file

    """
    hndl_nc = open_or_die(path_nc)

    if do_time:
        ts = get_nc_var1d(hndl_nc, var=time_var)
        nc_data = get_nc_var3d(hndl_nc, var=var_name, year=len(ts)-1)
    else:
        nc_data = get_nc_var2d(hndl_nc, var=var_name)

    # Save the data
    np.savetxt(csv_out+'.csv', nc_data, delimiter=', ')

    hndl_nc.close()


def subtract_netcdf(left_hndl, right_hndl, left_var, right_var=None, date=-1, tme_name='time'):
    """
    Subtract right_hndl from left_nc and return numpy array
    Args:
        left_hndl: netCDF file to subtract from
        right_hndl: netCDF file getting subtracted
        left_var: Variable to extract from left_nc
        right_var: Variable to extract from right_hndl
        date: Which year to extract (or last year)
        tme_name:

    Returns:
        numpy array (left_hndl - right_hndl)
    """

    # If right_var is not specified then assume it to be same as left_var
    # Useful to find out if netCDF is staying constant
    if right_var is None:
        right_var = left_var

    ts = get_nc_var1d(left_hndl, var=tme_name)  # time-series
    ts_r = get_nc_var1d(right_hndl, var=tme_name)  # time-series
    ts = ts if len(ts) < len(ts_r) else ts_r
    if date == -1:  # Plot either the last year {len(ts)-1} or whatever year the user wants
        yr = len(ts) - 1
    else:
        yr = date - ts[0]

    left_data = get_nc_var3d(left_hndl, var=left_var, year=yr)
    right_data = get_nc_var3d(right_hndl, var=right_var, year=yr)

    diff_data = left_data - right_data

    return diff_data


def avg_netcdf(path_nc, var, do_area_wt=False, area_data='', date=-1, tme_name='time'):
    """
    Average across netCDF, can also do area based weighted average
    Args:
        path_nc: path to netCDF file
        var: variable in netCDF file to average
        do_area_wt:
        area_data:
        date: Do it for specific date or entire time range (if date == -1)
        tme_name:

    Returns:
        List of sum values (could be single value if date == -1)

    """
    arr_avg = []
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, tme_name)  # time-series
    if date == -1:  # Plot either the last year {len(ts)-1} or whatever year the user wants
        iyr = ts - ts[0]
    else:
        iyr = date - ts[0]

    if do_area_wt:
        max_ar = np.ma.max(area_data)

        if date == -1:
            for yr in iyr:
                arr_avg.append(np.ma.sum(get_nc_var3d(hndl_nc, var=var, year=yr) * area_data) / max_ar)
        else:
            arr_avg.append((get_nc_var3d(hndl_nc, var=var, year=iyr) * area_data) / max_ar)
    else:
        if date == -1:
            for yr in iyr:
                arr_avg.append(np.ma.mean(get_nc_var3d(hndl_nc, var=var, year=yr)))
        else:
            arr_avg.append(get_nc_var3d(hndl_nc, var=var, year=yr))

    hndl_nc.close()
    return arr_avg


def avg_hist_netcdf(path_nc, var, bins=[], use_pos_vals=True, subset_asc=None, do_area_wt=False, area_data='', date=2015,
                    tme_name='time'):
    """
    Create a weighted average array, return histogram and bin edge values
    Args:
        path_nc: path to netCDF file
        var: variable in netCDF file to average
        bins:
        use_pos_vals: Use +ve values only
        subset_asc:
        do_area_wt:
        area_data:
        date: average variable for specific date (year)
        tme_name: name of time dimension in netCDF file

    Returns:
        The values of the histogram, bin edges

    """
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, var=tme_name)  # time-series
    iyr = date - ts[0]
    hndl_nc = open_or_die(path_nc)

    return avg_hist_asc(get_nc_var3d(hndl_nc, var=var, year=iyr), bins=bins, subset_asc=subset_asc,
                        do_area_wt=do_area_wt, area_data=area_data)


def sum_netcdf(path_nc, var, do_area_wt=False, arr_area=None, date=-1, tme_name='time', subset_arr=None):
    """
    Sum across netCDF, can also do area based weighted sum
    Args:
        path_nc: path to netCDF file
        var: variable in netCDF file to sum
        do_area_wt: Should we multiply grid cell area with fraction of grid cell
        arr_area: Array specifying area of each cell
        date: Do it for specific date or entire time range (if date == -1)
        tme_name:
        subset_arr: Subset the netCDF based on this 2D array (assuming it has 1's and 0's)

    Returns:
        List of sum values (could be single value if date == -1)

    """
    arr_sum = []
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, var=tme_name)  # time-series

    # Plot either the last year {len(ts)-1} or whatever year the user wants
    iyr = ts - ts[0] if date == -1 else [date - ts[0]]

    # Multiply fraction of grid cell by area
    for yr in tqdm(iyr, desc='sum netcdf', disable=(len(iyr) < 2)):
        if do_area_wt:
            arr_sum.append(np.ma.sum(get_nc_var3d(hndl_nc, var=var, year=yr, subset_arr=subset_arr) * arr_area))
        else:
            arr_sum.append(np.ma.sum(get_nc_var3d(hndl_nc, var=var, year=yr, subset_arr=subset_arr)))

    return arr_sum


def max_diff_netcdf(path_nc, var, fill_mask=False, tme_name='time'):
    """

    Args:
        path_nc: path to netCDF file
        var: variable in netCDF file to sum
        fill_mask:
        tme_name:

    Returns:
        List of sum values (could be single value if date == -1)

    """
    arr_diff = []
    hndl_nc = open_or_die(path_nc)

    ts = get_nc_var1d(hndl_nc, var=tme_name)  # time-series
    iyr = ts - ts[0]

    for yr in tqdm(iyr, desc='max_diff_netcdf', disable=(len(iyr) < 2)):
        if yr == iyr.max():
            break
        if fill_mask:
            left_yr = np.ma.filled(get_nc_var3d(hndl_nc, var=var, year=yr + 1), fill_value=np.nan)
            right_yr = np.ma.filled(get_nc_var3d(hndl_nc, var=var, year=yr), fill_value=np.nan)
        else:
            left_yr = get_nc_var3d(hndl_nc, var=var, year=yr + 1)
            right_yr = get_nc_var3d(hndl_nc, var=var, year=yr)
        arr_diff.append(np.max(left_yr - right_yr))

    hndl_nc.close()
    return arr_diff


def downscale_nc(path_nc, var_name, out_nc_name, scale=1.0, area_name='cell_area', lat_name='lat', lon_name='lon',
                 tme_name='time'):
    """

    Args:
        path_nc:
        var_name:
        out_nc_name:
        scale:
        area_name:
        lat_name:
        lon_name:
        tme_name:

    Returns:

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
        onc.createDimension('time', np.shape(ts)[0])
        time = onc.createVariable('time', 'i4', ('time',), zlib=True)
        # Assign time
        time[:] = ts

    # variables
    latitudes = onc.createVariable('lat', 'f4', ('lat',), zlib=True)
    longitudes = onc.createVariable('lon', 'f4', ('lon',), zlib=True)
    cell_area = onc.createVariable(area_name, 'f4', ('lat', 'lon',), fill_value=np.nan, zlib=True)

    # Metadata
    latitudes.units = 'degrees_north'
    latitudes.standard_name = 'latitude'
    longitudes.units = 'degrees_east'
    longitudes.standard_name = 'longitude'

    # Assign lats/lons
    # CF conventions - cell boundaries essentially stating that lat/lon are in the center of the grid cell
    latitudes[:] = np.arange(90.0 - fl_res/2.0, -90.0, -fl_res)
    longitudes[:] = np.arange(-180.0 + fl_res/2.0, 180.0,  fl_res)

    # Assign data
    if len(ts) > 1:
        onc_var = onc.createVariable(var_name, 'f4', ('time', 'lat', 'lon',), fill_value=np.nan, zlib=True)
        # Iterate over all years
        for j in np.arange(len(ts)):
            # Get data from coarse resolution netCDF
            coarse_arr = get_nc_var3d(hndl_nc, var=var_name, year=j)
            # Create finer resolution numpy array
            finer_arr = coarse_arr.repeat(scale, 0).repeat(scale, 1)
            onc_var[j, :, :] = finer_arr[:, :]
    else:
        # Only single year data
        onc_var = onc.createVariable(var_name, 'f4', ('lat', 'lon',), fill_value=np.nan, zlib=True)
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


def add_bounds_to_nc(path_inp_nc, do_lat_bounds=True, do_lon_bounds=True, do_time_bounds=True):
    """

    Args:
        path_inp_nc:
        do_lat_bounds:
        do_lon_bounds:
        do_time_bounds:

    Returns:

    """
    with open_or_die(path_inp_nc, perm='r+') as hndl_nc:
        name_dims = get_dims_in_nc(path_inp_nc)
        name_vars = get_vars_in_nc(path_inp_nc)

        # Create bounds dimension
        if 'bounds' not in name_dims:
            hndl_nc.createDimension('bounds', 2)

        if do_lat_bounds and 'lat_bounds' not in name_vars:
            lats = hndl_nc.variables['lat'][:]
            out_var = hndl_nc.createVariable('lat_bounds', 'f8', ('lat', 'bounds',), zlib=True)
            out_var[:] = np.vstack((lats - 0.5 * (lats[1] - lats[0]), lats + 0.5 * (lats[1] - lats[0]))).T

        if do_lon_bounds and 'lon_bounds' not in name_vars:
            lons = hndl_nc.variables['lon'][:]
            out_var = hndl_nc.createVariable('lon_bounds', 'f8', ('lon', 'bounds',), zlib=True)
            out_var[:] = np.vstack((lons - 0.5 * (lons[1] - lons[0]), lons + 0.5 * (lons[1] - lons[0]))).T

        if do_time_bounds and 'time_bnds' not in name_vars:
            time = hndl_nc.variables['time'][:]
            out_var = hndl_nc.createVariable('time_bnds', 'i4', ('time', 'bounds',), zlib=True)

            for idx in range(time.shape[0]):
                out_var[idx, 0] = 1
                out_var[idx, 1] = 365


def modify_desc_in_nc(path_nc, val_att):
    """
    Modify the global description attribute in netCDF file
    Args:
        path_nc:
        val_att

    Returns:

    """
    with open_or_die(path_nc, perm='r+') as hndl_nc:
        if 'description' in hndl_nc.ncattrs():
            hndl_nc.description = hndl_nc.description + '; ' + val_att
        else:
            hndl_nc.description = val_att


def add_nc_vars_to_new_var(path_inp, vars, new_var='tmp'):
    """

    Args:
        path_inp:
        vars:
        new_var:

    Returns:

    """
    if new_var in get_vars_in_nc(path_inp):
        return

    with open_or_die(path_inp, perm='r+') as hndl_inp:
        for idx, (name_var, var) in enumerate(hndl_inp.variables.items()):
            if name_var in vars:
                out_var = hndl_inp.createVariable(new_var, var.datatype, var.dimensions, zlib=True)
                out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                break

        # Create empty array
        arr3d = np.zeros_like(hndl_inp.variables[vars[0]])
        for v in vars:
            arr3d[:] = arr3d[:].data + hndl_inp.variables[v][:].data

        # Assign data to new variable
        out_var[:] = arr3d[:]


def replace_negative_vals_in_nc(path_inp):
    """

    Args:
        path_inp:

    Returns:

    """

    with open_or_die(path_inp, perm='r+') as hndl_inp:
        for idx, (name_var, var) in enumerate(hndl_inp.variables.items()):
            hndl_inp[name_var][:][hndl_inp[name_var][:] < 0] = 0.0


def create_new_var_in_nc(path_inp, example_var, new_var='tmp'):
    """
    Add new_var to existing netCDF, fill with 0's and assign it dimension and datatype of existing variable
    Args:
        path_inp:
        vars:
        new_var:

    Returns:

    """
    if new_var in get_vars_in_nc(path_inp):
        return

    with open_or_die(path_inp, perm='r+') as hndl_inp:
        for idx, (name_var, var) in enumerate(hndl_inp.variables.items()):
            if name_var in example_var:
                out_var = hndl_inp.createVariable(new_var, var.datatype, var.dimensions, zlib=True)
                out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                break

        # Assign data to new variable
        out_var[:] = np.zeros_like(hndl_inp.variables[example_var])


def modify_nc_att(path_inp, vars, att_to_modify, new_att_value):
    """

    Args:
        path_inp:
        vars:
        att_to_modify:
        new_att_value:

    Returns:

    """
    with open_or_die(path_inp, perm='r+') as hndl_inp:
        for idx, (name_var, var) in enumerate(hndl_inp.variables.items()):
            if name_var in vars:
                for k in var.ncattrs():
                    if k == att_to_modify:
                        var.setncatts({k: new_att_value})


def modify_nc_val(path_inp, var, new_val):
    """

    Args:
        path_inp:
        var:
        new_val:

    Returns:

    """
    with open_or_die(path_inp, perm='r+') as hndl_inp:
        hndl_inp[var][:] = new_val


def merge_nc_files(list_nc_files, path_out_nc, common_var_name='', replace_var_by_file_name=False):
    """

    Args:
        list_nc_files:
        path_out_nc:
        common_var_name:
        replace_var_by_file_name: If True, then replace common_var_name variable by name of file

    Returns:

    """
    list_dims = []  # List of dimensions in input netCDF file
    list_vars = []

    if os.path.exists(path_out_nc):
        return

    with open_or_die(path_out_nc, perm='w') as hndl_out_nc:
        for fl in list_nc_files:
            with open_or_die(fl) as hndl_nc:
                # Copy dimensions
                for name_dim, dim in hndl_nc.dimensions.items():
                    if name_dim not in list_dims:
                        # Append dimension names to list_dims
                        list_dims.append(name_dim)
                        hndl_out_nc.createDimension(name_dim, len(dim) if not dim.isunlimited() else None)

                # Copy variables
                for idx, (name_var, var) in enumerate(hndl_nc.variables.items()):
                    if name_var not in list_vars:
                        if replace_var_by_file_name and name_var == common_var_name:
                            new_name_var = os.path.splitext(os.path.basename(fl))[0]
                            list_vars.append(new_name_var)

                            out_var = hndl_out_nc.createVariable(new_name_var, var.datatype, var.dimensions, zlib=True)
                        else:
                            list_vars.append(name_var)
                            out_var = hndl_out_nc.createVariable(name_var, var.datatype, var.dimensions, zlib=True)

                        # Copy variable attributes
                        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})

                        # Copy variable data
                        out_var[:] = var[:]


if __name__ == '__main__':
    pass
