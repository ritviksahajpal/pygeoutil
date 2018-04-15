import os
import pdb
import time
import errno
import subprocess
import logging
import tempfile
import distutils.version
import atexit
import functools
import math

import numpy
import gdal
import gdalconst
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

LOGGER = logging.getLogger('pygeoprocessing.geoprocessing')
LOGGER.addHandler(logging.NullHandler())  # silence logging by default
_LOGGING_PERIOD = 5.0  # min 5.0 seconds per update log message for the module
_DEFAULT_GTIFF_CREATION_OPTIONS = (
    'TILED=YES', 'BIGTIFF=IF_SAFER', 'COMPRESS=LZW')
_LARGEST_ITERBLOCK = 2**20  # largest block for iterblocks to read in cells

# A dictionary to map the resampling method input string to the gdal type
_RESAMPLE_DICT = {
    "nearest": gdal.GRA_NearestNeighbour,
    "bilinear": gdal.GRA_Bilinear,
    "cubic": gdal.GRA_Cubic,
    "cubic_spline": gdal.GRA_CubicSpline,
    "lanczos": gdal.GRA_Lanczos,
    'mode': gdal.GRA_Mode,
    'average': gdal.GRA_Average,
}


# GDAL 2.2.3 added a couple of useful interpolation values.
if (distutils.version.LooseVersion(gdal.__version__)
        >= distutils.version.LooseVersion('2.2.3')):
    _RESAMPLE_DICT.update({
        'max': gdal.GRA_Max,
        'min': gdal.GRA_Min,
        'med': gdal.GRA_Med,
        'q1': gdal.GRA_Q1,
        'q3': gdal.GRA_Q3,
    })



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


def _gdal_to_numpy_type(band):
    """Calculates the equivalent numpy datatype from a GDAL raster band type

        band - GDAL band

        returns numpy equivalent of band.DataType"""

    gdal_type_to_numpy_lookup = {
        gdal.GDT_Int16: numpy.int16,
        gdal.GDT_Int32: numpy.int32,
        gdal.GDT_UInt16: numpy.uint16,
        gdal.GDT_UInt32: numpy.uint32,
        gdal.GDT_Float32: numpy.float32,
        gdal.GDT_Float64: numpy.float64
    }

    if band.DataType in gdal_type_to_numpy_lookup:
        return gdal_type_to_numpy_lookup[band.DataType]

    #only class not in the lookup is a Byte but double check.
    if band.DataType != gdal.GDT_Byte:
        raise ValueError("Unknown DataType: %s" % str(band.DataType))

    metadata = band.GetMetadata('IMAGE_STRUCTURE')
    if 'PIXELTYPE' in metadata and metadata['PIXELTYPE'] == 'SIGNEDBYTE':
        return numpy.int8
    return numpy.uint8


def get_datatype_from_uri(dataset_uri):
    """
    Returns the datatype for the first band from a gdal dataset

    Args:
        dataset_uri (string): a uri to a gdal dataset

    Returns:
        datatype: datatype for dataset band 1"""

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    datatype = band.DataType

    #Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return datatype


def get_row_col_from_uri(dataset_uri):
    """
    Returns a tuple of number of rows and columns of that dataset uri.

    Args:
        dataset_uri (string): a uri to a gdal dataset

    Returns:
        tuple (tuple): 2-tuple (n_row, n_col) from dataset_uri"""

    dataset = gdal.Open(dataset_uri)
    n_rows = dataset.RasterYSize
    n_cols = dataset.RasterXSize

    #Make sure the dataset is closed and cleaned up
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return (n_rows, n_cols)


def calculate_raster_stats_uri(dataset_uri):
    """
    Calculates and sets the min, max, stdev, and mean for the bands in
    the raster.

    Args:
        dataset_uri (string): a uri to a GDAL raster dataset that will be
            modified by having its band statistics set

    Returns:
        nothing
    """

    dataset = gdal.Open(dataset_uri, gdal.GA_Update)

    for band_number in range(dataset.RasterCount):
        band = dataset.GetRasterBand(band_number + 1)
        band.ComputeStatistics(False)

    # Close and clean up dataset
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None


def get_statistics_from_uri(dataset_uri):
    """
    Retrieves the min, max, mean, stdev from a GDAL Dataset

    Args:
        dataset_uri (string): a uri to a gdal dataset

    Returns:
        statistics: min, max, mean, stddev

    """

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    statistics = band.GetStatistics(0, 1)

    # Close and clean up dataset
    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return statistics


def get_cell_size_from_uri(dataset_uri):
    """
    Returns the cell size of the dataset in meters.  Raises an exception if the
    raster is not square since this'll break most of the raster_utils
    algorithms.

    Args:
        dataset_uri (string): uri to a gdal dataset

    Returns:
        size_meters: cell size of the dataset in meters"""

    srs = osr.SpatialReference()
    dataset = gdal.Open(dataset_uri)
    if dataset is None:
        raise IOError(
            'File not found or not valid dataset type at: %s' % dataset_uri)
    srs.SetProjection(dataset.GetProjection())
    linear_units = srs.GetLinearUnits()
    geotransform = dataset.GetGeoTransform()
    # take absolute value since sometimes negative widths/heights
    try:
        numpy.testing.assert_approx_equal(
            abs(geotransform[1]), abs(geotransform[5]))
        size_meters = abs(geotransform[1]) * linear_units
    except AssertionError as e:
        size_meters = (
            abs(geotransform[1]) + abs(geotransform[5])) / 2.0 * linear_units

    # Close and clean up dataset
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return size_meters


def get_rat_as_dictionary_uri(dataset_uri):
    """
    Returns the RAT of the first band of dataset as a dictionary.

    Args:
        dataset_uri: a GDAL dataset that has a RAT associated with the first band

    Returns:
        value (dictionary): a 2D dictionary where the first key is the column name and second is the row number

    """

    dataset = gdal.Open(dataset_uri)
    value = get_rat_as_dictionary(dataset)

    # Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return value


def get_rat_as_dictionary(dataset):
    """
    Returns the RAT of the first band of dataset as a dictionary.

    Args:
        dataset: a GDAL dataset that has a RAT associated with the first band

    Returns:
        rat_dictionary (dictionary): a 2D dictionary where the first key is the column name and second is the row number
    """
    pdb.set_trace()
    band = dataset.GetRasterBand(1).GetDefaultRAT()
    rat = band.GetDefaultRAT()
    n_columns = rat.GetColumnCount()
    n_rows = rat.GetRowCount()
    rat_dictionary = {}

    for col_index in range(n_columns):
        # Initialize an empty list to store row data and figure out the type of data stored in that column.
        col_type = rat.GetTypeOfCol(col_index)
        col_name = rat.GetNameOfCol(col_index)
        rat_dictionary[col_name] = []

        # Now burn through all the rows to populate the column
        for row_index in range(n_rows):
            # This bit of python ugliness handles the known 3 types of gdal RAT fields.
            if col_type == gdal.GFT_Integer:
                value = rat.GetValueAsInt(row_index, col_index)
            elif col_type == gdal.GFT_Real:
                value = rat.GetValueAsDouble(row_index, col_index)
            else:
                # If the type is not int or real, default to a string, I think this is better than testing for a string
                # and raising an exception if not
                value = rat.GetValueAsString(row_index, col_index)

            rat_dictionary[col_name].append(value)

    return rat_dictionary


def get_raster_properties_uri(dataset_uri):
    """
    Wrapper function for get_raster_properties() that passes in the dataset
    URI instead of the datasets itself

    Args:
        dataset_uri (string): a URI to a GDAL raster dataset

    Returns:
        value (dictionary): a dictionary with the properties stored under relevant keys. The current list of things
        returned is: width (w-e pixel resolution), height (n-s pixel resolution), XSize, YSize
    """
    dataset = gdal.Open(dataset_uri)
    value = get_raster_properties(dataset)

    # Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return value


def get_raster_properties(dataset):
    """
    Get the width, height, X size, and Y size of the dataset and return the
    values in a dictionary.
    *This function can be expanded to return more properties if needed*

    Args:
       dataset: a GDAL raster dataset to get the properties from

    Returns:
        dataset_dict (dictionary): a dictionary with the properties stored under relevant keys. The current list of
        things returned is: width (w-e pixel resolution), height (n-s pixel resolution), XSize, YSize
    """
    dataset_dict = {}

    geo_transform = dataset.GetGeoTransform()
    dataset_dict['width'] = float(geo_transform[1])
    dataset_dict['height'] = float(geo_transform[5])
    dataset_dict['x_size'] = dataset.GetRasterBand(1).XSize
    dataset_dict['y_size'] = dataset.GetRasterBand(1).YSize

    return dataset_dict


def get_nodata_from_uri(dataset_uri):
    """
    Returns the nodata value for the first band from a gdal dataset cast to its
        correct numpy type.

    Args:
        dataset_uri (string): a uri to a gdal dataset

    Returns:
        nodata_cast: nodata value for dataset band 1

    """

    dataset = gdal.Open(dataset_uri)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        nodata = _gdal_to_numpy_type(band)(nodata)
    else:
        pass

    band = None
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None
    return nodata


def reclassify(rasterio_rst, reclass_list, output_filename, band=1, creation_options=dict()):
    """
        MODIFIED: removed window walking...  too slow..
        this function will take a raster image as input and
        reclassify its values given in the reclass_list.
        The reclass list is a simple list of lists with the
        following formatting:
            [[begin_range, end_range, new_value]]
            ie. [ [ 1,3,5 ],[ 3,4,6 ] ]
                * which converts values 1 to 2.99999999 to 5
                    and values 3 to 3.99999999 to 6
                    all other values stay the same.
        arguments:
            rasterio_rst = raster image instance from rasterio package
            reclass_list = list of reclassification values * see explanation
            band = integer marking which band you wnat to return from the raster
                    default is 1.
            creation_options = gdal style creation options, but in the rasterio implementation
                * options must be in a dict where the key is the name of the gdal -co and the
                  value is the value passed to that flag.
                  i.e.
                    ["COMPRESS=LZW"] becomes dict([('compress','lzw')])
    """
    # this will update the metadata if a creation_options dict is passed as an arg.
    import rasterio
    meta = rasterio_rst.meta

    if len(creation_options) < 0:
        meta.update(creation_options)

    with rasterio.open(output_filename, mode='w', **meta) as out_rst:
        band_arr = rasterio_rst.read_band(band).data # this is a gotcha with the .data stuff

        for rcl in reclass_list:
            band_arr[numpy.logical_and(band_arr >= rcl[0], band_arr < rcl[1])] = rcl[2]

        out_rst.write_band(band, band_arr)

    return rasterio.open(output_filename)


def get_cell_size_from_uri(dataset_uri):
    """Get the cell size of a dataset in units of meters.

    Raises an exception if the raster is not square since this'll break most of
    the pygeoprocessing algorithms.

    Args:
        dataset_uri (string): uri to a gdal dataset

    Returns:
        size_meters: cell size of the dataset in meters
    """

    srs = osr.SpatialReference()
    dataset = gdal.Open(dataset_uri)
    if dataset is None:
        raise IOError(
            'File not found or not valid dataset type at: %s' % dataset_uri)
    srs.SetProjection(dataset.GetProjection())
    linear_units = srs.GetLinearUnits()
    geotransform = dataset.GetGeoTransform()
    # take absolute value since sometimes negative widths/heights
    try:
        numpy.testing.assert_approx_equal(
            abs(geotransform[1]), abs(geotransform[5]))
        size_meters = abs(geotransform[1]) * linear_units
    except AssertionError as e:
        size_meters = (
            abs(geotransform[1]) + abs(geotransform[5])) / 2.0 * linear_units

    # Close and clean up dataset
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return size_meters


def reclassify_dataset_uri(
        dataset_uri, value_map, raster_out_uri, out_datatype, out_nodata,
        exception_flag='values_required', assert_dataset_projected=True):
    """Reclassify values in a dataset.

    A function to reclassify values in dataset to any output type. By default
    the values except for nodata must be in value_map.

    Args:
        dataset_uri (string): a uri to a gdal dataset
        value_map (dictionary): a dictionary of values of
            {source_value: dest_value, ...}
            where source_value's type is a postive integer type and dest_value
            is of type out_datatype.
        raster_out_uri (string): the uri for the output raster
        out_datatype (gdal type): the type for the output dataset
        out_nodata (numerical type): the nodata value for the output raster.
            Must be the same type as out_datatype

    Keyword Args:
        exception_flag (string): either 'none' or 'values_required'.
            If 'values_required' raise an exception if there is a value in the
            raster that is not found in value_map
        assert_dataset_projected (boolean): if True this operation will
            test if the input dataset is not projected and raise an exception
            if so.

    Returns:
        nothing

    Raises:
        Exception: if exception_flag == 'values_required' and the value from
           'key_raster' is not a key in 'attr_dict'
    """
    if exception_flag not in ['none', 'values_required']:
        raise ValueError('unknown exception_flag %s', exception_flag)
    values_required = exception_flag == 'values_required'

    nodata = get_nodata_from_uri(dataset_uri)
    value_map_copy = value_map.copy()
    # possible that nodata value is not defined, so test for None first
    # otherwise if nodata not predefined, remap it into the dictionary
    if nodata is not None and nodata not in value_map_copy:
        value_map_copy[nodata] = out_nodata

    keys = sorted(value_map_copy.keys())
    values = numpy.array([value_map_copy[x] for x in keys])

    def map_dataset_to_value(original_values):
        """Convert a block of original values to the lookup values."""
        if values_required:
            unique = numpy.unique(original_values)
            has_map = numpy.in1d(unique, keys)
            if not all(has_map):
                raise ValueError(
                    'There was not a value for at least the following codes '
                    '%s for this file %s.\nNodata value is: %s' % (
                        str(unique[~has_map]), dataset_uri, str(nodata)))
        index = numpy.digitize(original_values.ravel(), keys, right=True)
        return values[index].reshape(original_values.shape)

    out_pixel_size = get_cell_size_from_uri(dataset_uri)
    vectorize_datasets(
        [dataset_uri], map_dataset_to_value,
        raster_out_uri, out_datatype, out_nodata, out_pixel_size,
        "intersection", dataset_to_align_index=0,
        vectorize_op=False, assert_datasets_projected=assert_dataset_projected,
        datasets_are_pre_aligned=True)


def clip_dataset_uri(
        source_dataset_uri, aoi_datasource_uri, out_dataset_uri,
        assert_projections=True, process_pool=None, all_touched=False):
    """Clip raster dataset to bounding box of provided vector datasource aoi.

    This function will clip source_dataset to the bounding box of the
    polygons in aoi_datasource and mask out the values in source_dataset
    outside of the AOI with the nodata values in source_dataset.

    Args:
        source_dataset_uri (string): uri to single band GDAL dataset to clip
        aoi_datasource_uri (string): uri to ogr datasource
        out_dataset_uri (string): path to disk for the clipped datset

    Keyword Args:
        assert_projections (boolean): a boolean value for whether the dataset
            needs to be projected
        process_pool: a process pool for multiprocessing
        all_touched (boolean): if true the clip uses the option
            ALL_TOUCHED=TRUE when calling RasterizeLayer for AOI masking.

    Returns:
        None
    """
    source_dataset = gdal.Open(source_dataset_uri)

    band = source_dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    datatype = band.DataType

    if nodata is None:
        nodata = -9999

    gdal.Dataset.__swig_destroy__(source_dataset)
    source_dataset = None

    pixel_size = get_raster_info(source_dataset_uri)['mean_pixel_size']
    vectorize_datasets(
        [source_dataset_uri], lambda x: x, out_dataset_uri, datatype, nodata,
        pixel_size, 'intersection', aoi_uri=aoi_datasource_uri,
        assert_datasets_projected=assert_projections,
        process_pool=process_pool, vectorize_op=False, all_touched=all_touched)


def get_raster_info(raster_path):
    """Get information about a GDAL raster dataset.

    Parameters:
       raster_path (String): a path to a GDAL raster.

    Returns:
        raster_properties (dictionary): a dictionary with the properties
            stored under relevant keys.

            'pixel_size' (tuple): (pixel x-size, pixel y-size) from
                geotransform.
            'mean_pixel_size' (float): the average size of the absolute value
                of each pixel size element.
            'raster_size' (tuple):  number of raster pixels in (x, y)
                direction.
            'nodata' (float or list): if number of bands is 1, then this value
                is the nodata value of the single band, otherwise a list of
                the nodata values in increasing band index
            'n_bands' (int): number of bands in the raster.
            'geotransform' (tuple): a 6-tuple representing the geotransform of
                (x orign, x-increase, xy-increase,
                 y origin, yx-increase, y-increase),
            'datatype' (int): An instance of an enumerated gdal.GDT_* int
                that represents the datatype of the raster.
    """
    raster_properties = {}
    raster = gdal.Open(raster_path)
    geo_transform = raster.GetGeoTransform()
    raster_properties['pixel_size'] = (geo_transform[1], geo_transform[5])
    raster_properties['mean_pixel_size'] = (
        (abs(geo_transform[1]) + abs(geo_transform[5])) / 2.0)
    raster_properties['raster_size'] = (
        raster.GetRasterBand(1).XSize,
        raster.GetRasterBand(1).YSize)
    raster_properties['n_bands'] = raster.RasterCount
    raster_properties['nodata'] = [
        raster.GetRasterBand(index).GetNoDataValue() for index in range(
            1, raster_properties['n_bands']+1)]
    if len(raster_properties['nodata']) == 1:
        raster_properties['nodata'] = raster_properties['nodata'][0]

    raster_properties['bounding_box'] = [
        geo_transform[0], geo_transform[3],
        (geo_transform[0] +
         raster_properties['raster_size'][0] * geo_transform[1]),
        (geo_transform[3] +
         raster_properties['raster_size'][1] * geo_transform[5])]

    raster_properties['geotransform'] = geo_transform

    # datatype is the same for the whole raster, but is associated with band
    raster_properties['datatype'] = raster.GetRasterBand(1).DataType

    raster = None
    return raster_properties


def vectorize_datasets(
        dataset_uri_list, dataset_pixel_op, dataset_out_uri, datatype_out,
        nodata_out, pixel_size_out, bounding_box_mode,
        resample_method_list=None, dataset_to_align_index=None,
        dataset_to_bound_index=None, aoi_uri=None,
        assert_datasets_projected=True, process_pool=None, vectorize_op=True,
        datasets_are_pre_aligned=False, dataset_options=None,
        all_touched=False):
    """Apply local raster operation on stack of datasets.

    This function applies a user defined function across a stack of
    datasets.  It has functionality align the output dataset grid
    with one of the input datasets, output a dataset that is the union
    or intersection of the input dataset bounding boxes, and control
    over the interpolation techniques of the input datasets, if
    necessary.  The datasets in dataset_uri_list must be in the same
    projection; the function will raise an exception if not.

    Args:
        dataset_uri_list (list): a list of file uris that point to files that
            can be opened with gdal.Open.
        dataset_pixel_op (function) a function that must take in as many
            arguments as there are elements in dataset_uri_list.  The arguments
            can be treated as interpolated or actual pixel values from the
            input datasets and the function should calculate the output
            value for that pixel stack.  The function is a parallel
            paradigmn and does not know the spatial position of the
            pixels in question at the time of the call.  If the
            `bounding_box_mode` parameter is "union" then the values
            of input dataset pixels that may be outside their original
            range will be the nodata values of those datasets.  Known
            bug: if dataset_pixel_op does not return a value in some cases
            the output dataset values are undefined even if the function
            does not crash or raise an exception.
        dataset_out_uri (string): the uri of the output dataset.  The
            projection will be the same as the datasets in dataset_uri_list.
        datatype_out: the GDAL output type of the output dataset
        nodata_out: the nodata value of the output dataset.
        pixel_size_out: the pixel size of the output dataset in
            projected coordinates.
        bounding_box_mode (string): one of "union" or "intersection",
            "dataset". If union the output dataset bounding box will be the
            union of the input datasets.  Will be the intersection otherwise.
            An exception is raised if the mode is "intersection" and the
            input datasets have an empty intersection. If dataset it will make
            a bounding box as large as the given dataset, if given
            dataset_to_bound_index must be defined.

    Keyword Args:
        resample_method_list (list): a list of resampling methods
            for each output uri in dataset_out_uri list.  Each element
            must be one of "nearest|bilinear|cubic|cubic_spline|lanczos".
            If None, the default is "nearest" for all input datasets.
        dataset_to_align_index (int): an int that corresponds to the position
            in one of the dataset_uri_lists that, if positive aligns the output
            rasters to fix on the upper left hand corner of the output
            datasets.  If negative, the bounding box aligns the intersection/
            union without adjustment.
        dataset_to_bound_index: if mode is "dataset" this indicates which
            dataset should be the output size.
        aoi_uri (string): a URI to an OGR datasource to be used for the
            aoi.  Irrespective of the `mode` input, the aoi will be used
            to intersect the final bounding box.
        assert_datasets_projected (boolean): if True this operation will
            test if any datasets are not projected and raise an exception
            if so.
        process_pool: a process pool for multiprocessing
        vectorize_op (boolean): if true the model will try to numpy.vectorize
            dataset_pixel_op.  If dataset_pixel_op is designed to use maximize
            array broadcasting, set this parameter to False, else it may
            inefficiently invoke the function on individual elements.
        datasets_are_pre_aligned (boolean): If this value is set to False
            this operation will first align and interpolate the input datasets
            based on the rules provided in bounding_box_mode,
            resample_method_list, dataset_to_align_index, and
            dataset_to_bound_index, if set to True the input dataset list must
            be aligned, probably by raster_utils.align_dataset_list
        dataset_options: this is an argument list that will be
            passed to the GTiff driver.  Useful for blocksizes, compression,
            etc.
        all_touched (boolean): if true the clip uses the option
            ALL_TOUCHED=TRUE when calling RasterizeLayer for AOI masking.

    Returns:
        None

    Raises:
        ValueError: invalid input provided
    """
    if not isinstance(dataset_uri_list, list):
        raise ValueError(
            "dataset_uri_list was not passed in as a list, maybe a single "
            "file was passed in?  Here is its value: %s" %
            (str(dataset_uri_list)))

    if aoi_uri is None:
        assert_file_existance(dataset_uri_list)
    else:
        assert_file_existance(dataset_uri_list + [aoi_uri])

    if dataset_out_uri in dataset_uri_list:
        raise ValueError(
            "%s is used as an output file, but it is also an input file "
            "in the input list %s" % (dataset_out_uri, str(dataset_uri_list)))

    valid_bounding_box_modes = ["union", "intersection", "dataset"]
    if bounding_box_mode not in valid_bounding_box_modes:
        raise ValueError(
            "Unknown bounding box mode %s; should be one of %s",
            bounding_box_mode, valid_bounding_box_modes)

    # Create a temporary list of filenames whose files delete on the python
    # interpreter exit
    if not datasets_are_pre_aligned:
        # Handle the cases where optional arguments are passed in
        if resample_method_list is None:
            resample_method_list = ["nearest"] * len(dataset_uri_list)
        if dataset_to_align_index is None:
            dataset_to_align_index = -1
        dataset_out_uri_list = [
            temporary_filename(suffix='.tif') for _ in dataset_uri_list]
        # Align and resample the datasets, then load datasets into a list
        align_dataset_list(
            dataset_uri_list, dataset_out_uri_list, resample_method_list,
            pixel_size_out, bounding_box_mode, dataset_to_align_index,
            dataset_to_bound_index=dataset_to_bound_index,
            aoi_uri=aoi_uri,
            assert_datasets_projected=assert_datasets_projected,
            all_touched=all_touched)
        aligned_datasets = [
            gdal.Open(filename, gdal.GA_ReadOnly) for filename in
            dataset_out_uri_list]
    else:
        # otherwise the input datasets are already aligned
        aligned_datasets = [
            gdal.Open(filename, gdal.GA_ReadOnly) for filename in
            dataset_uri_list]

    aligned_bands = [dataset.GetRasterBand(1) for dataset in aligned_datasets]

    n_rows = aligned_datasets[0].RasterYSize
    n_cols = aligned_datasets[0].RasterXSize

    output_dataset = new_raster_from_base(
        aligned_datasets[0], dataset_out_uri, 'GTiff', nodata_out,
        datatype_out, dataset_options=dataset_options)
    output_band = output_dataset.GetRasterBand(1)
    block_size = output_band.GetBlockSize()
    # makes sense to get the largest block size possible to reduce the number
    # of expensive readasarray calls
    for current_block_size in [band.GetBlockSize() for band in aligned_bands]:
        if (current_block_size[0] * current_block_size[1] >
                block_size[0] * block_size[1]):
            block_size = current_block_size

    cols_per_block, rows_per_block = block_size[0], block_size[1]
    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    # If there's an AOI, mask it out
    if aoi_uri is not None:
        mask_uri = temporary_filename(suffix='.tif')
        mask_dataset = new_raster_from_base(
            aligned_datasets[0], mask_uri, 'GTiff', 255, gdal.GDT_Byte,
            fill_value=0, dataset_options=dataset_options)
        mask_band = mask_dataset.GetRasterBand(1)
        aoi_datasource = ogr.Open(aoi_uri)
        aoi_layer = aoi_datasource.GetLayer()
        if all_touched:
            option_list = ["ALL_TOUCHED=TRUE"]
        else:
            option_list = []
        gdal.RasterizeLayer(
            mask_dataset, [1], aoi_layer, burn_values=[1], options=option_list)
        aoi_layer = None
        aoi_datasource = None

    # We only want to do this if requested, otherwise we might have a more
    # efficient call if we don't vectorize.
    if vectorize_op:
        dataset_pixel_op = numpy.vectorize(
            dataset_pixel_op, otypes=[_gdal_to_numpy_type(output_band)])

    last_time = time.time()

    last_row_block_width = None
    last_col_block_width = None
    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            current_time = time.time()
            if current_time - last_time > 5.0:
                last_time = current_time

            #This is true at least once since last_* initialized with None
            if (last_row_block_width != row_block_width or
                    last_col_block_width != col_block_width):
                dataset_blocks = [
                    numpy.zeros(
                        (row_block_width, col_block_width),
                        dtype=_gdal_to_numpy_type(band)) for band in aligned_bands]

                if aoi_uri != None:
                    mask_array = numpy.zeros(
                        (row_block_width, col_block_width), dtype=numpy.int8)

                last_row_block_width = row_block_width
                last_col_block_width = col_block_width

            for dataset_index in range(len(aligned_bands)):
                aligned_bands[dataset_index].ReadAsArray(
                    xoff=col_offset, yoff=row_offset,
                    win_xsize=col_block_width,
                    win_ysize=row_block_width,
                    buf_obj=dataset_blocks[dataset_index])

            out_block = dataset_pixel_op(*dataset_blocks)

            # Mask out the row if there is a mask
            if aoi_uri is not None:
                mask_band.ReadAsArray(
                    xoff=col_offset, yoff=row_offset,
                    win_xsize=col_block_width,
                    win_ysize=row_block_width,
                    buf_obj=mask_array)
                out_block[mask_array == 0] = nodata_out

            output_band.WriteArray(
                out_block[0:row_block_width, 0:col_block_width],
                xoff=col_offset, yoff=row_offset)

    # Making sure the band and dataset is flushed and not in memory before
    # adding stats
    output_band.FlushCache()
    output_band = None
    output_dataset.FlushCache()
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None

    # Clean up the files made by temporary file because we had an issue once
    # where I was running the water yield model over 2000 times and it made
    # so many temporary files I ran out of disk space.
    if aoi_uri is not None:
        mask_band = None
        gdal.Dataset.__swig_destroy__(mask_dataset)
        mask_dataset = None
        os.remove(mask_uri)
    aligned_bands = None
    for dataset in aligned_datasets:
        gdal.Dataset.__swig_destroy__(dataset)
    aligned_datasets = None
    if not datasets_are_pre_aligned:
        # if they weren't pre-aligned then we have temporary files to remove
        for temp_dataset_uri in dataset_out_uri_list:
            try:
                os.remove(temp_dataset_uri)
            except OSError:
                pass
    calculate_raster_stats_uri(dataset_out_uri
)

def assert_file_existance(dataset_uri_list):
    """Assert that provided uris exist in filesystem.

    Verify that the uris passed in the argument exist on the filesystem
    if not, raise an exeception indicating which files do not exist

    Args:
        dataset_uri_list (list): a list of relative or absolute file paths to
            validate

    Returns:
        None

    Raises:
        IOError: if any files are not found
    """
    not_found_uris = []
    for uri in dataset_uri_list:
        if not os.path.exists(uri):
            not_found_uris.append(uri)

    if len(not_found_uris) != 0:
        error_message = (
            "The following files do not exist on the filesystem: " +
            str(not_found_uris))
        raise IOError(error_message)


def temporary_filename(suffix=''):
    """Get path to new temporary file that will be deleted on program exit.

    Returns a temporary filename using mkstemp. The file is deleted
    on exit using the atexit register.

    Keyword Args:
        suffix (string): the suffix to be appended to the temporary file

    Returns:
        fname: a unique temporary filename
    """
    file_handle, path = tempfile.mkstemp(suffix=suffix)
    os.close(file_handle)

    def remove_file(path):
        """Function to remove a file and handle exceptions to register
            in atexit."""
        try:
            os.remove(path)
        except OSError:
            # This happens if the file didn't exist, which is okay because
            # maybe we deleted it in a method
            pass

    atexit.register(remove_file, path)
    return path


def new_raster_from_base_uri(base_uri, *args, **kwargs):
    """A wrapper for the function new_raster_from_base that opens up
        the base_uri before passing it to new_raster_from_base.

        base_uri - a URI to a GDAL dataset on disk.

        All other arguments to new_raster_from_base are passed in.

        Returns nothing.
        """
    base_raster = gdal.Open(base_uri)
    if base_raster is None:
        raise IOError("%s not found when opening GDAL raster")
    new_raster = new_raster_from_base(base_raster, *args, **kwargs)

    gdal.Dataset.__swig_destroy__(new_raster)
    gdal.Dataset.__swig_destroy__(base_raster)
    new_raster = None
    base_raster = None

def new_raster_from_base(
    base, output_uri, gdal_format, nodata, datatype, fill_value=None,
    n_rows=None, n_cols=None, dataset_options=None):
    """Create a new, empty GDAL raster dataset with the spatial references,
        geotranforms of the base GDAL raster dataset.

        base - a the GDAL raster dataset to base output size, and transforms on
        output_uri - a string URI to the new output raster dataset.
        gdal_format - a string representing the GDAL file format of the
            output raster.  See http://gdal.org/formats_list.html for a list
            of available formats.  This parameter expects the format code, such
            as 'GTiff' or 'MEM'
        nodata - a value that will be set as the nodata value for the
            output raster.  Should be the same type as 'datatype'
        datatype - the pixel datatype of the output raster, for example
            gdal.GDT_Float32.  See the following header file for supported
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
        fill_value - (optional) the value to fill in the raster on creation
        n_rows - (optional) if set makes the resulting raster have n_rows in it
            if not, the number of rows of the outgoing dataset are equal to
            the base.
        n_cols - (optional) similar to n_rows, but for the columns.
        dataset_options - (optional) a list of dataset options that gets
            passed to the gdal creation driver, overrides defaults

        returns a new GDAL raster dataset."""

    #This might be a numpy type coming in, set it to native python type
    try:
        nodata = nodata.item()
    except AttributeError:
        pass

    if n_rows is None:
        n_rows = base.RasterYSize
    if n_cols is None:
        n_cols = base.RasterXSize
    projection = base.GetProjection()
    geotransform = base.GetGeoTransform()
    driver = gdal.GetDriverByName(gdal_format)

    base_band = base.GetRasterBand(1)
    block_size = base_band.GetBlockSize()
    metadata = base_band.GetMetadata('IMAGE_STRUCTURE')
    base_band = None

    if dataset_options == None:
        #make a new list to make sure we aren't ailiasing one passed in
        dataset_options = []
        #first, should it be tiled?  yes if it's not striped
        if block_size[0] != n_cols:
            #just do 256x256 blocks
            dataset_options = [
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=IF_SAFER']
        if 'PIXELTYPE' in metadata:
            dataset_options.append('PIXELTYPE=' + metadata['PIXELTYPE'])

    new_raster = driver.Create(
        output_uri.encode('utf-8'), n_cols, n_rows, 1, datatype,
        options=dataset_options)
    new_raster.SetProjection(projection)
    new_raster.SetGeoTransform(geotransform)
    band = new_raster.GetRasterBand(1)

    if nodata is not None:
        band.SetNoDataValue(nodata)
    else:
        pass

    if fill_value != None:
        band.Fill(fill_value)
    elif nodata is not None:
        band.Fill(nodata)
    band = None

    return new_raster


def get_bounding_box(dataset_uri):
    """Get bounding box where coordinates are in projected units.

    Args:
        dataset_uri (string): a uri to a GDAL dataset

    Returns:
        bounding_box (list):
            [upper_left_x, upper_left_y, lower_right_x, lower_right_y] in
            projected coordinates
    """
    dataset = gdal.Open(dataset_uri)

    geotransform = dataset.GetGeoTransform()
    n_cols = dataset.RasterXSize
    n_rows = dataset.RasterYSize

    bounding_box = [geotransform[0],
                    geotransform[3],
                    geotransform[0] + n_cols * geotransform[1],
                    geotransform[3] + n_rows * geotransform[5]]

    # Close and cleanup dataset
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return bounding_box


def align_dataset_list(
        dataset_uri_list, dataset_out_uri_list, resample_method_list,
        out_pixel_size, mode, dataset_to_align_index,
        dataset_to_bound_index=None, aoi_uri=None,
        assert_datasets_projected=True, all_touched=False):
    """Create a new list of datasets that are aligned based on a list of
        inputted datasets.

    Take a list of dataset uris and generates a new set that is completely
    aligned with identical projections and pixel sizes.

    Args:
        dataset_uri_list (list): a list of input dataset uris
        dataset_out_uri_list (list): a parallel dataset uri list whose
            positions correspond to entries in dataset_uri_list
        resample_method_list (list): a list of resampling methods for each
            output uri in dataset_out_uri list.  Each element must be one of
            "nearest|bilinear|cubic|cubic_spline|lanczos"
        out_pixel_size: the output pixel size
        mode (string): one of "union", "intersection", or "dataset" which
            defines how the output output extents are defined as either the
            union or intersection of the input datasets or to have the same
            bounds as an existing raster.  If mode is "dataset" then
            dataset_to_bound_index must be defined
        dataset_to_align_index (int): an int that corresponds to the position
            in one of the dataset_uri_lists that, if positive aligns the output
            rasters to fix on the upper left hand corner of the output
            datasets.  If negative, the bounding box aligns the intersection/
            union without adjustment.
        all_touched (boolean): if True and an AOI is passed, the
            ALL_TOUCHED=TRUE option is passed to the RasterizeLayer function
            when determining the mask of the AOI.

    Keyword Args:
        dataset_to_bound_index: if mode is "dataset" then this index is
            used to indicate which dataset to define the output bounds of the
            dataset_out_uri_list
        aoi_uri (string): a URI to an OGR datasource to be used for the
            aoi.  Irrespective of the `mode` input, the aoi will be used
            to intersect the final bounding box.

    Returns:
        None
    """
    import functools
    last_time = time.time()

    # make sure that the input lists are of the same length
    list_lengths = [
        len(dataset_uri_list), len(dataset_out_uri_list),
        len(resample_method_list)]
    if not functools.reduce(lambda x, y: x if x == y else False, list_lengths):
        raise Exception(
            "dataset_uri_list, dataset_out_uri_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths)))

    if assert_datasets_projected:
        assert_datasets_in_same_projection(dataset_uri_list)
    if mode not in ["union", "intersection", "dataset"]:
        raise Exception("Unknown mode %s" % (str(mode)))

    if dataset_to_align_index >= len(dataset_uri_list):
        raise Exception(
            "Alignment index is out of bounds of the datasets index: %s"
            "n_elements %s" % (dataset_to_align_index, len(dataset_uri_list)))
    if mode == "dataset" and dataset_to_bound_index is None:
        raise Exception(
            "Mode is 'dataset' but dataset_to_bound_index is not defined")
    if mode == "dataset" and (dataset_to_bound_index < 0 or
                              dataset_to_bound_index >= len(dataset_uri_list)):
        raise Exception(
            "dataset_to_bound_index is out of bounds of the datasets index: %s"
            "n_elements %s" % (dataset_to_bound_index, len(dataset_uri_list)))

    def merge_bounding_boxes(bb1, bb2, mode):
        """Helper function to merge two bounding boxes through union or
            intersection"""
        less_than_or_equal = lambda x, y: x if x <= y else y
        greater_than = lambda x, y: x if x > y else y

        if mode == "union":
            comparison_ops = [
                less_than_or_equal, greater_than, greater_than,
                less_than_or_equal]
        if mode == "intersection":
            comparison_ops = [
                greater_than, less_than_or_equal, less_than_or_equal,
                greater_than]

        bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
        return bb_out

    # get the intersecting or unioned bounding box
    if mode == "dataset":
        bounding_box = get_bounding_box(
            dataset_uri_list[dataset_to_bound_index])
    else:
        bounding_box = functools.reduce(
            functools.partial(merge_bounding_boxes, mode=mode),
            [get_bounding_box(dataset_uri) for dataset_uri in dataset_uri_list])

    if aoi_uri is not None:
        bounding_box = merge_bounding_boxes(
            bounding_box, get_datasource_bounding_box(aoi_uri), "intersection")

    if (bounding_box[0] >= bounding_box[2] or
            bounding_box[1] <= bounding_box[3]) and mode == "intersection":
        raise Exception("The datasets' intersection is empty "
                        "(i.e., not all the datasets touch each other).")

    if dataset_to_align_index >= 0:
        # bounding box needs alignment
        align_bounding_box = get_bounding_box(
            dataset_uri_list[dataset_to_align_index])
        align_pixel_size = get_cell_size_from_uri(
            dataset_uri_list[dataset_to_align_index])

        for index in [0, 1]:
            n_pixels = int(
                (bounding_box[index] - align_bounding_box[index]) /
                float(align_pixel_size))
            bounding_box[index] = \
                n_pixels * align_pixel_size + align_bounding_box[index]

    for original_dataset_uri, out_dataset_uri, resample_method, index in zip(
            dataset_uri_list, dataset_out_uri_list, resample_method_list,
            range(len(dataset_uri_list))):
        current_time = time.time()
        if current_time - last_time > 5.0:
            last_time = current_time

        resize_and_resample_dataset_uri(
            original_dataset_uri, bounding_box, out_pixel_size,
            out_dataset_uri, resample_method)

    # If there's an AOI, mask it out
    if aoi_uri is not None:
        first_dataset = gdal.Open(dataset_out_uri_list[0])
        n_rows = first_dataset.RasterYSize
        n_cols = first_dataset.RasterXSize
        gdal.Dataset.__swig_destroy__(first_dataset)
        first_dataset = None

        mask_uri = temporary_filename(suffix='.tif')
        new_raster_from_base_uri(
            dataset_out_uri_list[0], mask_uri, 'GTiff', 255, gdal.GDT_Byte,
            fill_value=0)

        mask_dataset = gdal.Open(mask_uri, gdal.GA_Update)
        mask_band = mask_dataset.GetRasterBand(1)
        aoi_datasource = ogr.Open(aoi_uri)
        aoi_layer = aoi_datasource.GetLayer()
        if all_touched:
            option_list = ["ALL_TOUCHED=TRUE"]
        else:
            option_list = []
        gdal.RasterizeLayer(
            mask_dataset, [1], aoi_layer, burn_values=[1], options=option_list)
        mask_row = numpy.zeros((1, n_cols), dtype=numpy.int8)

        out_dataset_list = [
            gdal.Open(uri, gdal.GA_Update) for uri in dataset_out_uri_list]
        out_band_list = [
            dataset.GetRasterBand(1) for dataset in out_dataset_list]
        nodata_out_list = [
            get_nodata_from_uri(uri) for uri in dataset_out_uri_list]

        for row_index in range(n_rows):
            mask_row = (mask_band.ReadAsArray(
                0, row_index, n_cols, 1) == 0)
            for out_band, nodata_out in zip(out_band_list, nodata_out_list):
                dataset_row = out_band.ReadAsArray(
                    0, row_index, n_cols, 1)
                out_band.WriteArray(
                    numpy.where(mask_row, nodata_out, dataset_row),
                    xoff=0, yoff=row_index)

        # Remove the mask aoi if necessary
        mask_band = None
        gdal.Dataset.__swig_destroy__(mask_dataset)
        mask_dataset = None
        os.remove(mask_uri)

        # Close and clean up datasource
        aoi_layer = None
        ogr.DataSource.__swig_destroy__(aoi_datasource)
        aoi_datasource = None

        # Clean up datasets
        out_band_list = None
        for dataset in out_dataset_list:
            dataset.FlushCache()
            gdal.Dataset.__swig_destroy__(dataset)
        out_dataset_list = None

def assert_datasets_in_same_projection(dataset_uri_list):
    """Assert that provided datasets are all in the same projection.

    Tests if datasets represented by their uris are projected and in
    the same projection and raises an exception if not.

    Args:
        dataset_uri_list (list): (description)

    Returns:
        is_true (boolean): True (otherwise exception raised)

    Raises:
        DatasetUnprojected: if one of the datasets is unprojected.
        DifferentProjections: if at least one of the datasets is in
            a different projection
    """
    dataset_list = [gdal.Open(dataset_uri) for dataset_uri in dataset_uri_list]
    dataset_projections = []

    unprojected_datasets = set()

    for dataset in dataset_list:
        projection_as_str = dataset.GetProjection()
        dataset_sr = osr.SpatialReference()
        dataset_sr.ImportFromWkt(projection_as_str)
        if not dataset_sr.IsProjected():
            unprojected_datasets.add(dataset.GetFileList()[0])
        dataset_projections.append((dataset_sr, dataset.GetFileList()[0]))

    if len(unprojected_datasets) > 0:
        pass

    for index in range(len(dataset_projections)-1):
        if not dataset_projections[index][0].IsSame(
                dataset_projections[index+1][0]):
            pass

    for dataset in dataset_list:
        # Close and clean up dataset
        gdal.Dataset.__swig_destroy__(dataset)
    dataset_list = None
    return True

def resize_and_resample_dataset_uri(
        original_dataset_uri, bounding_box, out_pixel_size, output_uri,
        resample_method):
    """Resize and resample the given dataset.

    Args:
        original_dataset_uri (string): a GDAL dataset
        bounding_box (list): [upper_left_x, upper_left_y, lower_right_x,
            lower_right_y]
        out_pixel_size: the pixel size in projected linear units
        output_uri (string): the location of the new resampled GDAL dataset
        resample_method (string): the resampling technique, one of
            "nearest|bilinear|cubic|cubic_spline|lanczos"

    Returns:
        None
    """
    resample_dict = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos
        }

    original_dataset = gdal.Open(original_dataset_uri)
    original_band = original_dataset.GetRasterBand(1)
    original_nodata = original_band.GetNoDataValue()

    if original_nodata is None:
        original_nodata = -9999

    original_sr = osr.SpatialReference()
    original_sr.ImportFromWkt(original_dataset.GetProjection())

    output_geo_transform = [
        bounding_box[0], out_pixel_size, 0.0, bounding_box[1], 0.0,
        -out_pixel_size]
    new_x_size = abs(
        int(numpy.round((bounding_box[2] - bounding_box[0]) / out_pixel_size)))
    new_y_size = abs(
        int(numpy.round((bounding_box[3] - bounding_box[1]) / out_pixel_size)))

    if new_x_size == 0:
        new_x_size = 1
    if new_y_size == 0:
        new_y_size = 1

    # create the new x and y size
    block_size = original_band.GetBlockSize()
    # If the original band is tiled, then its x blocksize will be different
    # than the number of columns
    if original_band.XSize > 256 and original_band.YSize > 256:
        # it makes sense for many functions to have 256x256 blocks
        block_size[0] = 256
        block_size[1] = 256
        gtiff_creation_options = [
            'TILED=YES', 'BIGTIFF=IF_SAFER', 'BLOCKXSIZE=%d' % block_size[0],
            'BLOCKYSIZE=%d' % block_size[1]]

        metadata = original_band.GetMetadata('IMAGE_STRUCTURE')
        if 'PIXELTYPE' in metadata:
            gtiff_creation_options.append('PIXELTYPE=' + metadata['PIXELTYPE'])
    else:
        # it is so small or strangely aligned, use the default creation options
        gtiff_creation_options = []

    create_directories([os.path.dirname(output_uri)])
    gdal_driver = gdal.GetDriverByName('GTiff')
    output_dataset = gdal_driver.Create(
        output_uri, new_x_size, new_y_size, 1, original_band.DataType,
        options=gtiff_creation_options)
    output_band = output_dataset.GetRasterBand(1)

    output_band.SetNoDataValue(original_nodata)

    # Set the geotransform
    output_dataset.SetGeoTransform(output_geo_transform)
    output_dataset.SetProjection(original_sr.ExportToWkt())

    # need to make this a closure so we get the current time and we can affect
    # state
    def reproject_callback(df_complete, psz_message, p_progress_arg):
        """The argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - reproject_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and reproject_callback.total_time >= 5.0)):

                reproject_callback.last_time = current_time
                reproject_callback.total_time += current_time
        except AttributeError:
            reproject_callback.last_time = time.time()
            reproject_callback.total_time = 0.0

    # Perform the projection/resampling
    gdal.ReprojectImage(
        original_dataset, output_dataset, original_sr.ExportToWkt(),
        original_sr.ExportToWkt(), resample_dict[resample_method], 0, 0,
        reproject_callback, [output_uri])

    # Make sure the dataset is closed and cleaned up
    original_band = None
    gdal.Dataset.__swig_destroy__(original_dataset)
    original_dataset = None

    output_dataset.FlushCache()
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None
    calculate_raster_stats_uri(output_uri)


def create_directories(directory_list):
    """Make directories provided in list of path strings.

    This function will create any of the directories in the directory list
    if possible and raise exceptions if something exception other than
    the directory previously existing occurs.

    Args:
        directory_list (list): a list of string uri paths

    Returns:
        None
    """
    for dir_name in directory_list:
        try:
            os.makedirs(dir_name)
        except OSError as exception:
            #It's okay if the directory already exists, if it fails for
            #some other reason, raise that exception
            if (exception.errno != errno.EEXIST and
                    exception.errno != errno.ENOENT):
                raise

def get_datasource_bounding_box(datasource_uri):
    """Get datasource bounding box where coordinates are in projected units.

    Args:
        dataset_uri (string): a uri to a GDAL dataset

    Returns:
        bounding_box (list):
            [upper_left_x, upper_left_y, lower_right_x, lower_right_y] in
            projected coordinates
    """
    datasource = ogr.Open(datasource_uri)
    layer = datasource.GetLayer(0)
    extent = layer.GetExtent()
    # Reindex datasource extents into the upper left/lower right coordinates
    bounding_box = [extent[0],
                    extent[3],
                    extent[1],
                    extent[2]]
    return bounding_boxz


def iterblocks(
        raster_path, band_index_list=None, largest_block=_LARGEST_ITERBLOCK,
        astype=None, offset_only=False):
    """Iterate across all the memory blocks in the input raster.

    Result is a generator of block location information and numpy arrays.

    This is especially useful when a single value needs to be derived from the
    pixel values in a raster, such as the sum total of all pixel values, or
    a sequence of unique raster values.  In such cases, `raster_local_op`
    is overkill, since it writes out a raster.

    As a generator, this can be combined multiple times with itertools.izip()
    to iterate 'simultaneously' over multiple rasters, though the user should
    be careful to do so only with prealigned rasters.

    Parameters:
        raster_path (string): Path to raster file to iterate over.
        band_index_list (list of ints or None): A list of the bands for which
            the matrices should be returned. The band number to operate on.
            Defaults to None, which will return all bands.  Bands may be
            specified in any order, and band indexes may be specified multiple
            times.  The blocks returned on each iteration will be in the order
            specified in this list.
        largest_block (int): Attempts to iterate over raster blocks with
            this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        astype (list of numpy types): If none, output blocks are in the native
            type of the raster bands.  Otherwise this parameter is a list
            of len(band_index_list) length that contains the desired output
            types that iterblock generates for each band.
        offset_only (boolean): defaults to False, if True `iterblocks` only
            returns offset dictionary and doesn't read any binary data from
            the raster.  This can be useful when iterating over writing to
            an output.

    Returns:
        If `offset_only` is false, on each iteration, a tuple containing a dict
        of block data and `n` 2-dimensional numpy arrays are returned, where
        `n` is the number of bands requested via `band_list`. The dict of
        block data has these attributes:

            data['xoff'] - The X offset of the upper-left-hand corner of the
                block.
            data['yoff'] - The Y offset of the upper-left-hand corner of the
                block.
            data['win_xsize'] - The width of the block.
            data['win_ysize'] - The height of the block.

        If `offset_only` is True, the function returns only the block offset
            data and does not attempt to read binary data from the raster.
    """
    raster = gdal.OpenEx(raster_path)

    if band_index_list is None:
        band_index_list = range(1, raster.RasterCount + 1)

    band_index_list = [
        raster.GetRasterBand(index) for index in band_index_list]

    block = band_index_list[0].GetBlockSize()
    cols_per_block = block[0]
    rows_per_block = block[1]

    n_cols = raster.RasterXSize
    n_rows = raster.RasterYSize

    block_area = cols_per_block * rows_per_block
    # try to make block wider
    if largest_block / block_area > 0:
        width_factor = largest_block / block_area
        cols_per_block *= width_factor
        if cols_per_block > n_cols:
            cols_per_block = n_cols
        block_area = cols_per_block * rows_per_block
    # try to make block taller
    if largest_block / block_area > 0:
        height_factor = largest_block / block_area
        rows_per_block *= height_factor
        if rows_per_block > n_rows:
            rows_per_block = n_rows

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    # Initialize to None so a block array is created on the first iteration
    last_row_block_width = None
    last_col_block_width = None

    if astype is not None:
        block_type_list = [astype] * len(band_index_list)
    else:
        block_type_list = [
            _gdal_to_numpy_type(ds_band) for ds_band in band_index_list]

    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # resize the raster block cache if necessary
            if (last_row_block_width != row_block_width or
                    last_col_block_width != col_block_width):
                raster_blocks = [
                    numpy.zeros(
                        (row_block_width, col_block_width),
                        dtype=block_type) for block_type in
                    block_type_list]

            offset_dict = {
                'xoff': col_offset,
                'yoff': row_offset,
                'win_xsize': col_block_width,
                'win_ysize': row_block_width,
            }
            result = offset_dict
            if not offset_only:
                for ds_band, block in zip(band_index_list, raster_blocks):
                    ds_band.ReadAsArray(buf_obj=block, **offset_dict)
                result = (result,) + tuple(raster_blocks)
            yield result


def get_vector_info(vector_path, layer_index=0):
    """Get information about an OGR vector (datasource).

    Parameters:
        vector_path (str): a path to a OGR vector.
        layer_index (int): index of underlying layer to analyze.  Defaults to
            0.

    Returns:
        raster_properties (dictionary): a dictionary with the following
            properties stored under relevant keys.

            'projection' (string): projection of the vector in Well Known
                Text.
            'bounding_box' (list): list of floats representing the bounding
                box in projected coordinates as [minx, miny, maxx, maxy].
    """
    vector = gdal.OpenEx(vector_path)
    vector_properties = {}
    layer = vector.GetLayer(iLayer=layer_index)
    # projection is same for all layers, so just use the first one
    vector_properties['projection'] = layer.GetSpatialRef().ExportToWkt()
    layer_bb = layer.GetExtent()
    layer = None
    vector = None
    # convert form [minx,maxx,miny,maxy] to [minx,miny,maxx,maxy]
    vector_properties['bounding_box'] = [layer_bb[i] for i in [0, 2, 1, 3]]
    return vector_properties


def _merge_bounding_boxes(bb1, bb2, mode):
    """Merge two bounding boxes through union or intersection.

    Parameters:
        bb1, bb2 (list): list of float representing bounding box in the
            form bb=[minx,miny,maxx,maxy]
        mode (string); one of 'union' or 'intersection'

    Returns:
        Reduced bounding box of bb1/bb2 depending on mode.
    """
    def _less_than_or_equal(x_val, y_val):
        return x_val if x_val <= y_val else y_val

    def _greater_than(x_val, y_val):
        return x_val if x_val > y_val else y_val

    if mode == "union":
        comparison_ops = [
            _less_than_or_equal, _less_than_or_equal,
            _greater_than, _greater_than]
    if mode == "intersection":
        comparison_ops = [
            _greater_than, _greater_than,
            _less_than_or_equal, _less_than_or_equal]

    bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
    return bb_out


def _invoke_timed_callback(
        reference_time, callback_lambda, callback_period):
    """Invoke callback if a certain amount of time has passed.

    This is a convenience function to standardize update callbacks from the
    module.

    Parameters:
        reference_time (float): time to base `callback_period` length from.
        callback_lambda (lambda): function to invoke if difference between
            current time and `reference_time` has exceeded `callback_period`.
        callback_period (float): time in seconds to pass until
            `callback_lambda` is invoked.

    Returns:
        `reference_time` if `callback_lambda` not invoked, otherwise the time
        when `callback_lambda` was invoked.
    """
    current_time = time.time()
    if current_time - reference_time > callback_period:
        callback_lambda()
        return current_time
    return reference_time


def align_and_resize_raster_stack(
        base_raster_path_list, target_raster_path_list, resample_method_list,
        target_pixel_size, bounding_box_mode, base_vector_path_list=None,
        raster_align_index=None,
        gtiff_creation_options=_DEFAULT_GTIFF_CREATION_OPTIONS):
    """Generate rasters from a base such that they align geospatially.

    This function resizes base rasters that are in the same geospatial
    projection such that the result is an aligned stack of rasters that have
    the same cell size, dimensions, and bounding box. This is achieved by
    clipping or resizing the rasters to intersected, unioned, or equivocated
    bounding boxes of all the raster and vector input.

    Parameters:
        base_raster_path_list (list): a list of base raster paths that will
            be transformed and will be used to determine the target bounding
            box.
        target_raster_path_list (list): a list of raster paths that will be
            created to one-to-one map with `base_raster_path_list` as aligned
            versions of those original rasters.
        resample_method_list (list): a list of resampling methods which
            one to one map each path in `base_raster_path_list` during
            resizing.  Each element must be one of
            "nearest|bilinear|cubic|cubic_spline|lanczos|mode".
        target_pixel_size (tuple): the target raster's x and y pixel size
            example: [30, -30].
        bounding_box_mode (string): one of "union", "intersection", or
            a list of floats of the form [minx, miny, maxx, maxy].  Depending
            on the value, output extents are defined as the union,
            intersection, or the explicit bounding box.
        base_vector_path_list (list): a list of base vector paths whose
            bounding boxes will be used to determine the final bounding box
            of the raster stack if mode is 'union' or 'intersection'.  If mode
            is 'bb=[...]' then these vectors are not used in any calculation.
        raster_align_index (int): indicates the index of a
            raster in `base_raster_path_list` that the target rasters'
            bounding boxes pixels should align with.  This feature allows
            rasters whose raster dimensions are the same, but bounding boxes
            slightly shifted less than a pixel size to align with a desired
            grid layout.  If `None` then the bounding box of the target
            rasters is calculated as the precise intersection, union, or
            bounding box.
        gtiff_creation_options (list): list of strings that will be passed
            as GDAL "dataset" creation options to the GTIFF driver, or ignored
            if None.

    Returns:
        None
    """
    last_time = time.time()

    # make sure that the input lists are of the same length
    list_lengths = [
        len(base_raster_path_list), len(target_raster_path_list),
        len(resample_method_list)]
    if len(set(list_lengths)) != 1:
        raise ValueError(
            "base_raster_path_list, target_raster_path_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths)))

    # we can accept 'union', 'intersection', or a 4 element list/tuple
    if bounding_box_mode not in ["union", "intersection"] and (
            not isinstance(bounding_box_mode, (list, tuple)) or
            len(bounding_box_mode) != 4):
        raise ValueError("Unknown bounding_box_mode %s" % (
            str(bounding_box_mode)))

    if ((raster_align_index is not None) and
            ((raster_align_index < 0) or
             (raster_align_index >= len(base_raster_path_list)))):
        raise ValueError(
            "Alignment index is out of bounds of the datasets index: %s"
            " n_elements %s" % (
                raster_align_index, len(base_raster_path_list)))

    raster_info_list = [
        get_raster_info(path) for path in base_raster_path_list]
    if base_vector_path_list is not None:
        vector_info_list = [
            get_vector_info(path) for path in base_vector_path_list]
    else:
        vector_info_list = []

    # get the literal or intersecting/unioned bounding box
    if isinstance(bounding_box_mode, (list, tuple)):
        target_bounding_box = bounding_box_mode
    else:
        # either intersection or union
        from functools import reduce
        target_bounding_box = reduce(
            functools.partial(_merge_bounding_boxes, mode=bounding_box_mode),
            [info['bounding_box'] for info in
             (raster_info_list + vector_info_list)])

    if bounding_box_mode == "intersection" and (
            target_bounding_box[0] > target_bounding_box[2] or
            target_bounding_box[1] > target_bounding_box[3]):
        raise ValueError("The rasters' and vectors' intersection is empty "
                         "(not all rasters and vectors touch each other).")

    if raster_align_index >= 0:
        # bounding box needs alignment
        align_bounding_box = (
            raster_info_list[raster_align_index]['bounding_box'])
        align_pixel_size = (
            raster_info_list[raster_align_index]['pixel_size'])
        # adjust bounding box so lower left corner aligns with a pixel in
        # raster[raster_align_index]
        for index in [0, 1]:
            n_pixels = int(
                (target_bounding_box[index] - align_bounding_box[index]) /
                float(align_pixel_size[index]))
            target_bounding_box[index] = (
                n_pixels * align_pixel_size[index] +
                align_bounding_box[index])

    for index, (base_path, target_path, resample_method) in enumerate(zip(
            base_raster_path_list, target_raster_path_list,
            resample_method_list)):
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "align_dataset_list aligning dataset %d of %d",
                index, len(base_raster_path_list)), _LOGGING_PERIOD)
        warp_raster(
            base_path, target_pixel_size,
            target_path, resample_method,
            target_bb=target_bounding_box,
            gtiff_creation_options=gtiff_creation_options)


def warp_raster(
        base_raster_path, target_pixel_size, target_raster_path,
        resample_method, target_bb=None, target_sr_wkt=None,
        gtiff_creation_options=_DEFAULT_GTIFF_CREATION_OPTIONS):
    """Resize/resample raster to desired pixel size, bbox and projection.

    Parameters:
        base_raster_path (string): path to base raster.
        target_pixel_size (list): a two element list or tuple indicating the
            x and y pixel size in projected units.
        target_raster_path (string): the location of the resized and
            resampled raster.
        resample_method (string): the resampling technique, one of
            "nearest|bilinear|cubic|cubic_spline|lanczos|mode"
        target_bb (list): if None, target bounding box is the same as the
            source bounding box.  Otherwise it's a list of float describing
            target bounding box in target coordinate system as
            [minx, miny, maxx, maxy].
        target_sr_wkt (string): if not None, desired target projection in Well
            Known Text format.
        gtiff_creation_options (list or tuple): list of strings that will be
            passed as GDAL "dataset" creation options to the GTIFF driver.

    Returns:
        None
    """
    base_raster = gdal.OpenEx(base_raster_path)
    base_sr = osr.SpatialReference()
    base_sr.ImportFromWkt(base_raster.GetProjection())

    if target_bb is None:
        target_bb = get_raster_info(base_raster_path)['bounding_box']
        # transform the target_bb if target_sr_wkt is not None
        if target_sr_wkt is not None:
            target_bb = transform_bounding_box(
                get_raster_info(base_raster_path)['bounding_box'],
                get_raster_info(base_raster_path)['projection'],
                target_sr_wkt)

    target_geotransform = [
        target_bb[0], target_pixel_size[0], 0.0, target_bb[1], 0.0,
        target_pixel_size[1]]
    # this handles a case of a negative pixel size in which case the raster
    # row will increase downward
    if target_pixel_size[0] < 0:
        target_geotransform[0] = target_bb[2]
    if target_pixel_size[1] < 0:
        target_geotransform[3] = target_bb[3]
    target_x_size = abs((target_bb[2] - target_bb[0]) / target_pixel_size[0])
    target_y_size = abs((target_bb[3] - target_bb[1]) / target_pixel_size[1])

    if target_x_size - int(target_x_size) > 0:
        target_x_size = int(target_x_size) + 1
    else:
        target_x_size = int(target_x_size)

    if target_y_size - int(target_y_size) > 0:
        target_y_size = int(target_y_size) + 1
    else:
        target_y_size = int(target_y_size)

    if target_x_size == 0:
        LOGGER.warn(
            "bounding_box is so small that x dimension rounds to 0; "
            "clamping to 1.")
        target_x_size = 1
    if target_y_size == 0:
        LOGGER.warn(
            "bounding_box is so small that y dimension rounds to 0; "
            "clamping to 1.")
        target_y_size = 1

    local_gtiff_creation_options = list(gtiff_creation_options)
    # PIXELTYPE is sometimes used to define signed vs. unsigned bytes and
    # the only place that is stored is in the IMAGE_STRUCTURE metadata
    # copy it over if it exists; get this info from the first band since
    # all bands have the same datatype
    base_band = base_raster.GetRasterBand(1)
    metadata = base_band.GetMetadata('IMAGE_STRUCTURE')
    if 'PIXELTYPE' in metadata:
        local_gtiff_creation_options.append(
            'PIXELTYPE=' + metadata['PIXELTYPE'])

    # make directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(target_raster_path))
    except OSError:
        pass
    gdal_driver = gdal.GetDriverByName('GTiff')
    target_raster = gdal_driver.Create(
        target_raster_path, target_x_size, target_y_size,
        base_raster.RasterCount, base_band.DataType,
        options=local_gtiff_creation_options)
    base_band = None

    for index in range(target_raster.RasterCount):
        base_nodata = base_raster.GetRasterBand(1+index).GetNoDataValue()
        if base_nodata is not None:
            target_band = target_raster.GetRasterBand(1+index)
            target_band.SetNoDataValue(base_nodata)
            target_band = None

    # Set the geotransform
    target_raster.SetGeoTransform(target_geotransform)
    if target_sr_wkt is None:
        target_sr_wkt = base_sr.ExportToWkt()
    target_raster.SetProjection(target_sr_wkt)

    # need to make this a closure so we get the current time and we can affect
    # state
    reproject_callback = _make_logger_callback(
        "ReprojectImage %.1f%% complete %s, psz_message '%s'")

    # Perform the projection/resampling
    gdal.ReprojectImage(
        base_raster, target_raster, base_sr.ExportToWkt(),
        target_sr_wkt, _RESAMPLE_DICT[resample_method], 0, 0,
        reproject_callback, [target_raster_path])

    target_raster = None
    base_raster = None
    calculate_raster_stats(target_raster_path)


def transform_bounding_box(
        bounding_box, base_ref_wkt, target_ref_wkt, edge_samples=11):
    """Transform input bounding box to output projection.

    This transform accounts for the fact that the reprojected square bounding
    box might be warped in the new coordinate system.  To account for this,
    the function samples points along the original bounding box edges and
    attempts to make the largest bounding box around any transformed point
    on the edge whether corners or warped edges.

    Parameters:
        bounding_box (list): a list of 4 coordinates in `base_epsg` coordinate
            system describing the bound in the order [xmin, ymin, xmax, ymax]
        base_ref_wkt (string): the spatial reference of the input coordinate
            system in Well Known Text.
        target_ref_wkt (string): the spatial reference of the desired output
            coordinate system in Well Known Text.
        edge_samples (int): the number of interpolated points along each
            bounding box edge to sample along. A value of 2 will sample just
            the corners while a value of 3 will also sample the corners and
            the midpoint.

    Returns:
        A list of the form [xmin, ymin, xmax, ymax] that describes the largest
        fitting bounding box around the original warped bounding box in
        `new_epsg` coordinate system.
    """
    base_ref = osr.SpatialReference()
    base_ref.ImportFromWkt(base_ref_wkt)

    target_ref = osr.SpatialReference()
    target_ref.ImportFromWkt(target_ref_wkt)

    transformer = osr.CoordinateTransformation(base_ref, target_ref)

    def _transform_point(point):
        """Transform an (x,y) point tuple from base_ref to target_ref."""
        trans_x, trans_y, _ = (transformer.TransformPoint(*point))
        return (trans_x, trans_y)

    # The following list comprehension iterates over each edge of the bounding
    # box, divides each edge into `edge_samples` number of points, then
    # reduces that list to an appropriate `bounding_fn` given the edge.
    # For example the left edge needs to be the minimum x coordinate so
    # we generate `edge_samples` number of points between the upper left and
    # lower left point, transform them all to the new coordinate system
    # then get the minimum x coordinate "min(p[0] ...)" of the batch.
    # points are numbered from 0 starting upper right as follows:
    # 0--3
    # |  |
    # 1--2
    p_0 = numpy.array((bounding_box[0], bounding_box[3]))
    p_1 = numpy.array((bounding_box[0], bounding_box[1]))
    p_2 = numpy.array((bounding_box[2], bounding_box[1]))
    p_3 = numpy.array((bounding_box[2], bounding_box[3]))
    transformed_bounding_box = [
        bounding_fn(
            [_transform_point(
                p_a * v + p_b * (1 - v)) for v in numpy.linspace(
                    0, 1, edge_samples)])
        for p_a, p_b, bounding_fn in [
            (p_0, p_1, lambda p_list: min([p[0] for p in p_list])),
            (p_1, p_2, lambda p_list: min([p[1] for p in p_list])),
            (p_2, p_3, lambda p_list: max([p[0] for p in p_list])),
            (p_3, p_0, lambda p_list: max([p[1] for p in p_list]))]]
    return transformed_bounding_box


def _make_logger_callback(message):
    """Build a timed logger callback that prints `message` replaced.

    Parameters:
        message (string): a string that expects 3 placement %% variables,
            first for % complete from `df_complete`, second `psz_message`
            and last is `p_progress_arg[0]`.

    Returns:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)
    """
    def logger_callback(df_complete, psz_message, p_progress_arg):
        """The argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - logger_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and
                     logger_callback.total_time >= 5.0)):
                LOGGER.info(
                    message, df_complete * 100, p_progress_arg[0],
                    psz_message)
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0

    return logger_callback


def _is_raster_path_band_formatted(raster_path_band):
    """Returns true if raster path band is a (str, int) tuple/list."""
    import types

    if not isinstance(raster_path_band, (list, tuple)):
        return False
    elif len(raster_path_band) != 2:
        return False
    elif not isinstance(raster_path_band[0], types.StringTypes):
        return False
    elif not isinstance(raster_path_band[1], int):
        return False
    else:
        return True


def zonal_statistics(
        base_raster_path_band, aggregate_vector_path,
        aggregate_field_name, aggregate_layer_name=None,
        ignore_nodata=True, all_touched=False, polygons_might_overlap=True,
        working_dir=None):
    """Collect stats on pixel values which lie within polygons.

    This function summarizes raster statistics including min, max,
    mean, stddev, and pixel count over the regions on the raster that are
    overlaped by the polygons in the vector layer.  This function can
    handle cases where polygons overlap, which is notable since zonal stats
    functions provided by ArcGIS or QGIS usually incorrectly aggregate
    these areas.  Overlap avoidance is achieved by calculating a minimal set
    of disjoint non-overlapping polygons from `aggregate_vector_path` and
    rasterizing each set separately during the raster aggregation phase.  That
    set of rasters are then used to calculate the zonal stats of all polygons
    without aggregating vector overlap.

    Parameters:
        base_raster_path_band (tuple): a str/int tuple indicating the path to
            the base raster and the band index of that raster to analyze.
        aggregate_vector_path (string): a path to an ogr compatable polygon
            vector whose geometric features indicate the areas over
            `base_raster_path_band` to calculate statistics over.
        aggregate_field_name (string): field name in `aggregate_vector_path`
            that represents an identifying value for a given polygon. Result
            of this function will be indexed by the values found in this
            field.
        aggregate_layer_name (string): name of shapefile layer that will be
            used to aggregate results over.  If set to None, the first layer
            in the DataSource will be used as retrieved by `.GetLayer()`.
            Note: it is normal and expected to set this field at None if the
            aggregating shapefile is a single layer as many shapefiles,
            including the common 'ESRI Shapefile', are.
        ignore_nodata: if true, then nodata pixels are not accounted for when
            calculating min, max, count, or mean.  However, the value of
            `nodata_count` will always be the number of nodata pixels
            aggregated under the polygon.
        all_touched (boolean): if true will account for any pixel whose
            geometry passes through the pixel, not just the center point.
        polygons_might_overlap (boolean): if True the function calculates
            aggregation coverage close to optimally by rasterizing sets of
            polygons that don't overlap.  However, this step can be
            computationally expensive for cases where there are many polygons.
            Setting this flag to False directs the function rasterize in one
            step.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.

    Returns:
        nested dictionary indexed by aggregating feature id, and then by one
        of 'min' 'max' 'sum' 'mean' 'count' and 'nodata_count'.  Example:
        {0: {'min': 0, 'max': 1, 'mean': 0.5, 'count': 2, 'nodata_count': 1}}
    """
    import uuid
    import shutil

    if not _is_raster_path_band_formatted(base_raster_path_band):
        raise ValueError(
            "`base_raster_path_band` not formatted as expected.  Expects "
            "(path, band_index), recieved %s" + base_raster_path_band)
    aggregate_vector = gdal.OpenEx(aggregate_vector_path)
    if aggregate_layer_name is not None:
        aggregate_layer = aggregate_vector.GetLayerByName(
            aggregate_layer_name)
    else:
        aggregate_layer = aggregate_vector.GetLayer()
    aggregate_layer_defn = aggregate_layer.GetLayerDefn()
    aggregate_field_index = aggregate_layer_defn.GetFieldIndex(
        aggregate_field_name)
    if aggregate_field_index == -1:  # -1 returned when field does not exist.
        # Raise exception if user provided a field that's not in vector
        raise ValueError(
            'Vector %s must have a field named %s' %
            (aggregate_vector_path, aggregate_field_name))

    aggregate_field_def = aggregate_layer_defn.GetFieldDefn(
        aggregate_field_index)

    # create a new aggregate ID field to map base vector aggregate fields to
    # local ones that are guaranteed to be integers.
    local_aggregate_field_name = str(uuid.uuid4())[-8:-1]
    local_aggregate_field_def = ogr.FieldDefn(
        local_aggregate_field_name, ogr.OFTInteger)

    # Adding the rasterize by attribute option
    rasterize_layer_args = {
        'options': [
            'ALL_TOUCHED=%s' % str(all_touched).upper(),
            'ATTRIBUTE=%s' % local_aggregate_field_name]
        }

    # clip base raster to aggregating vector intersection
    raster_info = get_raster_info(base_raster_path_band[0])
    # -1 here because bands are 1 indexed
    raster_nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    with tempfile.NamedTemporaryFile(
            prefix='clipped_raster', delete=False,
            dir=working_dir) as clipped_raster_file:
        clipped_raster_path = clipped_raster_file.name
    align_and_resize_raster_stack(
        [base_raster_path_band[0]], [clipped_raster_path], ['nearest'],
        raster_info['pixel_size'], 'intersection',
        base_vector_path_list=[aggregate_vector_path], raster_align_index=0)
    clipped_raster = gdal.OpenEx(clipped_raster_path)

    # make a shapefile that non-overlapping layers can be added to
    driver = ogr.GetDriverByName('ESRI Shapefile')
    disjoint_vector_dir = tempfile.mkdtemp(dir=working_dir)
    disjoint_vector = driver.CreateDataSource(
        os.path.join(disjoint_vector_dir, 'disjoint_vector.shp'))
    spat_ref = aggregate_layer.GetSpatialRef()

    # Initialize these dictionaries to have the shapefile fields in the
    # original datasource even if we don't pick up a value later
    base_to_local_aggregate_value = {}
    for feature in aggregate_layer:
        aggregate_field_value = feature.GetField(aggregate_field_name)
        # this builds up a map of aggregate field values to unique ids
        if aggregate_field_value not in base_to_local_aggregate_value:
            base_to_local_aggregate_value[aggregate_field_value] = len(
                base_to_local_aggregate_value)
    aggregate_layer.ResetReading()

    # Loop over each polygon and aggregate
    if polygons_might_overlap:
        minimal_polygon_sets = calculate_disjoint_polygon_set(
            aggregate_vector_path)
    else:
        minimal_polygon_sets = [
            set([feat.GetFID() for feat in aggregate_layer])]

    clipped_band = clipped_raster.GetRasterBand(base_raster_path_band[1])

    with tempfile.NamedTemporaryFile(
            prefix='aggregate_id_raster',
            delete=False, dir=working_dir) as aggregate_id_raster_file:
        aggregate_id_raster_path = aggregate_id_raster_file.name

    aggregate_id_nodata = len(base_to_local_aggregate_value)
    new_raster_from_base(
        clipped_raster_path, aggregate_id_raster_path, gdal.GDT_Int32,
        [aggregate_id_nodata])
    aggregate_id_raster = gdal.OpenEx(aggregate_id_raster_path, gdal.GA_Update)
    aggregate_stats = {}

    for polygon_set in minimal_polygon_sets:
        disjoint_layer = disjoint_vector.CreateLayer(
            'disjoint_vector', spat_ref, ogr.wkbPolygon)
        disjoint_layer.CreateField(local_aggregate_field_def)
        # add polygons to subset_layer
        for index, poly_fid in enumerate(polygon_set):
            poly_feat = aggregate_layer.GetFeature(poly_fid)
            disjoint_layer.CreateFeature(poly_feat)
            # we seem to need to reload the feature and set the index because
            # just copying over the feature left indexes as all 0s.  Not sure
            # why.
            new_feat = disjoint_layer.GetFeature(index)
            new_feat.SetField(
                local_aggregate_field_name, base_to_local_aggregate_value[
                    poly_feat.GetField(aggregate_field_name)])
            disjoint_layer.SetFeature(new_feat)
        disjoint_layer.SyncToDisk()

        # nodata out the mask
        aggregate_id_band = aggregate_id_raster.GetRasterBand(1)
        aggregate_id_band.Fill(aggregate_id_nodata)
        aggregate_id_band = None

        gdal.RasterizeLayer(
            aggregate_id_raster, [1], disjoint_layer, **rasterize_layer_args)
        aggregate_id_raster.FlushCache()

        # Delete the features we just added to the subset_layer
        disjoint_layer = None
        disjoint_vector.DeleteLayer(0)

        # create a key array
        # and parallel min, max, count, and nodata count arrays
        for aggregate_id_offsets, aggregate_id_block in iterblocks(
                aggregate_id_raster_path):
            clipped_block = clipped_band.ReadAsArray(**aggregate_id_offsets)
            # guard against a None nodata type
            valid_mask = numpy.ones(aggregate_id_block.shape, dtype=bool)
            if aggregate_id_nodata is not None:
                valid_mask[:] = aggregate_id_block != aggregate_id_nodata
            valid_aggregate_id = aggregate_id_block[valid_mask]
            valid_clipped = clipped_block[valid_mask]
            for aggregate_id in numpy.unique(valid_aggregate_id):
                aggregate_mask = valid_aggregate_id == aggregate_id
                masked_clipped_block = valid_clipped[aggregate_mask]
                clipped_nodata_mask = (masked_clipped_block == raster_nodata)
                if aggregate_id not in aggregate_stats:
                    aggregate_stats[aggregate_id] = {
                        'min': None,
                        'max': None,
                        'count': 0,
                        'nodata_count': 0,
                        'sum': 0.0
                    }
                aggregate_stats[aggregate_id]['nodata_count'] += (
                    numpy.count_nonzero(clipped_nodata_mask))
                if ignore_nodata:
                    masked_clipped_block = (
                        masked_clipped_block[~clipped_nodata_mask])
                if masked_clipped_block.size == 0:
                    continue

                if aggregate_stats[aggregate_id]['min'] is None:
                    aggregate_stats[aggregate_id]['min'] = (
                        masked_clipped_block[0])
                    aggregate_stats[aggregate_id]['max'] = (
                        masked_clipped_block[0])

                aggregate_stats[aggregate_id]['min'] = min(
                    numpy.min(masked_clipped_block),
                    aggregate_stats[aggregate_id]['min'])
                aggregate_stats[aggregate_id]['max'] = max(
                    numpy.max(masked_clipped_block),
                    aggregate_stats[aggregate_id]['max'])
                aggregate_stats[aggregate_id]['count'] += (
                    masked_clipped_block.size)
                aggregate_stats[aggregate_id]['sum'] += numpy.sum(
                    masked_clipped_block)

    # clean up temporary files
    clipped_band = None
    clipped_raster = None
    aggregate_id_raster = None
    disjoint_layer = None
    disjoint_vector = None
    for filename in [aggregate_id_raster_path, clipped_raster_path]:
        os.remove(filename)
    shutil.rmtree(disjoint_vector_dir)

    # map the local ids back to the original base value
    local_to_base_aggregate_value = {
        value: key for key, value in
        base_to_local_aggregate_value.iteritems()}

    return {
        local_to_base_aggregate_value[key]: value
        for key, value in aggregate_stats.iteritems()}

def calculate_disjoint_polygon_set(vector_path, layer_index=0):
    """Create a list of sets of polygons that don't overlap.

    Determining the minimal number of those sets is an np-complete problem so
    this is an approximation that builds up sets of maximal subsets.

    Parameters:
        vector_path (string): a path to an OGR vector.
        layer_index (int): index of underlying layer in `vector_path` to
            calculate disjoint set. Defaults to 0.

    Returns:
        subset_list (list): list of sets of FIDs from vector_path
    """
    import heapq
    import shapely.wkt
    import shapely.ops
    from shapely import speedups
    import shapely.prepared

    vector = gdal.OpenEx(vector_path)
    vector_layer = vector.GetLayer()

    poly_intersect_lookup = {}
    for poly_feat in vector_layer:
        poly_wkt = poly_feat.GetGeometryRef().ExportToWkt()
        shapely_polygon = shapely.wkt.loads(poly_wkt)
        poly_wkt = None
        poly_fid = poly_feat.GetFID()
        poly_intersect_lookup[poly_fid] = {
            'poly': shapely_polygon,
            'intersects': set(),
        }
    vector_layer = None
    vector = None

    for poly_fid in poly_intersect_lookup:
        polygon = shapely.prepared.prep(
            poly_intersect_lookup[poly_fid]['poly'])
        for intersect_poly_fid in poly_intersect_lookup:
            if intersect_poly_fid == poly_fid or polygon.intersects(
                    poly_intersect_lookup[intersect_poly_fid]['poly']):
                poly_intersect_lookup[poly_fid]['intersects'].add(
                    intersect_poly_fid)
        polygon = None

    # Build maximal subsets
    subset_list = []
    while len(poly_intersect_lookup) > 0:
        # sort polygons by increasing number of intersections
        heap = []
        for poly_fid, poly_dict in poly_intersect_lookup.iteritems():
            heapq.heappush(
                heap, (len(poly_dict['intersects']), poly_fid, poly_dict))

        # build maximal subset
        maximal_set = set()
        while len(heap) > 0:
            _, poly_fid, poly_dict = heapq.heappop(heap)
            for maxset_fid in maximal_set:
                if maxset_fid in poly_intersect_lookup[poly_fid]['intersects']:
                    # it intersects and can't be part of the maximal subset
                    break
            else:
                # made it through without an intersection, add poly_fid to
                # the maximal set
                maximal_set.add(poly_fid)
                # remove that polygon and update the intersections
                del poly_intersect_lookup[poly_fid]
        # remove all the polygons from intersections once they're compuated
        for maxset_fid in maximal_set:
            for poly_dict in poly_intersect_lookup.itervalues():
                poly_dict['intersects'].discard(maxset_fid)
        subset_list.append(maximal_set)
    return subset_list

def calculate_raster_stats(raster_path):
    """Calculate and set min, max, stdev, and mean for all bands in raster.

    Parameters:
        raster_path (string): a path to a GDAL raster raster that will be
            modified by having its band statistics set

    Returns:
        None
    """
    raster = gdal.OpenEx(raster_path, gdal.GA_Update)
    raster_properties = get_raster_info(raster_path)
    for band_index in range(raster.RasterCount):
        target_min = None
        target_max = None
        target_n = 0
        target_sum = 0.0
        for _, target_block in iterblocks(
                raster_path, band_index_list=[band_index+1]):
            nodata_target = raster_properties['nodata'][band_index]
            # guard against an undefined nodata target
            valid_mask = numpy.ones(target_block.shape, dtype=bool)
            if nodata_target is not None:
                valid_mask[:] = target_block != nodata_target
            valid_block = target_block[valid_mask]
            if valid_block.size == 0:
                continue
            if target_min is None:
                # initialize first min/max
                target_min = target_max = valid_block[0]
            target_sum += numpy.sum(valid_block)
            target_min = min(numpy.min(valid_block), target_min)
            target_max = max(numpy.max(valid_block), target_max)
            target_n += valid_block.size

        if target_min is not None:
            target_mean = target_sum / float(target_n)
            stdev_sum = 0.0
            for _, target_block in iterblocks(
                    raster_path, band_index_list=[band_index+1]):
                # guard against an undefined nodata target
                valid_mask = numpy.ones(target_block.shape, dtype=bool)
                if nodata_target is not None:
                    valid_mask = target_block != nodata_target
                valid_block = target_block[valid_mask]
                stdev_sum += numpy.sum((valid_block - target_mean) ** 2)
            target_stddev = (stdev_sum / float(target_n)) ** 0.5

            target_band = raster.GetRasterBand(band_index+1)
            target_band.SetStatistics(
                float(target_min), float(target_max), float(target_mean),
                float(target_stddev))
            target_band = None
        else:
            LOGGER.warn(
                "Stats not calculated for %s band %d since no non-nodata "
                "pixels were found.", raster_path, band_index+1)
    raster = None
