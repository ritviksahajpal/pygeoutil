import os
import pdb
import time
import errno
import subprocess
import tempfile
import atexit
import functools
import math

import numpy
import gdal
import gdalconst
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

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
    last_time = time.time()

    # make sure that the input lists are of the same length
    list_lengths = [
        len(dataset_uri_list), len(dataset_out_uri_list),
        len(resample_method_list)]
    if not reduce(lambda x, y: x if x == y else False, list_lengths):
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
        bounding_box = reduce(
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
    return bounding_box
