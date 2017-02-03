import os
import pdb

import numpy
import gdal
import gdalconst
from osgeo import osr


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
        band.ComputeStatistics(0)

    #Make sure the dataset is closed and cleaned up
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

    #Make sure the dataset is closed and cleaned up
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

    if dataset == None:
        raise IOError('File not found or not valid dataset type at: %s' % dataset_uri)

    srs.SetProjection(dataset.GetProjection())
    linear_units = srs.GetLinearUnits()
    geotransform = dataset.GetGeoTransform()

    # take absolute value since sometimes negative widths/heights
    try:
        numpy.testing.assert_approx_equal(
            abs(geotransform[1]), abs(geotransform[5]))
        size_meters = abs(geotransform[1]) * linear_units
    except AssertionError as e:
        size_meters = (abs(geotransform[1]) + abs(geotransform[5])) / 2.0 * linear_units

    # Make sure the dataset is closed and cleaned up
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

    band = dataset.GetRasterBand(1)
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
