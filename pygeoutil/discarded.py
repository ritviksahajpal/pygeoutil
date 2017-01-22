import os
import util

import numpy as np

def merge_nc_files(list_nc_files, path_out_nc, common_var_name='', mask_val=np.NaN, default_val=np.NaN,
                   replace_var_by_file_name=False, normalize_arr=None):
    """

    Args:
        list_nc_files:
        path_out_nc:
        common_var_name:
        replace_var_by_file_name: If True, then replace common_var_name variable by name of file
        normalize_arr:

    Returns:

    """
    list_dims = []  # List of dimensions in input netCDF file
    list_vars = []

    if os.path.isfile(path_out_nc):
        return

    with util.open_or_die(path_out_nc, perm='w') as hndl_out_nc:
        for fl in list_nc_files:
            with util.open_or_die(fl) as hndl_nc:
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

                            out_var = hndl_out_nc.createVariable(new_name_var, var.datatype, var.dimensions, zlib=True,
                                                                 fill_value=default_val)
                        else:
                            # Dimensions handled here
                            list_vars.append(name_var)
                            out_var = hndl_out_nc.createVariable(name_var, var.datatype, var.dimensions, zlib=True,
                                                                 fill_value=default_val)

                        # Copy variable attributes
                        # out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})

                        # Copy variable data
                        if np.isnan(mask_val) and len(var.shape) >= 2:
                            #pdb.set_trace()
                            #if not np.isnan(default_val):
                            #var[:].data[var[:].data == default_val] = np.NaN
                            var = var[:].filled(0.0)
                            if normalize_arr is not None:
                                var = var[:] * normalize_arr
                            out_var[:] = var[:]
                        else:
                            out_var[:] = var[:]