from __future__ import division, print_function
import pandas as pd
import numpy as np


def icevol_corr_prep(record, agelabel, age_in_ky):
    """
    Prepare ``record`` for the ice volume correction.

    Converts all numbers to actual number dtypes, converts
    Age to integer years and sets it as index.

    Parameters
    ----------
    record : pandas DataFrame
        A pandas DataFrame containing the data to prepare.
    agelabel : string
        The column label for the age data.
    age_in_ky : Boolean
        If True, ages will be converted from kyrs to years.

    """
    record = record.convert_objects(convert_numeric=True,
                                    convert_dates=False,
                                    convert_timedeltas=False,
                                    copy=False)

    if age_in_ky:
        record[agelabel] = record[agelabel] * 1000

    record[agelabel] = record[agelabel] // 1

    record.set_index(agelabel, inplace=True)

    return record


def convert_d18o(record, target, newlabel, vsmow2vpdb=True):
    """
    Convert d18O from VSMOW to VPDB scale or vice versa.

    Parameters
    ----------
    record : pandas DataFrame
        A pandas DataFrame containing the data to convert.
    target : string
        The column label for the d18O data.
    newlabel : string
        The column label for the converted d18O data.
    vsmow2vpdb : Boolean
        Default True.  Indicates the direction of the conversion.

    """
    if vsmow2vpdb:
        record[newlabel] = (record[target] * 0.97002) - 29.98
    else:
        record[newlabel] = (record[target] * 1.03091) + 30.91


def icevolume_correction(record, target, newlabel, sealevel_data,
                         is_vsmow, output_vpdb):
    """
    Correct d18O VSMOW data for ice volume.

    The d18O of the ocean (nominally 0.0 permil today)
    has changed depending on the amount of (isotopically
    light) water locked up in ice sheets, which we can determine
    from sea level information.  This function
    removes that signal from your d18O data.

    Parameters
    ----------
    record : pandas DataFrame
        A pandas DataFrame containing the data to correct.
    target : string
        The column label for the d18O data to correct.
    newlabel : string
        The column label for the corrected d18o data.
    sealevel_data : pandas DataFrame
        The data to use to correct for ice volume.
    is_vsmow : Boolean
        Confirmation that the d18O data to correct is on
        the VSMOW scale.  If ``False``, then the data will
        be converted and placed in ``record`` prior to correction.
    output_vpdb : Boolean
        Algorithm takes VSMOW and outputs VSMOW.  If ``True``, this
        will convert the VSMOW output to VPDB and add it to ``record``.
    """
    if not is_vsmow:
        convert_d18o(record, target, target+' (VSMOW)', vsmow2vpdb=False)
        target = target + ' (VSMOW)'

    record[newlabel] = (
        ((record[target]/1000+1) /
         (sealevel_data.loc[record.index, 'd18o']/1000+1)-1)*1000)

    if output_vpdb:
        convert_d18o(record, newlabel, newlabel+' (VPDB)', vsmow2vpdb=True)


def join_records(record_list, record_names, separator_name='hiatus', 
                 **concat_kw):
    """
    Put distinct records in the same DataFrame separated by rows of NaNs.

    Parameters
    ----------
    record_list : list of pandas DataFrames
        The list of records to concatenate.
    record_names : list of strings
        The list of names of the records; the outer indices.
    separator_name : string
        The basename of the row of NaNs used to separate records in the
        final DataFrame (the outer index for each of these rows)
    concat_kw
        Any keywords accepted by ``pandas.concat()``

    """
    num_separators = len(record_list)-1

    nonedfs = [pd.DataFrame(np.array([[None]]),
                            columns=[record_list[0].columns[0]])]*num_separators

    separator_namelist = []
    for i in range(0, num_separators):
        separator_namelist.append(separator_name + str(i))

    try:
        keys = [None]*(len(record_names)+num_separators)
    except:
        keys=None
    else:
        keys[::2] = record_names
        keys[1::2] = separator_namelist

    dfs = [None]*(len(record_list)+num_separators)
    dfs[::2] = record_list
    dfs[1::2] = nonedfs

    return pd.concat(dfs, keys=keys, **concat_kw)
