from __future__ import division, print_function
import pandas as pd
import numpy as np
import xlrd


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


def wrong_corrector(record, target, newlabel, sealevel_data,
                    is_vsmow, output_vpdb):
    """
    Correct d18O VSMOW data for ice volume.  BAD VERSION.

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
        convert_d18o(record, target, target + ' (VSMOW)')
        target = target + ' (VSMOW)'

    record[newlabel] = (
        record.loc[:, target] - sealevel_data.loc[record.index,
                                                  'd18o'])
    if output_vpdb:
        convert_d18o(record, newlabel, newlabel + ' (VPDB)', vsmow2vpdb=True)


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


class PaleoRecord(pd.DataFrame):
    """
    Class for holding paleorecord data.

    Also does d18o conversions.

    """

    def __init__(self, data, columns, **pandas_kw):
        """
        Initialize PaleoRecord from pandas DataFrame.

        Other features will probably be added later.
        To use ``self.icevolume_correction()``,
        set index to ages (should be rounded to the nearest year).

        Parameters
        ----------
        data : numpy ndarray (structured or homogeneous), dict, or DataFrame
            Dict can contain Series, arrays, constants, or list-like objects
        columns : Index or array-like
            Column labels to use for resulting frame. Required.
        **pandas_kw
            Other keyword arguments accepted by pandas.DataFrame():
            index, dtype, copy
        """
        pd.DataFrame.__init__(self, data, columns=columns, **pandas_kw)

    def convert_d18o(self, target_label, new_label, vsmow2vpdb=True):
        """
        Convert d18O from VPDB to VSMOW scale or vice versa.

        Parameters
        ----------
        target_label : string
            The label for the column of d18O data to convert
        new_label : string
            Label for converted data
        vsmow2vpdb : Boolean
            Default True, convert from VSMOW to VPDB scale.  Set ``False`` to
            go in opposite direction

        """
        if vsmow2vpdb:
            self[new_label] = (self.loc[:, target_label] * 0.97002) - 29.98
        else:
            self[new_label] = (self.loc[:, target_label] * 1.03091) + 30.91

    def icevolume_correction(self, d18o_label, corrected_d18o_label,
                             sealevel_data, sld18o_label='d18o'):

        self[corrected_d18o_label] = (
            self.loc[:, d18o_label] - sealevel_data.loc[self.index,
                                                        sld18o_label])

        # ((std_d18o/1000+1)/(target_d18o/1000+1)-1)*1000

    def new_corrector(self, d18o_label, corrected_label, sealevel_data,
                      d18o_is_vsmow, sld18o_label='d18o'):
        if not d18o_is_vsmow:
            self.convert_d18o(d18o_label, d18o_label + ' (VSMOW)', vsmow2vpdb=False)
            d18o_label = d18o_label + ' (VSMOW)'

        self[corrected_label] = (
            ((self.loc[:, d18o_label]/1000+1) /
             (sealevel_data.loc[self.index, sld18o_label]/1000+1)-1)*1000)

    # def append(self, other, add_hiatus, **pandas_kw):
    #     # add hiatus stuff
    #     pd.append(self, other, **pandas_kw)


# def simple_concat(objs, add_hiatus=False, keys=None, ignore_index=False,
#                   copy=True, hiatus_inds=None):
#     """
#     Does the default concatenation operation in pandas.  Can add hiatus
#     (row of NaNs) in between objs. axis is 0, join is outer.
#     If you provide keys and add_hiatus == True, must provide keys for
#     hiatuses

#     """

def load_worksheet(workbook, worksheet_name):
    """
    Open Excel workbook using xlrd and get worksheet.

    Parameters
    ----------
    workbook : string
        Full or relative path to the Excel workbook
    worksheet_name : Unicode string
        The name of the worksheet containing data

    Returns
    -------
    wksht : Sheet object
        Sheet object containing data

    """
    wkbk = xlrd.open_workbook(workbook)
    wksht = wkbk.sheet_by_name(worksheet_name)

    return wksht


def load_data(worksheet, sample_labelcol, data_cols, start_row, end_row=None):
    """
    Load data from Excel worksheet into a list of NumPy arrays.

    Parameters
    ----------
    worksheet : Sheet object
        Sheet object containing data
    sample_labelcol : int
        Index of the sample label column.
        Used to only grab cells in the data columns that correspond to actual
        samples.  Can pass None to collect all cells in column slice.
    data_col : list of ints
        Column of actual data wanted.
    start_row : int
        Row of first line of data

    Keyword Arguments
    -----------------
    end_row : int
        Default None.  Row of last line of data

    Returns
    -------
    data : list of (M) ndarray of floats
        The data in 1D arrays of length M in a list

    """
    # Get sample label cell types (0= empty)
    sample_labels = worksheet.col_types(sample_labelcol, start_rowx=start_row,
                                        end_rowx=end_row)

    # Find indices of nonzero sample label cells
    samplename_present = np.nonzero(sample_labels)[0]

    data = []

    for col in data_cols:

        # Get data column values
        data_vals = np.asarray(worksheet.col_values(col, start_rowx=start_row,
                                                    end_rowx=end_row))
        try:
            # Get data values that correspond to an actual sample name
            data_vals = data_vals[samplename_present].astype(float)
        except:
            print(np.where(type(data_vals[samplename_present]) == str))
            raise TypeError('There is a string in your data!')

        data.append(data_vals)

    return data
