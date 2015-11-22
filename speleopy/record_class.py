from __future__ import division, print_function
import pandas as pd
import numpy as np
import xlrd


class PaleoRecord(pd.DataFrame):
    """
    Class for holding paleorecord data.

    """
    def __init__(self, data, columns, **pandas_kw):
        """
        Initialize PaleoRecord from pandas DataFrame.  Other features will
        probably be added later.  To use ``self.icevolume_correction()``,
        set index to ages (should be rounded to the nearest year)

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
        """
        Use either regular data or rolling mean
        Ages should be set as index

        Parameters
        ----------
        d18o_label : string
            Label of d18O data to correct for ice volume
        corrected_d18o_label : string
            Label of new corrected data column
        sealevel_data : pandas DataFrame
            The data to use to correct for sea level
        sld18o_label : string
            Default 'd18o'.  The label of the d18o data in ``sealevel_data``

        """
        self[corrected_d18o_label] = (
            self.loc[:, d18o_label] - sealevel_data.loc[self.index,
                                                        sld18o_label])

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
    Open Excel workbook using xlrd and get worksheet

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
    Load data from Excel worksheet into a list of NumPy arrays

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
