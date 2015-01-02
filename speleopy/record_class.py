import numpy as np
import xlrd
import sealevel as sl


class Record:
    """
    Class for holding data
    """

    def __init__(self, datacols, header, age_is_ky=False):
        """
        Initialize Record.

        Stacks data list into a 2D array of data columns.

        Parameters
        ----------
        datacols : list of 1D ndarrays
        header : list of strings

        Keyword Arguments
        -----------------
        age_is_ky : Boolean

        """

        data = np.empty((datacols[0].size, 0))

        for d in datacols:
            if d.ndim == 1:
                d = np.atleast_2d(d).T
            data = np.concatenate((data, d), axis=1)

        self.data = np.ma.masked_where(data==-999, data)


        self.header = header
        self.ageky = age_is_ky


    def __add__(self, other):
        """
        Put records together.  Must have same data columns
        (not necessarily in same order)

        Parameters
        ----------
        other : Record object

        """

        # Check if header is the same in each Record
        if not self.header == other.header:
            if (set(self.header) ==set(other.header) and
                len(self.header)==len(other.header)):
                # Get column orders that each data array is in
                self_order = np.array(self.header).argsort()
                other_order = np.array(other.header).argsort()

                self.header = np.array(self.header)[self_order]
                other.header = np.array(other.header[other_order])

                self.header = self.header.tolist()
                other.header = other.header.tolist()

                self.data = self.data[self_order]
                other.data = other.data[other_order]

            else:
                raise ValueError('Data array columns cannot be matched!')

        # At this point all data columns are in the correct order

        # Initialize empty row
        hiatus = np.empty((1, self.data.shape[1]))
        hiatus.fill(-999)

        new_data = np.concatenate((self.data, hiatus, other.data))

        new_data = np.hsplit(new_data, self.data.shape[1])
        new_record = Record(new_data, self.header)

        return new_record


    def vsmow2vpdb(self, convert_from, d18o_label):
        """
        Convert d18O in VSMOW or VPDB scale to other scale.

        Parameters
        ----------
        convert_from : string
            The reference standard of the oxygen isotope values in self.data
            Accepts any reasonable variation of VSMOW and VPDB
        d18o_label : string
            Column header string for d18O data

        Notes
        -----
        Carbonates may be referenced to either VPDB or VSMOW;
            waters may only be referenced to VSMOW

        """

        current_scale = str.lower(convert_from)
        current_data = self.data[:,self.header.index(d18o_label)]

        if 'smow' in current_scale or 'water' in current_scale:
            converted = (current_data * 0.97002) - 29.98
            newlabel = '(VPDB)'
        elif 'pdb' in current_scale or 'dee' in current_scale:
            converted = (current_data * 1.03091) + 30.91
            newlabel = '(VSMOW)'
        else:
            raise ValueError('Unidentified reference standard: ' + convert_from)

        converted = np.atleast_2d(converted).T

        self.data = np.concatenate((self.data, converted), axis=1)
        self.header.append(d18o_label + newlabel)


    def icevolume_correction(self, d18o_label, age_label, corr_label,
                             smoothing):
        """
        Correct d18O data in Record for sea level.

        Adds a column to Record with corrected data.

        Parameters
        ----------
        d18o_label : string
        age_label : string
        corr_label : string
        smoothing : tuple of ints

        """

        age = self.data[:, self.header.index(age_label)]
        d18o = self.data[:, self.header.index(d18o_label)]

        # If The label is in the header but you haven't done the correction:
        if corr_label in self.header and len(self.header) > self.data.shape[1]:
            app = False

        # If the label in header and have done correction:
        elif corr_label in self.header and len(self.header)==self.data.shape[1]:
            self.data = np.delete(self.data,
                                  np.s_[:, self.header.index(corr_label)])
            del self.header[self.header.index(corr_label)]
            app = True
        else:
            app = True

        # If age is in ky, convert to y
        if self.ageky:
            age = age*1000
        # Find indices of hiatus masking and split into sub-arrays
        try:
            hiatus_location = np.nonzero(age.mask)[0]
            hiatus_location[0]
        except:
            # Because there is no mask when there are no hiatuses
            orig_age = [age]
            orig_d18o = [d18o]
            add_hiatus = False
        else:
            # When split, the hiatus goes with the lower array
            orig_age = np.split(age, hiatus_location)
            orig_d18o = np.split(d18o, hiatus_location)

            add_hiatus = True

            for i in range(1, len(orig_age)):
                orig_age[i] = orig_age[i][1:,]
                orig_d18o[i] = orig_d18o[i][1:,]

        # make copies of the original data for editing purposes
        age = orig_age
        d18o = orig_d18o

        # If any of the data is too old, get rid of it
        for i in range(0, len(age)):
            too_old = np.nonzero(age[i] > 168000)

            if too_old[0].size > 0:

                first_occurrence = too_old[0][0]

                if first_occurrence == 0:
                    first_missing = i
                    # Get rid of data and arrays after it- assumes arrays are
                    # in age order and that subsequent arrays don't start
                    # at significantly younger ages
                    age = age[:i]
                    d18o = d18o[:i]
                    break

                age[i] = age[i][:first_occurrence]
                d18o[i] = d18o[i][:first_occurrence]

        # Get SL data
        sl_ages, sl_m, sl_correction = sl.get_sealevel_data()

        # Smooth via running average
        sl_correction = sl.smooth(smoothing[0], smoothing[1],
                                  sl_correction)

        # Get corrected data
        corrected = []

        for i in range(0, len(age)):
            corrected_data = sl.sealevel_corrector(sl_correction, sl_ages,
                                                   d18o[i], age[i])
            corrected.append(corrected_data)

        # If we lost any age arrays because they were too old,
        # Initialize and mask arrays the same size as those
        try:
            missing_ages = orig_age[first_missing:]
        except NameError:
            pass
        else:
            for gone in missing_ages:
                replacement = np.empty(gone.size)
                replacement.fill(-999)

                corrected.append(replacement)

        if add_hiatus:
            for i in range(1, len(corrected)):
                corrected[i] = np.pad(corrected[i], (1,0), 'constant',
                                      constant_values=(-999, -999))


        # Check that the size of each array is ok
        for i in range(0, len(corrected)):

            if corrected[i].size != orig_d18o[i].size:
                padlength = (orig_d18o[i].size + 1) - corrected[i].size

                corrected[i] = np.pad(corrected[i], (0, padlength), 'constant',
                                      constant_values=(-999, -999))

        corrected = np.hstack(corrected)
        corrected[np.isnan(corrected)]=-999
        corrected = np.ma.masked_where(corrected==-999, corrected)
        corrected = np.atleast_2d(corrected).T

        self.data = np.ma.concatenate((self.data, corrected), axis=1)
        if app:
            self.header.append(corr_label)


    def get_data(self, data_label):
        """
        Shortcut for getting a particular column

        Parameters
        ----------
        data_label : string

        Returns
        -------
        data_col : 1D ndarray of floats

        """

        data_col = self.data[:, self.header.index(data_label)]
        data_col = np.ma.masked_where(data_col == -999, data_col)

        return data_col


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
            print np.where(type(data_vals[samplename_present]) == str)
            raise TypeError('There is a string in your data!')


        data.append(data_vals)

    return data