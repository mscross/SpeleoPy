from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from . import data_dir


def get_sealevel_data(interpolation='linear'):
    """
    Acquire sea level data.

    Loads sealevel data from file, performs linear or cubic interpolation,
    then returns the results in a pandas DataFrame indexed by age.

    Parameters
    ----------
    interpolation : string
        Default linear.  'Cubic' or 'linear'.  How to interpolate the
        sea level data.

    """
    sealevel_file = os.path.join(data_dir, 'sealevel.txt')

    with open(sealevel_file) as sealevel:
        contents = sealevel.readlines()
        sldata = np.empty((len(contents) - 1, 3))
        header = contents[0].split()
        for ind, line in enumerate(contents[1:]):
            sldata[ind, :] = [float(x) for x in line.split()]

    sl_interpolation_eq = interp1d(sldata[:, 0], sldata[:, 1],
                                   kind=interpolation)
    d18o_interpolation_eq = interp1d(sldata[:, 0], sldata[:, 2],
                                     kind=interpolation)

    every_year = np.expand_dims(np.arange(sldata[0, 0], sldata[-1, 0] + 1), axis=1)
    sl = sl_interpolation_eq(every_year)
    d18o = d18o_interpolation_eq(every_year)

    sealevel_data = pd.DataFrame(np.hstack([every_year, sl, d18o]), columns=header)
    sealevel_data.set_index('Age', inplace=True)

    return sealevel_data
