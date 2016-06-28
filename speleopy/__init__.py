"""
SpeleoPy package containing tools for gathering paleoclimate
data from Excel workbooks, performing VSMOW<->VPDB conversions,
correcting oxygen isotope data for ice volume, and combining and
managing fragmentary records.  Also contains tools for creating
a simple Monte Carlo age model and formatting age information
into an OxCal codeblock.

"""

__all__ = ['AgeModel',
           'icevol_corr_prep',
           'convert_d18o',
           'icevolume_correction',
           'join_records',
           'get_sealevel_data']

# Find package and data directories
import os.path
_pkg_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(_pkg_dir, 'data')

from .damp import AgeModel

from .record_class import (icevol_corr_prep, convert_d18o,
  icevolume_correction,	join_records)

from .sealevel import get_sealevel_data
