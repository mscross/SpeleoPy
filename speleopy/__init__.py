"""
SpeleoPy package containing tools for gathering paleoclimate
data from Excel workbooks, performing VSMOW<->VPDB conversions,
correcting oxygen isotope data for ice volume, and combining and
managing fragmentary records.  Also contains tools for creating
a simple Monte Carlo age model and formatting age information
into an OxCal codeblock.

"""

__all__ = ['AgeModel',
           'PaleoRecord',
           'load_worksheet',
           'load_data',
           'get_sealevel_data']


from .damp import AgeModel

from .record_class import PaleoRecord, load_worksheet, load_data

from .sealevel import get_sealevel_data
