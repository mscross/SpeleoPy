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
           'icevol_corr_prep',
           'convert_d18o',
           'wrong_corrector',
           'icevolume_correction',
           'join_records',
           'get_sealevel_data']


from .damp import AgeModel

from .record_class import (PaleoRecord, load_worksheet, load_data,
	icevol_corr_prep, convert_d18o, wrong_corrector, icevolume_correction,
	join_records)

from .sealevel import get_sealevel_data
