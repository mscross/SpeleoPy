"""
SpeleoPy package containing tools for gathering paleoclimate
data from Excel workbooks, performing VSMOW<->VPDB conversions,
correcting oxygen isotope data for ice volume, and combining and
managing fragmentary records.  Also contains tools for creating
a simple Monte Carlo age model and formatting age information
into an OxCal codeblock.

"""

__all__ = ['AgeModel',
		   'Record',
		   'load_worksheet',
		   'load_data',
		   'sealevel_corrector',
		   'sealevelagebracket',
		   'smooth',
		   'get_sealevel_data']


from .damp import AgeModel

from .record_class import Record, load_worksheet, load_data

from .sealevel import (sealevel_corrector, sealevelagebracket, smooth,
					   get_sealevel_data)