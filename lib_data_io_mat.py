"""
Library Features:

Name:          lib_data_io_mat
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20220320'
Version:       '1.0.0'
"""

# -------------------------------------------------------------------------------------
# Libraries
import logging
import scipy.io
import os

from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)

# Debug
# import matplotlib.pylab as plt
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read mat obj
def read_file_mat(file_name):
    data = None
    if os.path.exists(file_name):
        data = scipy.io.loadmat(file_name)
    return data
# -------------------------------------------------------------------------------------

