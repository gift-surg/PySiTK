import os
import sys
import tempfile

DIR_ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_TEST = os.path.join(DIR_ROOT, "..", "data")
# DIR_TMP = tempfile.gettempdir()
DIR_TMP = tempfile.mkdtemp()

# Linked executables
ITKSNAP_EXE = "itksnap"
FSLVIEW_EXE = "fsleyes"  # Viewer in FSL 5.0.10
# FSLVIEW_EXE = "fslview" # deprecated in FSL 5.0.10 -> fslview_deprecated
# FSLVIEW_EXE = "fslview_deprecated"
NIFTYVIEW_EXE = "NiftyView"

# Set default viewer
VIEWER = ITKSNAP_EXE
