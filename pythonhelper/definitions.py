import os
import sys

DIR_ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_TEST = os.path.join(DIR_ROOT, "..", "data")
DIR_TMP = "/tmp/"

# Linked executables
ITKSNAP_EXE = "itksnap"
# FSLVIEW_EXE = "fslview" # deprecated in FSL 5.0.10 -> fslview_deprecated
# FSLVIEW_EXE = "fslview_deprecated"
FSLVIEW_EXE = "fsleyes"  # Viewer in FSL 5.0.10
NIFTYVIEW_EXE = "NiftyView"
BET_EXE = "bet"
REG_ALADIN_EXE = "reg_aladin"
REG_F3D_EXE = "reg_f3d"
