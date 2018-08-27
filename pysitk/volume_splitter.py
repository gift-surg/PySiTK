##
# \file volume_splitter.py
# \brief      Class to split a volume (3D data array) into slices and/or to
#             create a video
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       December 2017
#

import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph


##
# Class to split a volume (3D data array) to
#  *- export individual slices as png file to output directory
#  *- create a video from exported png files
# \date       2017-12-10 21:43:20+0000
#
class VolumeSplitter(object):

    ##
    # Store 3D numpy data array
    # \date       2017-12-10 21:13:43+0000
    #
    # \param      self  The object
    # \param      nda   3D numpy data array; slices are along the i-axis, i.e.
    #                   nda[i, :, :] (following a potential SimpleITK to numpy
    #                   array conversion)
    #
    def __init__(self, nda, axis=0):
        self._nda = nda
        self._axis = axis

    ##
    # Rescale data array such that volume is scaled between val_min and val_max
    #
    # Rationale: Prepare for export to png image (with range between 0 and 255)
    # \date       2017-12-10 21:15:21+0000
    #
    # \param      self     The object
    # \param      scale    Provide scaling for images; scalar>0
    # \param      val_min  Minimium intensity value; scalar >= 0
    # \param      val_max  Minimium intensity value; scalar > val_min >= 0
    #
    def rescale_array(self, scale=None, val_min=0, val_max=255):

        if scale is None:
            scale = np.max(self._nda)

        # Scale image intensities
        self._nda = val_max * self._nda / float(scale)

        # Ensure image intensities are between val_min and val_max
        self._nda = np.clip(self._nda, val_min, val_max)

    ##
    # Export individual slices of 3D data array as png file
    # \date       2017-12-10 21:18:09+0000
    #
    # \param      self        The object
    # \param      dir_output  Output director for export; string
    # \param      filename    Filename of image files; string. Slice number
    #                         will be automatically appended
    #
    def export_slices(self, dir_output, filename, begin=None, end=None):
        ph.create_directory(dir_output)

        if begin is None:
            begin = 0
        if end is None:
            end = self._nda.shape[self._axis]

        # Write each slice individually
        for k in range(begin, end):
            if self._axis == 0:
                nda_2d = self._nda[k, :, :]
            elif self._axis == 1:
                nda_2d = self._nda[:, k, :]
            else:
                nda_2d = self._nda[:, :, k]
            # nda_2d = np.swapaxes(nda_2d, 0, 1)
            nda_2d = nda_2d[::-1, :]

            path_to_image = os.path.join(
                dir_output, self._get_filename_slice(filename, k + 1))
            ph.write_image(nda_2d, path_to_image, verbose=True)

    ##
    # Creates a video.
    # \date       2017-12-10 21:19:42+0000
    #
    # \param      self        The object
    # \param      dir_output  The dir output
    # \param      filename    The filename
    # \param      fps         The fps
    #
    # \return     { description_of_the_return_value }
    #
    def create_video(self, path_to_video, dir_input_slices, fps=1):

        dir_output_video = os.path.dirname(path_to_video)
        filename = os.path.basename(path_to_video).split(".")[0]
        path_to_slices = "%s*.png" % os.path.join(dir_input_slices, filename)
        path_to_video = os.path.join(dir_output_video, "%s.mp4" % filename)

        path_to_video_tmp = os.path.join(
            dir_output_video, "%s_tmp.mp4" % filename)

        # Check that folder containing the slices exist
        if not ph.directory_exists(dir_input_slices):
            raise IOError("Folder '%s' meant to contain exported slices does "
                          "not exist" % dir_input_slices)

        # Check that the folder contains exported slices as png files        
        # if not ph.file_exists(os.path.join(
        #         dir_input_slices, self._get_filename_slice(filename, 1))):
        #     raise IOError(
        #         "Slices '%s' need to be generated first using "
        #         "'export_slices'" % (path_to_slices))

        # Create output folder for video
        ph.create_directory(dir_output_video)

        # ---------------Create temp video from exported slices----------------
        cmd_args = []
        cmd_args.append("-monitor")
        cmd_args.append("-delay %d" % (100. / fps))

        cmd_exe = "convert"

        cmd = "%s %s %s %s" % (
            cmd_exe, (" ").join(cmd_args), path_to_slices, path_to_video_tmp)
        flag = ph.execute_command(cmd)
        if flag != 0:
            raise RuntimeError("Unable to create video from slices")

        # ----------------------Use more common codec (?)----------------------
        cmd_args = []
        # overwrite possibly existing image
        cmd_args.append("-y")
        # Define input video to be converted
        cmd_args.append("-i %s" % path_to_video_tmp)
        # Use H.264 codec for video compression of MP4 file
        cmd_args.append("-vcodec libx264")
        # Define used pixel format
        cmd_args.append("-pix_fmt yuv420p")
        # Avoid error message associated to odd rows
        # (https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2)
        cmd_args.append("-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2'")
        cmd = "ffmpeg %s %s" % ((" ").join(cmd_args), path_to_video)
        ph.execute_command(cmd)

        # Delete temp video
        os.remove(path_to_video_tmp)

    ##
    # Gets the filename for an exported slice.
    # \date       2017-12-10 21:41:31+0000
    #
    # \param      filename      Filename as string
    # \param      slice_number  Slice number as int
    #
    # \return     Slice filename with preceding zeros as string.
    #
    @staticmethod
    def _get_filename_slice(filename, slice_number):
        return "%s_%04d.png" % (filename, slice_number)
