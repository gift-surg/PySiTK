##
# \file create_video_from_volume.py
# \brief      script to create a video from a 3D image volume
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2018
#

import argparse
import numpy as np
import SimpleITK as sitk
import os

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import pysitk.volume_splitter as vol_split


def main():
    parser = argparse.ArgumentParser(description="Create video from volume")

    parser.add_argument('--image',
                        required=True,
                        type=str,
                        help="Path to 3D image (*.nii.gz or *.nii)",
                        )
    parser.add_argument('--fps',
                        required=False,
                        type=float,
                        help="Frames per second",
                        default=1,
                        )
    parser.add_argument('--axis',
                        required=False,
                        type=int,
                        help="Axis to sweep through the volume",
                        default=2,
                        )
    parser.add_argument('--begin',
                        required=False,
                        type=int,
                        help="Starting slice for video",
                        default=None,
                        )
    parser.add_argument('--end',
                        required=False,
                        type=int,
                        help="End slice for video",
                        default=None,
                        )
    parser.add_argument('--output',
                        required=True,
                        type=str,
                        help="Path to output video (*.mp4)",
                        )

    args = parser.parse_args()

    image_sitk = sitk.ReadImage(args.image)

    image_nda = sitk.GetArrayFromImage(image_sitk)

    scale = np.max(image_nda)
    filename = os.path.basename(args.output).split(".")[0]
    dir_output = os.path.dirname(args.output)
    dir_output_slices = os.path.join(dir_output, "slices")
    ph.create_directory(dir_output_slices)
    ph.clear_directory(dir_output_slices)

    splitter = vol_split.VolumeSplitter(image_nda, axis=args.axis)
    splitter.rescale_array(scale=scale)
    splitter.export_slices(
        dir_output=dir_output_slices,
        filename=filename,
        begin=args.begin,
        end=args.end,
        )
    splitter.create_video(
        dir_input_slices=dir_output_slices,
        path_to_video=args.output,
        fps=args.fps,
    )

    return 0


if __name__ == '__main__':
    main()
