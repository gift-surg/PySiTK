##
# \file create_video_from_volume.py
# \brief      script to create a video from a 3D image volume
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2018
#

import os
import argparse
import numpy as np
import SimpleITK as sitk

import pysitk.data_anonymizer as da
import pysitk.python_helper as ph


def main():

    parser = argparse.ArgumentParser(
        description="Script to anonymize multiple files. "
        "This usually comes in three steps: "
        "1) --create-dictionary "
        "2) --anonymize-files "
        "3) --reveal-files (after assessment)"
    )

    parser.add_argument(
        '-f', '--filenames',
        required=True,
        type=str,
        nargs="+",
        help="Path to filenames",
    )
    parser.add_argument(
        '-d', '--dictionary',
        required=True,
        type=str,
        help="Path to dictionary that shall be used for anonymization (*.o). "
        "Can be computed using the '--create-dictionary' flag if not available yet.",
    )
    parser.add_argument(
        '-o', '--dir-output',
        required=False,
        type=str,
        help="Path to output directory for whatever anonymization step. ",
    )
    parser.add_argument(
        "-p", "--prefix",
        help="Prefix used when anonymization dictionary is created",
        type=str,
        default="anonymized_",
    )

    # Options for anonymization runs
    parser.add_argument(
        "--create-dictionary",
        help="Create a dictionary that shall be used for anonymization. "
        "Typically, this is the first step.",
        action='store_true'
    )
    parser.add_argument(
        "--anonymize-files",
        help="Anonymize files based on a given dictionary. "
        "Typically, this is the second step.",
        action='store_true'
    )
    parser.add_argument(
        "--reveal-files",
        help="Reveal anonymized files. "
        "Typically, this is the third step (after assessment).",
        action='store_true'
    )

    args = parser.parse_args()

    data_anonymizer = da.DataAnonymizer(
        filenames=args.filenames,
        prefix_identifiers=args.prefix,
    )

    if ph.strip_filename_extension(args.dictionary)[1] != "o":
        raise IOError("--dictionary must point to an *.o file")

    if args.create_dictionary:
        data_anonymizer.generate_identifiers()
        data_anonymizer.generate_randomized_dictionary()
        data_anonymizer.write_dictionary(args.dictionary)

    if args.anonymize_files:
        if args.dir_output is None:
            raise IOError("--dir-output must be provided")

        data_anonymizer.read_dictionary(args.dictionary)
        data_anonymizer.anonymize_files(args.dir_output)

    if args.reveal_files:
        if args.dir_output is None:
            raise IOError("--dir-output must be provided")

        data_anonymizer.read_dictionary(args.dictionary)
        data_anonymizer.reveal_anonymized_files(args.dir_output)

    return 0


if __name__ == '__main__':
    main()
