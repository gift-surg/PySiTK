##
# \file python_helper.py
# \brief      Scripts to facilitate IO, plotting, printing etc
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
#


# Import libraries
import os
import re
import sys
import pickle
import subprocess
import numpy as np
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import json
import time
import errno
import datetime
import skimage.io
import itertools
import shutil

from six.moves import input

from pysitk.definitions import DIR_TMP
from pysitk.definitions import ITKSNAP_EXE, FSLVIEW_EXE, NIFTYVIEW_EXE
from pysitk.definitions import VIEWER

##
COLORS_STANDARD = [
    "r",        # red
    "b",        # blue
    "g",        # green
    "c",        # cyan
    "m",        # magenta
    "y",        # yellow
    "k",        # black
    "w",        # white
]


# Used for (box plot) texture/hatches
HATCHES = [
    "/",
    ".",
    "o",
    "\\",
    "*",
    "x",
    "//",
    "///",
]

# https://matplotlib.org/users/colormaps.html
# COLORS_TAB20 = [matplotlib.cm.tab20(x/10.) for x in range(0, 10)]
# Tableau20
COLORS_TABLEAU20 = np.array([
    (31, 119, 180),     # blue
    (174, 199, 232),
    (255, 127, 14),     # orange
    (255, 187, 120),
    (44, 160, 44),      # green
    (152, 223, 138),
    (214, 39, 40),      # red
    (255, 152, 150),
    (148, 103, 189),    # purple
    (197, 176, 213),
    (127, 127, 127),    # grey
    (199, 199, 199),
    (23, 190, 207),     # cyan
    (158, 218, 229),
    (140, 86, 75),      # brown
    (196, 156, 148),
    (227, 119, 194),    # pink
    (247, 182, 210),
    (188, 189, 34),     # "dirty" yellow
    (219, 219, 141),
]) / 255.
COLORS = COLORS_TABLEAU20

MARKERS = [
    "o",        # circle
    "s",        # square
    "v",        # triangle_down
    "X",        # x (filled)
    "p",        # pentagon
    "x",        # x
    "*",        # star
    "P",        # plus (filled)
    "^",        # triangle_up
    "<",        # triangle_left
    ">",        # triangle_right
    ".",        # point
    "+",        # plus
    "1",        # tri_down
    "2",        # tri_up
    "3",        # tri_left
    "4",        # tri_right
    "8",        # octagon
    ",",        # pixel
    "h",        # hexagon1
    "H",        # hexagon2
    "D",        # diamond
    "d",        # thin_diamond
    "|",        # vline
    "_",        # hline
    # "None",     # nothing
    # " ",        # nothing
    # "",         # nothing
    # '$...$',    # render the string using mathtext
]

LINESTYLES = [
    "-",        # solid line
    "--",       # dashed line
    "-.",       # dash-dotted line
    ":",        # dotted line
    "None",     # draw nothing
    # " ",        # draw nothing
    # "",         # draw nothing
]


#
# Check whether file exists to given file_path
# \date       2017-06-27 01:52:45+0100
#
# \param      file_path  path to file whose existence to be checked
#
# \return     true if the file exists, otherwise false.
#
def file_exists(file_path):
    return True if os.path.isfile(file_path) else False


##
# Determines whether input value can be converted to a float
# \date       2019-03-23 23:26:51+0000
#
# \param      value  input value
#
# \return     True if float, False otherwise.
#
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


##
# Check whether directory exists to given directory path
# \date       2017-06-27 01:53:20+0100
#
# \param      directory_path  path to directory whose existence to be checked
#
# \return     true if the file exists, otherwise false.
#
def directory_exists(directory_path):
    return True if os.path.isdir(directory_path) else False


##
# Writes variables.
# \date       2017-02-18 16:51:31+0000
#
# \param      variables  List of variables to store
# \param      directory  The directory
# \param      filename   The filename without extension
#
def write_variables(variables, directory, filename, filetype=".pckl"):

    filename_out = directory + filename + filetype

    f = safe_open(filename_out, 'wb')
    pickle.dump(variables, f, protocol=-1)  # protocol=-1 for large files
    flag = f.close()

    if not flag:
        print_info("Variables written to " + filename_out)
    else:
        print_info("Error: Variables could not be written")


##
# Reads variables.
# \date       2017-02-18 16:55:35+0000
#
# \param      directory  The directory
# \param      filename   The filename
#
# \return     List of read variables
#
def read_variables(directory, filename, filetype=".pckl"):

    filename_in = directory + filename + filetype

    f = open(filename_in, 'rb')
    variables = pickle.load(f)
    flag = f.close()

    if not flag:
        print_info("Variables read from " + filename_in)
    else:
        print_info("Error: Variables could not be read")

    return variables


def replace_string_for_print(string):
    string = re.sub(" ", "_", string)
    string = re.sub(":", "", string)
    string = re.sub("/", "_", string)
    return string


##
# Open "path" for writing, creating any parent directories if needed
# \date       2017-07-12 11:15:47+0100
#
# \param      path  The path
#
# \return     file open handle
#
def safe_open(path, access_mode='w'):
    if not isinstance(path, str):
        raise IOError("Given path must be of type string")

    directory = os.path.dirname(path)
    if directory != "":
        mkdir_p(os.path.dirname(path))
    return open(path, access_mode)


##
# Get "mkdir -p" functionality in python
#
# See https://stackoverflow.com/a/600612/119527 for further information.
# \date       2017-07-12 11:17:21+0100
#
# \param      path  string specifying command "mkdir -p path"
#
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


##
# Gets the executed call of script including all its options.
# \date       2017-07-12 11:46:39+0100
#
# \param      script_name  The name of current script, e.g. "my_script.py"
# \param      args         The arguments as given by \p parse_args of argparse
#
# \return     The performed script execution call as string.
#
def get_performed_script_execution(script_name, args):
    cmd = "python " + script_name + " \\\n"

    for arg in sorted(vars(args)):
        argument = ("%s " % (arg)).replace("_", "-")
        if type(getattr(args, arg)) is list:
            text = (" ").join([str(s) for s in getattr(args, arg)])
        else:
            text = getattr(args, arg)
        cmd += "\t--" + argument + "%s" % text + " \\\n"

    return cmd


##
# Writes a function call to executable file.
# \date       2017-07-12 11:45:45+0100
#
# \param      performed_script  The function call
# \param      filename       The filename
#
def write_performed_script_execution_to_executable_file(function_call, filename):

    call = "#!/bin/sh\n\n" + function_call

    text_file = safe_open(filename, "w")
    text_file.write("%s" % call)
    text_file.close()
    print_info("File " + filename + " generated.")

    # Make file executable
    os.system("chmod +x " + filename)


##
#       Wait for <ENTER> to proceed the execution
# \date       2016-11-06 15:41:43+0000
#
def pause():
    programPause = input("Press the <ENTER> key to continue ...")


##
# Exit/Terminate execution
# \date       2017-02-02 16:04:48+0000
#
def exit():
    sys.exit()


##
# Kill all ITK-SNAP processes
# \date       2018-02-21 17:36:54+0000
#
# \return     { description_of_the_return_value }
#
def killall_itksnap():
    with open(os.devnull, "wb") as devnull:
        subprocess.call(
            ["killall", "itksnap"], stdout=devnull, stderr=subprocess.STDOUT)
        subprocess.call(
            ["killall", "ITK-SNAP"], stdout=devnull, stderr=subprocess.STDOUT)


##
# Kill all FSLeyes processes
# \date       2018-02-21 17:36:54+0000
#
# \return     { description_of_the_return_value }
#
def killall_fsleyes():
    with open(os.devnull, "wb") as devnull:
        subprocess.call(
            ["killall", "fsleyes"], stdout=devnull, stderr=subprocess.STDOUT)
        subprocess.call(
            ["killall", "FSLeyes"], stdout=devnull, stderr=subprocess.STDOUT)


##
# Adds one to variable by taking advantage of pass by reference of list objects
#
# Idea is to use it for consecutive numbering, e.g. to generate batch jobs
# \date       2017-09-14 20:32:43+0100
#
# \param      x     list with one integer element, e.g. [0]
# \post       x integer element incremented by one
#
# \return     incremented integer value of list
#
def add_one(x):
    x[0] += 1
    return x[0]


##
# Open ITK-SNAP for given filename
# \date       2017-07-06 12:34:12+0100
#
# \param      path_to_filename  The path to filename as string
#
def itksnap(path_to_filename):
    show_nifti(path_to_filename, viewer="itksnap")


##
# Open FSLView for given filename
# \date       2017-07-06 12:34:12+0100
#
# \param      path_to_filename  The path to filename as string
#
def fsleyes(path_to_filename):
    show_nifti(path_to_filename, viewer="fsleyes")


##
# Open NiftyView for given filename
# \date       2017-07-06 12:34:12+0100
#
# \param      path_to_filename  The path to filename as string
#
def niftyview(path_to_filename):
    show_nifti(path_to_filename, viewer="niftyview")


##
# Open viewer for given filename
# \date       2017-07-06 12:34:12+0100
#
# \param      path_to_filename  The path to filename as string;
# \param      viewer            The viewer; either "fsleyes", "itksnap" or
#                               "niftyview"
#
def show_nifti(
        path_to_filename,
        viewer=VIEWER,
        segmentation=None,
        verbose=True,
):
    show_niftis(
        [path_to_filename],
        viewer=viewer,
        segmentation=segmentation,
        verbose=verbose)


##
# Open viewer for given filenames.
# \date       2017-07-06 12:36:05+0100
#
# \param      paths_to_filenames  List of strings containing paths to filenames
# \param      viewer            The viewer; either "fsleyes", "itksnap" or
#                               "niftyview"
#
def show_niftis(
    paths_to_filenames,
    viewer=VIEWER,
    segmentation=None,
    verbose=True,
):
    cmd = globals()["get_function_call_" + viewer](
        paths_to_filenames, segmentation)
    return execute_command(cmd, verbose=verbose)


##
# Writes a executable to show niftis.
# \date       2018-02-21 17:22:23+0000
#
# \param      paths_to_filenames  The paths to filenames
# \param      dir_output          The dir output
# \param      output_filename     The output filename
# \param      viewer              The viewer
# \param      segmentation        The segmentation
#
def write_show_niftis_exe(
        paths_to_filenames,
        dir_output,
        output_filename="showComparison.sh",
        viewer=VIEWER,
        segmentation=None):
    cmd_args = ["#!/bin/sh"]
    cmd_args.append(globals()["get_function_call_" + viewer](
        paths_to_filenames, segmentation))

    cmd = "\n".join(cmd_args)
    path_to_file = os.path.join(dir_output, output_filename)
    write_to_file(path_to_file, cmd)
    make_file_executable(path_to_file)


##
# Makes a a file executable.
# \date       2018-02-21 17:22:40+0000
#
# \param      path_to_file  The path to file
#
def make_file_executable(path_to_file):
    os.system("chmod +x %s" % path_to_file)


##
# Gets the function call for ITK-SNAP.
# \date       2017-06-28 17:55:35+0100
#
# \param      filenames              list of filenames. If more than one image,
#                                    the remaining ones are overlaid
# \param      filename_segmentation  filename of segmentation to be overlaid
#
# \return     string to be executed.
#
def get_function_call_itksnap(filenames, filename_segmentation=None):

    cmd = ITKSNAP_EXE + " -g \\\n"
    cmd += filenames[0] + " \\\n"

    # Add overlays
    if len(filenames) > 1:
        cmd += "-o \\\n"

        for i in range(1, len(filenames)):
            cmd += filenames[i] + " \\\n"

    # Add segmentation
    if filename_segmentation is not None:
        cmd += "-s \\\n"
        cmd += filename_segmentation + " \\\n"

    # Add termination
    cmd += "&"

    return cmd


##
# Gets the function call for FSL viewer.
# \date       2017-06-28 17:55:35+0100
#
# \param      filenames              list of filenames. If more than one image,
#                                    the remaining ones are overlaid
# \param      filename_segmentation  filename of segmentation to be overlaid
#
# \return     string to be executed.
#
def get_function_call_fsleyes(filenames, filename_segmentation=None):

    cmd = FSLVIEW_EXE + " \\\n"
    for i in range(0, len(filenames)):
        cmd += filenames[i] + " \\\n"

    if filename_segmentation is not None:
        cmd += filename_segmentation + " --alpha 30 -cm hsv \\\n"

    cmd += "&"

    return cmd


##
# Gets the function call for FSL viewer.
# \date       2017-06-28 17:55:35+0100
#
# \param      filenames              list of filenames. If more than one image,
#                                    the remaining ones are overlaid
# \param      filename_segmentation  filename of segmentation to be overlaid
#
# \return     string to be executed.
#
def get_function_call_fslview(filenames, filename_segmentation=None):

    cmd = FSLVIEW_EXE + " \\\n"
    for i in range(0, len(filenames)):
        cmd += filenames[i] + " \\\n"

    if filename_segmentation is not None:
        cmd += filename_segmentation + " -t 0.3 \\\n"

    cmd += "&"

    return cmd


##
# Gets the function call for NiftyView.
# \date       2017-06-28 17:55:35+0100
#
# \param      filenames              list of filenames. If more than one image,
#                                    the remaining ones are overlaid
# \param      filename_segmentation  filename of segmentation to be overlaid
#
# \return     string to be executed.
#
def get_function_call_niftyview(filenames, filename_segmentation=None):

    cmd = NIFTYVIEW_EXE + " \\\n"
    for i in range(0, len(filenames)):
        cmd += filenames[i] + " \\\n"

    if filename_segmentation is not None:
        cmd += filename_segmentation + " \\\n"

    cmd += "&"

    return cmd


##
# Reads an input from the command line and returns it
# \date       2016-11-18 12:45:10+0000
#
# \param      infotext  The infotext
# \param      default   The default value which will be shown in square
#                       brackets
#
# \return     Input as either string, int or float, depending on what was
#             entered
#
def read_input(infotext, default=None):
    if default is None:
        text_in = input(infotext + ": ")
        return text_in
    else:
        text_in = input(infotext + " [" + str(default) + "]: ")

        if text_in in [""]:
            return default
        else:
            return text_in


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


##
# { function_description }
# \date       2017-02-18 04:21:27+0000
#
# \param      y                 { parameter_description }
# \param      x                 { parameter_description }
# \param      xlabel            The xlabel
# \param      ylabel            The ylabel
# \param      title             The title
# \param      y_axis_style      The y axis style ("plot", "semilogy")
# \param      labels            The label
# \param      label_location    The label location, e.g. "upper right", "best"
# \param      color             The color
# \param      markers           The marker
# \param      markevery         The markevery
# \param      linestyle         The linestyle
# \param      label_shadow      The label shadow
# \param      label_frameon     The label frameon
# \param      linewidth         The linewidth
# \param      markerfacecolors  The markerfacecolor ("none", None)
# \param      markersize        The markersize
# \param      fontfamily        The fontfamily ("serif", "sans-serif", "monospace")
# \param      fontname          The fontname ("Arial", "Times New Roman")
# \param      use_tex           The use tex
# \param      fontsize          The fontsize
# \param      backgroundcolor   The backgroundcolor
# \param      aspect_ratio      The aspect ratio ("auto", "equal")
# \param      save_figure       The save fig
# \param      directory         The directory
# \param      filename          The filename including extension
# \param      fig_number        The fig number
#
# \return     figure
#
def show_curves(y, x=None, xlabel="", ylabel="", title="", xlim=None, ylim=None, y_axis_style="plot", labels=None, label_location="best", color=None, markers=None, markevery=1, linestyle="-", label_shadow=False, label_frameon=True, label_fontsize=None, label_boundingboxtoanchor=None, label_ncol=1, linewidth=1, markerfacecolors=None, markersize=5, fontfamily="serif", fontname="Arial", use_tex=False, fontsize=12, backgroundcolor="None", aspect_ratio="auto", save_figure=False, directory=None, filename="figure.pdf", fig_number=None, show_compact=0, subplots_left=0.08, subplots_bottom=0.11, subplots_right=0.99, subplots_top=0.83, subplots_wspace=0, subplots_hspace=0, figuresize=None, block_show=False):

    if type(y) is not list:
        y = [y]
    N_curves = len(y)

    # Change font
    font = {
        "family":   fontfamily,
        fontfamily:   fontname,
        "size":   fontsize,
    }
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=use_tex)

    if x is None:
        x = [None] * N_curves
        for i in range(N_curves):
            x[i] = np.arange(y[i].size) + 1
    elif type(x) is not list:
        x = [x] * N_curves

    if type(linewidth) is not list and not isinstance(linewidth, np.ndarray):
        linewidth = [linewidth] * N_curves

    if type(labels) is not list:
        labels = [labels] * N_curves

    if type(linestyle) is not list:
        linestyle = [linestyle] * N_curves

    if figuresize is None:
        fig = plt.figure(fig_number)
    else:
        fig = plt.figure(fig_number, figsize=figuresize[::-1])

    if show_compact:
        plt.subplots_adjust(wspace=subplots_wspace, hspace=subplots_hspace, left=subplots_left,
                            right=subplots_right, bottom=subplots_bottom, top=subplots_top)
    fig.clf()

    ax = fig.add_subplot(111)
    for i in range(0, N_curves):

        # Print line with preferred y axis style
        lines = eval("plt." + y_axis_style + "(x[i], y[i], label=labels[i])")
        # lines = plt.plot(x[i], y[i], label=labels[i])

        # Extract line object to adjust line settings
        line = lines[0]

        # Specify line settings
        line.set_linewidth(linewidth[i])
        line.set_linestyle(linestyle[i])
        if color is not None:
            line.set_color(color[i])
        if markers is not None:
            line.set_marker(markers[i])
            line.set_markevery(markevery)
            line.set_markerfacecolor(markerfacecolors)
            line.set_markersize(markersize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add legend
    # legend = plt.legend(loc=label_location, shadow=label_shadow, frameon=label_frameon)
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(flip(handles, 2), flip(labels, 2), loc="lower center",
    # shadow=label_shadow, frameon=label_frameon, bbox_to_anchor=(0.5, 1),
    # ncol=N_curves/2)

    if label_fontsize is None:
        label_fontsize = fontsize

    # legend = plt.legend(loc="lower center", shadow=label_shadow,
    # frameon=label_frameon, bbox_to_anchor=(0.47, 1), ncol=4,
    # fontsize=label_fontsize)
    legend = plt.legend(loc=label_location, shadow=label_shadow, frameon=label_frameon,
                        bbox_to_anchor=label_boundingboxtoanchor, ncol=label_ncol, fontsize=label_fontsize)

    ax.set_aspect(aspect_ratio)
    ax.set_facecolor(backgroundcolor)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if block_show:
        plt.show()
    else:
        plt.show(block=False)

    if save_figure:

        # Save figure to directory
        save_fig(fig, directory, filename)

    return fig


##
# Show 2D images
# \date       2017-02-18 05:55:08+0000
#
# \param      images            The images
# \param      titles            The title
# \param      cmap              The cmap ("Greys_r", "jet", ...)
# \param      colorbar          The color
# \param      fontfamily        The fontfamily ("serif", "sans-serif", "monospace")
# \param      fontname          The fontname ("Arial", "Times New Roman")
# \param      use_tex           The use tex
# \param      fontsize          The fontsize
# \param      backgroundcolor   The backgroundcolor
# \param      aspect_ratio      The aspect ratio ("auto", "equal")
# \param      save_figure       The save fig
# \param      directory         The directory
# \param      filename          The filename including extension
# \param      fig_number        The fig number
# \param      use_same_scaling  The use same scaling
# \param      show_compact      The show compact
# \param      subplots_wspace   The subplots wspace
# \param      subplots_hspace   The subplots hspace
# \param      subplots_left     The subplots left
# \param      subplots_right    The subplots right
# \param      subplots_bottom   The subplots bottom
# \param      subplots_top      The subplots top
#
# \return     figure
#
def show_images(images, titles=None, cmap="Greys_r", use_colorbar=False, fontfamily="serif", fontname="Arial", use_tex=False, fontsize=12, backgroundcolor="None", aspect_ratio="auto", save_figure=False, directory=DIR_TMP, filename="figure.pdf", fig_number=None, use_same_scaling=False, show_compact=False, subplots_wspace=0, subplots_hspace=0, subplots_left=0, subplots_right=1, subplots_bottom=0, subplots_top=0.85, grid_shape=None, figuresize=None):

    images = list(images)
    N_images = len(images)

    # Change font
    font = {
        "family":   fontfamily,
        fontfamily:   fontname,
        "size":   fontsize,
    }
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=use_tex)

    # Define the grid to arrange the slices
    if grid_shape is None:
        grid = _get_grid_size(N_images)
    else:
        grid = grid_shape

    # Use same scaling for all images
    if use_same_scaling:
        # Extract min and max value of arrays for same scaling
        value_min = np.min(images[0])
        value_max = np.max(images[0])
        for i in range(1, N_images):
            value_min = np.min([value_min, np.min(images[i])])
            value_max = np.max([value_max, np.max(images[i])])
    else:
        value_min = None
        value_max = None

    # print("value_min = %.2f" %(value_min))
    # print("value_max = %.2f" %(value_max))
    # Plot figure
    if show_compact:
        if figuresize is None:
            fig = plt.figure(fig_number, figsize=2 * np.array(grid[::-1]))
        else:
            fig = plt.figure(fig_number, figsize=figuresize[::-1])

        plt.subplots_adjust(wspace=subplots_wspace, hspace=subplots_hspace, left=subplots_left,
                            right=subplots_right, bottom=subplots_bottom, top=subplots_top)
    else:
        if figuresize is None:
            fig = plt.figure(fig_number)
        else:
            fig = plt.figure(fig_number, figsize=figuresize[::-1])

    fig.clf()

    for i in range(0, N_images):
        if i is 0:
            ax1 = plt.subplot(grid[0], grid[1], i + 1)
            ax1.set_aspect(aspect_ratio)
        else:
            # ax2 = plt.subplot(gs1[i])
            ax2 = plt.subplot(grid[0], grid[1], i + 1, sharex=ax1)
            ax2.set_aspect(aspect_ratio)
        im = plt.imshow(images[i], cmap=cmap, vmin=value_min, vmax=value_max)
        if titles is not None:
            if i < len(titles):
                plt.title(titles[i])

        # Suppress axes
        plt.axis('off')

        # Add colorbar on the side
        if use_colorbar:
            # add_axes([left, bottom, width, height])
            cax = fig.add_axes([0.92, 0.05, 0.01, 0.9])
            fig.colorbar(im, cax=cax)

    plt.show(block=False)

    if save_figure:

        # Create directory in case it does not exist already
        create_directory(directory)

        # Save figure to directory
        save_fig(fig, directory, filename)

    return fig


##
# Shows single 2D/3D array or a list of 2D arrays.
# \date       2017-02-07 10:06:25+0000
#
# \param      nda               Either 2D/3D numpy array or list of 2D numpy
#                               arrays #
# \param      title             The title of the figure
# \param      cmap              Color map "Greys_r", "jet", etc.
# \param      use_colorbar      The colorbar
# \param      directory         In case given, figure will be saved to this
#                               directory
# \param      filename          The filename
# \param      save_figure       Filename extension of figure in case it is
#                               saved
# \param      fig_number        Figure number. If 'None' previous figure will
#                               not be closed
# \param      use_same_scaling  The use same scaling
# \param      fontsize          The fontsize
# \param      fontname          "Arial", "Times New Roman" etc
#
def show_arrays(nda,
                title=None,
                cmap="Greys_r",
                use_colorbar=False,
                directory=None,
                filename=None,
                save_figure=0,
                fig_number=None,
                use_same_scaling=False,
                fontsize=None,
                fontname="Arial",
                ):

    # Show list of 2D arrays slice by slice
    if type(nda) is list:
        fig = _show_2D_array_list_array_by_array(
            nda,
            title=title,
            cmap=cmap,
            colorbar=use_colorbar,
            fig_number=fig_number,
            directory=directory,
            filename=filename,
            save_figure=save_figure,
            use_same_scaling=use_same_scaling,
            fontsize=fontsize,
            fontname=fontname,
        )

    # Show single 2D/3D array
    else:
        fig = show_array(
            nda,
            title=title,
            cmap=cmap,
            colorbar=use_colorbar,
            directory=directory,
            fig_number=fig_number,
        )

    return fig


##
# Show single 2D or 3D array
# \date       2017-02-07 10:22:58+0000
#
# \param      nda         Single 2D or 3D numpy array
# \param      title       The title of the figure
# \param      cmap        Color map "Greys_r", "jet", etc.
# \param      colorbar    The colorbar
# \param      directory   In case given, figure will be saved to this directory
# \param      save_type   Filename extension of figure in case it is saved
# \param      fig_number  Figure number. If 'None' previous figure will not be
#                         closed
#
def show_array(nda, title="data", cmap="Greys_r", colorbar=False, directory=None, save_type="pdf", fig_number=None):

    # Show single 2D array
    if len(nda.shape) is 2:
        fig = _show_2D_array(
            nda,
            title=title,
            cmap=cmap,
            colorbar=colorbar,
            directory=directory,
            save_type=save_type,
            fig_number=fig_number)

    # Show single 3D array
    elif len(nda.shape) is 3:
        fig = _show_3D_array_slice_by_slice(
            nda,
            title=title,
            cmap=cmap,
            colorbar=colorbar,
            directory=directory,
            save_type=save_type,
            fig_number=fig_number)

    return fig


##
# Shows data array and save it if desired
# \date       2016-11-07 21:29:13+0000
#
# \param      nda        Data array (only 2D so far)
# \param      title      The title of the figure
# \param      cmap       Color map "Greys_r", "jet", etc.
# \param      directory  In case given, figure will be saved to this directory
# \param      save_type  Filename extension of figure in case it is saved
#
def _show_2D_array(nda, title="data", cmap="Greys_r", colorbar=False, directory=None, save_type="pdf", fig_number=None):

    # Plot figure
    fig = plt.figure(fig_number)
    fig.clf()
    plt.imshow(nda, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if colorbar:
        plt.colorbar()

    plt.show(block=False)

    # If directory is given: Save
    if directory is not None:
        # Create directory in case it does not exist already
        create_directory(directory)

        # Save figure to directory
        save_fig(fig, directory, title + "." + save_type)

    return fig


##
#       Plot 3D numpy array slice by slice next to each other
# \date       2016-11-06 01:39:28+0000
#
# All slices in the x-y-plane are plotted. The number of slices is given by the
# dimension in the z-axis.
#
# \param      nda3D_zyx  3D numpy data array in format (z,y,x) as it is given
#                        after sitk.GetArrayFromImage for instance
# \param      title      The title of the figure
# \param      cmap       Color map "Greys_r", "jet", etc.
#
def _show_3D_array_slice_by_slice(nda3D_zyx, title="data", cmap="Greys_r", colorbar=False, directory=None, save_type="pdf", fig_number=None):

    shape = nda3D_zyx.shape
    N_slices = shape[0]

    # Define the grid to arrange the slices
    grid = _get_grid_size(N_slices)

    # Plot figure
    fig = plt.figure(fig_number)
    fig.clf()
    plt.suptitle(title)
    ctr = 1
    for i in range(0, N_slices):

        plt.subplot(grid[0], grid[1], ctr)
        plt.imshow(nda3D_zyx[i, :, :], cmap=cmap)
        plt.title(str(i))
        plt.axis('off')

        ctr += 1

    print("Slices of " + title + " are shown in separate window.")
    plt.show(block=False)

    # If directory is given: Save
    if directory is not None:
        # Create directory in case it does not exist already
        create_directory(directory)

        # Save figure to directory
        save_fig(fig, directory, title + "_slice_0_to_" +
                 str(N_slices - 1) + ".pdf")

    return fig


##
# Plot list of 2D numpy arrays next to each other
# \date       2016-11-06 02:02:36+0000
#
# \param      nda2D_list   List of 2D numpy data arrays
# \param      title        The title
# \param      cmap         Color map "Greys_r", "jet", etc.
# \param      colorbar     The colorbar
# \param      fig_number   The fig number
# \param      directory    The directory
# \param      save_type    The save type
# \param      axis_aspect  The axis aspect, Can be 'auto', 'equal'
#
# \return     { description_of_the_return_value }
#
def _show_2D_array_list_array_by_array(nda2D_list,
                                       title=None,
                                       cmap="Greys_r",
                                       colorbar=False,
                                       fig_number=None,
                                       directory=None,
                                       filename=None,
                                       save_figure=0,
                                       axis_aspect='equal',
                                       use_same_scaling=False,
                                       fontsize=8,
                                       fontname="Arial",
                                       ):

    title_font = {
        'fontname': fontname,
        'size': fontsize,
        # 'color': 'black',
        # 'weight': 'normal',
        # 'verticalalignment': 'center'  # Bottom vertical alignment for more space
    }

    shape = nda2D_list[0].shape
    N_slices = len(nda2D_list)

    if type(title) is not list and title is not None:
        title = [title]

    # Define the grid to arrange the slices
    grid = _get_grid_size(N_slices)

    if use_same_scaling:
        # Extract min and max value of arrays for same scaling
        value_min = np.min(nda2D_list[0])
        value_max = np.max(nda2D_list[0])
        for i in range(1, N_slices):
            value_min = np.min([value_min, np.min(nda2D_list[i])])
            value_max = np.max([value_max, np.max(nda2D_list[i])])
    else:
        value_min = None
        value_max = None

    # print("value_min = %.2f" %(value_min))
    # print("value_max = %.2f" %(value_max))
    # Plot figure
    # fig = plt.figure(fig_number, figsize=2*np.array(grid[::-1]))
    fig = plt.figure(fig_number)
    fig.clf()

    # gs1 = gridspec.GridSpec(grid[0], grid[1])
    # gs1.update(wspace=0, hspace=0, ) #set spacing between axes

    for i in range(0, N_slices):
        if i is 0:
            # ax1 = plt.subplot(gs1[i])
            ax1 = plt.subplot(grid[0], grid[1], i + 1)
            ax1.set_aspect(axis_aspect)
        else:
            # ax2 = plt.subplot(gs1[i])
            ax2 = plt.subplot(grid[0], grid[1], i + 1, sharex=ax1)
            ax2.set_aspect(axis_aspect)
        im = plt.imshow(nda2D_list[i], cmap=cmap,
                        vmin=value_min, vmax=value_max)
        if title is not None:
            if i < len(title):
                plt.title(
                    title[i],
                    **title_font
                )
        plt.axis('off')
        if colorbar:
            # add_axes([left, bottom, width, height])
            cax = fig.add_axes([0.92, 0.05, 0.01, 0.9])
            fig.colorbar(im, cax=cax)

    # plt.subplots_adjust(wspace=0.01, hspace=0.1, left=0,
    #                     right=1, bottom=0, top=None)
    print("Slices of data arrays are shown in separate window.")
    plt.show(block=False)

    if save_figure:
        # Create directory in case it does not exist already
        create_directory(directory)

        # Save figure to directory
        save_fig(fig, directory, filename)

    return fig


##
#       Gets the grid size given a number of 2D images
# \date       2016-11-06 02:02:20+0000
#
# \param      N_slices  The n slices
#
# \return     The grid size.
#
def _get_grid_size(N_slices):

    if N_slices > 40:
        raise ValueError("Too many slices to print")

    # Define the view grid to arrange the slices
    if N_slices <= 3:
        grid = (1, N_slices)
    elif N_slices > 3 and N_slices <= 10:
        grid = (2, np.ceil(N_slices / 2.).astype('int'))
    elif N_slices > 10 and N_slices <= 15:
        grid = (3, np.ceil(N_slices / 3.).astype('int'))
    elif N_slices > 15 and N_slices <= 20:
        grid = (4, np.ceil(N_slices / 3.).astype('int'))
    elif N_slices > 21 and N_slices <= 30:
        grid = (5, np.ceil(N_slices / 4.).astype('int'))
    else:
        grid = (6, np.ceil(N_slices / 5.).astype('int'))

    return grid


##
# Saves a figure to given directory
# \date       2017-02-07 10:19:09+0000
#
# \param      fig           The fig
# \param      path_to_file  The path to file
#
def save_fig(fig, path_to_file, transparent=True):

    directory = os.path.dirname(path_to_file)
    create_directory(directory)

    fig.savefig(path_to_file, transparent=transparent)
    print_info("Figure was saved to '%s'" % path_to_file)


##
# Closes all pyplot figures.
# \date       2017-02-07 10:30:57+0000
#
def close_all_figures():
    plt.close('all')


##
# Returns start time of execution
# \date       2016-11-06 17:15:00+0000
#
# \return     Start time of execution
#
def start_timing():
    return time.time()


##
# Gets value representing zero for datetime objects.
# \date       2017-08-08 16:25:05+0100
#
# \param      self  The object
#
# \return     The zero time.
#
def get_zero_time():
    return datetime.timedelta(seconds=0)


##
#       Stops a timing and returns the time passed between given start
#             time.
# \date       2016-11-06 17:18:42+0000
#
# Conversion of elapsed time to 'reasonable' format,  i.e. hours, minutes,
# seconds, ... as appropriate.
#
# \param      start_time  The start time obtained via \p start_timing
#
# \return     Elapsed time as string
#
def stop_timing(start_time):
    end_time = time.time()
    elapsed_time_sec = end_time - start_time

    # Convert to 'readable' format
    return datetime.timedelta(seconds=elapsed_time_sec)


##
# Gets the seconds from timedelta string.
# \date       2018-06-25 14:33:16-0600
#
# \param      timedelta_string  String with format "%H:%M:%S.%f". This is the
#                               standard output of t.time() in case t is a
#                               datetime.timedelta object
#
# \return     The total seconds from timedelta string. Microseconds are
#             ignored.
#
def get_seconds_from_timedelta_string(timedelta_string):
    try:
        t = datetime.datetime.strptime(timedelta_string, "%H:%M:%S.%f")
    except:
        t = datetime.datetime.strptime(timedelta_string, "%H:%M:%S")

    total_seconds = t.second + t.minute * 60 + t.hour * 3600
    return total_seconds


##
# Print numpy array in certain format via \p printoptions below
# \date       2016-11-21 12:56:19+0000
# \see        http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
#
# \param      nda        numpy array
# \param      precision  Specifies the number of significant digits
# \param      suppress   Specifies whether or not scientific notation is
#                        suppressed for small numbers
#
def print_numpy_array(nda, title=None, precision=3, suppress=False):
    with printoptions(precision=precision, suppress=suppress):
        if title is not None:
            sys.stdout.write(title + " = ")
            sys.stdout.flush()
        print(nda)


##
# Used in print_numpy_array to apply specific print formatting
# \see http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
#
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def print_info(text, newline=True, prefix="--- "):

    if newline:
        print(prefix + text)
    else:
        sys.stdout.write(prefix + text)
        sys.stdout.flush()


def print_warning(text, prefix="WARNING: ", symbol="X"):
    print_line_separator(symbol=symbol, add_newline=True)
    print_info(prefix + text)
    print_line_separator(symbol=symbol, add_newline=False)
    print("")


def print_title(text, symbol="*", add_newline=False):
    print_line_separator(symbol=symbol)
    print_subtitle(text, symbol=symbol, add_newline=add_newline)


def print_subtitle(text, symbol="*", add_newline=True):
    if add_newline:
        print("")
    print(3 * symbol + " " + text + " " + 3 * symbol)


def print_line_separator(add_newline=True, symbol="*", length=99):
    if add_newline:
        print("\n")
    print(symbol * length)


def print_execution(cmd):
    cmd = re.sub(" --", " \\\n--", cmd)
    print("")
    print("---- Executed command: ----")
    print(cmd)
    print("---------------------------")
    print("")


##
# Execute and show command in command window.
# \date       2016-12-06 17:37:57+0000
#
# \param      cmd           The command
# \param      verbose  The show command
#
# \return     { description_of_the_return_value }
#
def execute_command(cmd,
                    verbose=True,
                    flag_print_to_file=False,
                    path_to_file=None):

    if flag_print_to_file:
        write_to_file(path_to_file, cmd, verbose=verbose)
        return

    if verbose:
        print_execution(cmd)

    # Does not seem to work the way I want it:
    # This configuration only prints errors (and excludes warnings) - wohoo!
    # but does wait for ITK-SNAP to be closed again. Using stderr=devnull
    # does not wait anymore but also does not print any potential error message
    # with open(os.devnull, "wb") as devnull:
    #     process = subprocess.Popen(
    #         [cmd], shell=True, stdout=devnull, stderr=subprocess.PIPE)
    #     stdoutdata, stderrdata = process.communicate()
    #     flag = process.returncode
    #     if flag != 0:
    #         print stderrdata
    #     return flag
    # with open(os.devnull, "wb") as devnull:
        # flag = subprocess.call(cmd)
    flag = os.system(cmd)

    return flag


##
# Creates a directory on the HDD
# \date       2016-12-06 18:02:23+0000
#
# \param      directory     The directory
# \param      delete_files  The delete files
#
def create_directory(directory, delete_files=False, verbose=False):

    # Add slash in case not existing
    # if directory[-1] not in ["/"]:
    #     directory += "/"

    # Create directory in case it does not exist already
    if not os.path.isdir(directory) and directory is not "":
        os.makedirs(directory)
        if verbose:
            print_info("Directory " + directory + " created.")

    if delete_files:
        clear_directory(directory, verbose=verbose)

    return directory


##
# Clear all data in given directory
# \date       2017-02-02 16:47:15+0000
#
# \param      directory  The directory to be cleared
# \param      verbose    The verbose
#
def clear_directory(directory, verbose=True):

    if directory_exists(directory):
        shutil.rmtree(directory)
        if verbose:
            print_info("All files in " + directory + " are removed.")
    else:
        if verbose:
            print_info("Directory %s did not exist. It has been created now."
                       % directory)
    create_directory(directory)

    # if directory[-1] not in ["/"]:
    #     directory += "/"

    # os.system("rm -rf " + directory + "*")


##
# Delete directory
# \date       2017-08-08 16:37:03+0100
#
# \param      directory  The directory to be deleted
# \param      verbose    The verbose
#
def delete_directory(directory, verbose=True):
    os.system("rm -rf " + directory)
    if verbose:
        print_info("Directory " + directory + " deleted.")


def delete_file(path_to_file, verbose=True):
    os.remove(path_to_file)
    if verbose:
        print_info("File '%s' deleted." % path_to_file)


##
# Copy a file
# \date       2019-04-09 10:29:08+0100
#
# \param      path_to_src  The path to the source file
# \param      path_to_dst  Can be a path to a file or a directory (in which
#                          case the basename of src will be used as basename
#                          for the file at dst)
# \param      verbose      The verbose
#
def copy_file(path_to_src, path_to_dst, verbose=True):
    shutil.copy2(path_to_src, path_to_dst)
    if verbose:
        print_info("File '%s' copied to '%s'" % (path_to_src, path_to_dst))


##
# Move a file
# \date       2019-05-19 23:30:33-0400
#
# \param      path_to_src  The path to the source file
# \param      path_to_dst  The path to destination
# \param      verbose      The verbose
#
def move_file(path_to_src, path_to_dst, verbose=True):
    shutil.move(path_to_src, path_to_dst)
    if verbose:
        print_info("File '%s' moved to '%s'" % (path_to_src, path_to_dst))


##
# Gets the current date in format year, month and day
# \date       2017-08-08 16:34:17+0100
#
# \param      separator  separator between year, month and day; string
#
# \return     The current date as string
#
def get_current_date(separator=""):
    now = datetime.datetime.now()
    date = "%s%s%s%s%s" % (
        str(now.year),
        separator,
        str(now.month).zfill(2),
        separator,
        str(now.day).zfill(2),
    )
    return date


##
# Gets the current time in format hour, minute and second
# \date       2017-08-08 16:34:17+0100
#
# \param      separator  separator between hour, minute and second; string
#
# \return     The current time as string
#
def get_current_time(separator=""):
    now = datetime.datetime.now()
    time = "%s%s%s%s%s" % (
        str(now.hour).zfill(2),
        separator,
        str(now.minute).zfill(2),
        separator,
        str(now.second).zfill(2),
    )
    return time


##
# Gets the time stamp in format year, month, day, hour, minute and second
# \date       2017-08-08 16:35:37+0100
#
# \param      separator       separator between date and time; string
# \param      separator_date  separator in between date information; string
# \param      separator_time  separator in between time information; string
#
# \return     The time stamp.
#
def get_time_stamp(separator=", ", separator_date="-", separator_time=":"):
    time_stamp = "%s%s%s" % (
        get_current_date(separator=separator_date),
        separator,
        get_current_time(separator=separator_time),
    )
    return time_stamp


##
# Create a grid. Can be used to visualize deformation fields
# \date       2017-02-07 12:12:32+0000
#
# \param      shape     The shape
# \param      spacing   The spacing
# \param      value_fg  The value foreground
# \param      value_bg  The value background
#
# \return    image grid as numpy array
#
def create_image_grid(shape, spacing, value_bg=1, value_fg=0):

    nda = np.ones(shape) * value_bg

    for i in range(0, shape[0]):
        nda[i, 0::spacing] = value_fg

    for i in range(0, shape[1]):
        nda[0::spacing, i] = value_fg

    return nda


##
# Creates an image with slope.
# \date       2017-02-07 12:28:08+0000
#
# \param      shape     The shape
# \param      slope     The slope
# \param      value_fg  The value foreground
# \param      value_bg  The value background
# \param      offset    The offset
#
# \return     image with slope intensity as numpy array
#
def create_image_with_slope(shape, slope=1, value_bg=0, value_fg=1, offset=0):

    nda = np.ones(shape) * value_bg

    i = 0
    while i < nda.shape[0]:
        nda[i, :] = np.max([np.min([slope * i - offset, value_fg]), 0])
        i = i + 1

    return nda


##
# Creates an image pyramid.
# \date       2017-02-07 12:29:19+0000
#
# \param      shape     The shape
# \param      slope     The slope
# \param      value_fg  The value foreground
# \param      value_bg  The value background
#
# \return     { description_of_the_return_value }
#
def create_image_pyramid(length, slope=1, value_bg=0, value_fg=500, offset=(0, 0)):

    shape = np.array([length, length])

    nda = np.ones(shape) * value_bg

    for i in range(0, nda.shape[0] / 2):
        nda[i:-i, i:-i] = np.min([slope * i, value_fg])

    if np.abs(offset).sum() > 0:
        nda_offset = np.ones(shape) * value_bg
        nda_offset[offset[0]:, offset[1]:] = nda[0:-offset[0], 0:-offset[1]]

        nda = nda_offset

    return nda


##
# Reads an image by using Image
# \date       2017-02-10 11:16:34+0000
#
# \param      filename  The filename including filename extension. E.g. 'png',
#                       'jpg'
#
# \return     Image data as numpy array
#
def read_image(filename):
    return skimage.io.imread(filename)


def read_file_line_by_line(path_to_file):
    with open(path_to_file) as f:
        lines = f.readlines()

    return lines


##
# Writes a file line by line.
# \date       2018-01-19 12:23:05+0000
#
# \param      path_to_file  The path to file as string
# \param      lines         The lines as list
# \param      access_mode   The access mode
#
# \return     { description_of_the_return_value }
#
def write_file_line_by_line(path_to_file, lines, access_mode="w"):
    file_handle = safe_open(path_to_file, access_mode)
    for line in lines:
        file_handle.write(line)
    file_handle.close()
    print_info("Lines successfully written to %s." % path_to_file)


##
# Writes data array to image file
# \date       2017-02-10 11:18:21+0000
#
# \param      filename  The filename including filename extension
#
def write_image(nda, path_to_file, verbose=True, access_mode="w"):
    create_directory(os.path.dirname(path_to_file))
    skimage.io.imsave(path_to_file, nda)

    if verbose:
        print_info("Data array written to '%s'." % (path_to_file))


##
# Write to file.
# \date       2017-06-30 13:51:39+0100
#
# \param      path_to_file  Path to filename with extension. e.g. "txt"
# \param      text          The text
# \param      access_mode   The access mode
# \param      verbose       The verbose
# \param      type  "w" for write, "a" for append
#
# \return     { description_of_the_return_value }
#
def write_to_file(
    path_to_file,
    text,
    access_mode="w",
    verbose=True
):
    file_handle = safe_open(path_to_file, access_mode)
    file_handle.write(text)
    file_handle.close()
    if verbose:
        if access_mode == "w":
            print_info("File '%s' written" % (path_to_file))
        elif access_mode == "a":
            print_info("File '%s' updated" % (path_to_file))


def write_dictionary_to_json(dic, path_to_file, access_mode="w", verbose=True):
    create_directory(os.path.dirname(path_to_file))
    with open(path_to_file, access_mode) as fp:
        json.dump(dic, fp, sort_keys=True, indent=4)
        if verbose:
            print_info("File written to '%s'." % path_to_file)


def read_dictionary_from_json(path_to_file):
    try:
        with open(path_to_file) as json_file:
            dic = json.load(json_file)
    except json.decoder.JSONDecodeError as e:
        raise IOError("JSON file cannot be read. %s" % e)
    return dic


##
# Writes a numpy array to file.
# \date       2017-06-30 13:53:23+0100
#
# \param      path_to_file  Path to filename with extension. e.g. "txt"
# \param      array         Numpy array
# \param      format        The format
# \param      delimiter     The delimiter
# \param      access_mode   The access mode
# \param      verbose       The verbose
#
def write_array_to_file(
    path_to_file,
    array,
    format="%.10e",
    delimiter="\t",
    access_mode="a",
    verbose=True
):

    if not isinstance(array, np.ndarray):
        raise IOError("Given array must be of type np.ndarray")

    file_handle = safe_open(path_to_file, access_mode)
    np.savetxt(file_handle, array, fmt=format, delimiter=delimiter)
    file_handle.close()
    if verbose:
        print_info("Array written to '%s'" % (path_to_file))


##
# Strip filename extension from path
# \date       2018-04-23 16:12:41-0600
#
# \param      path_to_file  path to file, string
#
# \return     Return full path to file without filename extension
#
def strip_filename_extension(path_to_file):
    directory = os.path.dirname(path_to_file)
    basename = os.path.basename(path_to_file)

    splits = basename.split(".")
    index = len(splits) - 1

    # Check for known extension
    # Rationale: if a decimal point is in the filename, the simple search for a
    # separating point causes problems
    for known_extension in ["nii", "mhd", "txt"]:
        if known_extension in splits:
            index = splits.index(known_extension)
            continue

    basename_no_ext = ".".join(splits[0:index])
    extension = ".".join(splits[index:])

    return os.path.join(directory, basename_no_ext), extension


def replace_filename_extension(path_to_file, extension):
    filename_no_ext = strip_filename_extension(path_to_file)[0]
    return ".".join([filename_no_ext, extension])


##
# Inserts a suffix between filename and its extension
# \date       2017-11-30 21:05:12+0000
#
# \param      filename  The filename as string
# \param      suffix    The suffix as string
#
# \return appended filename as string
#
def append_to_filename(filename, suffix):

    filename_no_ext, ext = strip_filename_extension(filename)
    filename_no_ext += suffix

    return ".".join([filename_no_ext, ext])


##
# Convert list of numbers into a string including hyphenated ranges, e.g. [0,
# 1, 3, 5, 6, 7, 9, 10, 11] gets converted into '0-1, 3, 5-7, 9-11'
# \date       2019-03-07 18:05:48+0000
#
# \param      numbers_list  The numbers list
# \see        https://stackoverflow.com/questions/29418693/write-ranges-of-numbers-with-dashes/29418827
#
# \return     hyphenated ranges, str.
#
def convert_numbers_to_hyphenated_ranges(numbers_list):
    seq = []
    final = []
    last = 0

    for index, val in enumerate(sorted(numbers_list)):

        if last + 1 == val or index == 0:
            seq.append(val)
            last = val
        else:
            if len(seq) > 1:
                final.append(str(seq[0]) + '-' + str(seq[len(seq) - 1]))
            else:
                final.append(str(seq[0]))
            seq = []
            seq.append(val)
            last = val

        if index == len(numbers_list) - 1:
            if len(seq) > 1:
                final.append(str(seq[0]) + '-' + str(seq[len(seq) - 1]))
            else:
                final.append(str(seq[0]))

    final_str = ', '.join(map(str, final))
    return final_str
