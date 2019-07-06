
import re
import six
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import collections

import scipy.stats
import pysitk.python_helper as ph

# Increase with so that there is no linebreak for wider tables
pd.set_option('display.width', 1000)


##
# Prints a table from array(s)
# \date       2018-02-02 19:14:26+0000
#
# \param      nda      (m x n) numpy data array
# \param      nda_std  optional (m x n) numpy array. If given 'nda' interpreted
#                      as means and 'nda_std' as standard deviation values
# \param      rows     List of strings to label the m rows
# \param      cols     List of strings to label the n columns
#
def print_table_from_array(nda, nda_std=None, rows=None, cols=None):

    if nda_std is None:
        nda_print = nda
    else:
        nda_print = np.zeros((nda.shape[0], 2 * nda.shape[1]))
        if cols is None:
            cols = [""] * nda.shape[1]
        cols = ["%s (%s)" % (m, t)
                for (m, t) in itertools.product(cols, ["mean", "std"])]

        nda_print[:, 0::2] = nda
        nda_print[:, 1::2] = nda_std
    df = pd.DataFrame(nda_print, index=rows, columns=cols)
    print(df)


def print_table_from_data_dic(data_dic, x_label, labels):
    nda, nda_std = get_arrays_from_data_dic(data_dic)
    rows = ["%s %s" % (x_label, str(key)) for key in data_dic.keys()]
    print_table_from_array(nda=nda, nda_std=nda_std, rows=rows, cols=labels)


def get_arrays_from_data_dic(data_dic):
    x_label_entries = len(data_dic.keys())
    x_label_entry_data = len(data_dic[tuple(data_dic.keys())[0]])
    nda = np.zeros((x_label_entries, x_label_entry_data))
    nda_std = np.zeros_like(nda)

    for (i, key), j in itertools.product(
            enumerate(data_dic.keys()), range(x_label_entry_data)):
        nda[i, j] = np.nanmean(data_dic[key][j])
        nda_std[i, j] = np.nanstd(data_dic[key][j])
    return nda, nda_std


##
# Writes an array to latex table file.
# \date       2018-12-15 20:34:50+0000
#
# \param      path_to_file    The path to file, string
# \param      nda             numpy data array
# \param      nda_std         standard deviation, numpy data array
# \param      nda_sig         significance outcome, bool numpy data array
# \param      rows            row titles as list of strings
# \param      cols            column titles as list of string
# \param      row_title       additional 'row title', i.e. left-upper corner,
#                             string
# \param      decimal_places  Number of decimal places to be printed, integer
# \param      compact         compact style, i.e. remove white spaces, bool
# \param      mark_best       List (or single string) indicating 'best' values
#                             per row, e.g. mark_best=["max", "max", "min"] or
#                             mark_best="max"; list length must match
#                             len(cols); best values (as well as ties) will be
#                             shown in bold
# \param      verbose         verbose output, bool
#
def write_array_to_latex(
        path_to_file,
        nda,
        nda_std=None,
        nda_sig=None,
        rows=None,
        cols=None,
        row_title=None,
        decimal_places=2,
        compact=False,
        nda_bold=None,
        mark_best=None,
        verbose=True,
):

    lines = []
    sep = " & "
    newline = " \\\\\n"
    nan_symbol = "---"

    if mark_best is not None and nda_bold is not None:
        raise ValueError("Either 'mark_best' or 'nda_bold' but not both.")

    # Mark best row values bold (ties are allowed)
    if mark_best is not None:
        e = "'mark_best' must be either a list with " \
            "'min' or 'max' as elements with len(mark_best) == len(cols), " \
            "or a single string being either 'min' or 'max'"
        if type(mark_best) is not list:
            if mark_best != "min" and mark_best != "max":
                raise ValueError(e)
            mark_best = [mark_best] * len(cols)
        else:
            if len(mark_best) != len(cols):
                raise ValueError(e)
            if not all(m in ["min", "max"] for m in mark_best):
                raise ValueError(e)

        # round to output decimal places
        nda_ = nda.round(decimal_places)

        # find best values along rows per column
        i_best = []
        for j in range(nda.shape[1]):
            topper = getattr(np, "arg%s" % mark_best[j])(nda_[:, j])
            i_best.append(topper)

        # find 'ties' that have the same value within each column
        j_i_best = {
            j: np.where(nda_[:, j] == nda_[i_best[j], j])[0]
            for j in range(nda.shape[1])
        }

        # reverse mapping for easier access later
        i_j_best = {i: [] for i in range(nda.shape[0])}
        for j, i_list in six.iteritems(j_i_best):
            for i in i_list:
                i_j_best[i].append(j)

    # Statistical significance (only printed if nda_sig given)
    sym = "$^*$"
    nda_sym = np.chararray(nda.shape, itemsize=8)
    if nda_sig is not None:
        for index, val in np.ndenumerate(nda_sig):
            if val:
                nda_sym[index] = sym
            else:
                nda_sym[index] = ""
    else:
        nda_sym[:] = ""

    # \begin{tabular}{tabular_options}
    tabular_options = 'c' * nda.shape[1]
    if rows is not None:
        tabular_options = 'l' + tabular_options
    lines.append("\\begin{tabular}{%s}\n" % tabular_options)

    # Header: column titles of table
    line_args = ["\\bf %s" % c for c in cols]
    if rows is not None:
        if row_title is None:
            line_args.insert(0, "")
        else:
            line_args.insert(0, "\\bf %s" % row_title)
    lines.append("%s%s" % (sep.join(line_args), newline))
    lines.append("\\hline\n")

    # Entries of the table
    for i_row in range(nda.shape[0]):
        if nda_std is None:
            line_args = [
                # "\\num{%s}" % (
                "%s%s" % (
                    '{:.{prec}f}'.format(f, prec=decimal_places),
                    p,
                ) if not np.isnan(f)
                else nan_symbol
                for (f, p) in zip(nda[i_row, :], nda_sym[i_row, :])
            ]
        else:
            line_args = [
                # "\\num{%s \\pm %s}" % (
                "%s $\\pm$ %s%s" % (
                    '{:.{prec}f}'.format(m, prec=decimal_places),
                    '{:.{prec}f}'.format(s, prec=decimal_places),
                    p,
                ) if not np.isnan(m)
                else nan_symbol
                for (m, s, p) in zip(nda[i_row, :], nda_std[i_row, :], nda_sym[i_row, :])
            ]
            if compact:
                # remove white spaces
                line_args = [re.sub(" ", "", l) for l in line_args]

        if mark_best:
            for j in i_j_best[i_row]:
                line_args[j] = "\\bf %s" % line_args[j]

        if nda_bold is not None:
            for j in range(nda_bold.shape[1]):
                if nda_bold[i_row, j]:
                    line_args[j] = "\\bf %s" % line_args[j]

        if rows is not None:
            line_args.insert(0, "\\bf %s" % rows[i_row])
        lines.append("%s%s" % (sep.join(line_args), newline))

    # \end{tabular}
    lines.append("\\end{tabular}")

    text = "".join(lines)
    if verbose:
        print(text)
    ph.write_to_file(path_to_file, text, access_mode="w", verbose=True)


##
# Writes an array to a file in csv style.
# \date       2018-06-26 09:36:22-0600
#
# \param      nda             numpy data array
# \param      path_to_file    The path to file as string
# \param      decimal_places  Number of decimals to be written
# \param      cols            Column labels as list
# \param      rows            Row labels as list
# \param      access_mode     Python write mode ("a" for append, "w" for write)
#
def write_array_to_csv_file(
        nda,
        path_to_file,
        decimal_places=2,
        cols=None,
        rows=None,
        access_mode="w"):

    # Round to selected number of significant places
    nda = np.round(nda, decimals=decimal_places)

    df = pd.DataFrame(nda, columns=cols, index=rows)
    if rows is None:
        index = False
    else:
        index = True

    if access_mode == "a":
        df.to_csv(path_to_file, mode=access_mode, header=False, index=index)
    else:
        df.to_csv(path_to_file, mode=access_mode, index=index)


##
# Helper to make current figure fullscreen
# \date       2018-02-02 19:22:54+0000
#
def make_figure_fullscreen():
    try:
        # Open windows (and also save them) in full screen
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
    except:
        pass


##
# Shows the boxplot.
# \see        https://stackoverflow.com/questions/44975337/side-by-side-boxplots-with-pandas
# \date       2018-02-02 18:45:11+0000
#
# \param      data_dic  Data given as dictionary. E.g.
#                       { 'group1': [np.array(), np.array(), np.array()],
#                         'group2': [np.array(), np.array(), np.array()],
#                         ... }
#                       with three arrays associated with labels.
# \param      x_label   String describing the groups. E.g. 'study' or 'subject'
# \param      labels    List of strings linking the data arrays to a label.
#                       E.g. labels = ["Ours", "IRTK", "BTK"]
# \param      ref       Overarching characteristic for labels; E.g. 'method'
#
# \return     handle to sns.boxplot
#
def show_boxplot(data_dic,
                 x_label,
                 labels,
                 ref="cls",
                 y_label="value",
                 palette=None,
                 # palette="husl"
                 # palette="hls"
                 # palette="Set1"
                 show_points=True,
                 show_legend=True,
                 ):

    # Get maximum array length over all groups and labels
    # Rationale: Create joint-maximum array where NaN's are used for padding
    # in case different groups have different sample size.
    N_labels = np.zeros(len(labels)).astype(np.int)
    for k in data_dic.keys():
        for i_label in range(N_labels.size):
            N_labels[i_label] = np.max(
                [N_labels[i_label], len(data_dic[k][i_label])])

    # Create classes as flattened string. E.g.
    # classes = ['label1', .., 'label1', 'label2', ..., 'label2', 'label3' ...]
    classes_list = [[labels[i]] * N_labels[i] for i in range(N_labels.size)]
    classes = [label for class_list in classes_list for label in class_list]

    # Get starting index for each label within joint-maximum array
    N_labels_i0 = [np.sum(N_labels[0:k]) for k in range(len(N_labels))]

    # Fill in data
    data = {}
    data[ref] = classes
    for k in data_dic.keys():

        # Initialize as NaN-array
        data[k] = np.full(len(classes), np.NaN)

        for i_label in range(len(labels)):
            N_vars = len(data_dic[k][i_label])
            i0 = int(N_labels_i0[i_label])
            i1 = i0 + len(data_dic[k][i_label])
            data[k][i0: i1] = data_dic[k][i_label].astype(np.float)
        # data[k][np.isnan(data[k])] = 0.

    df = pd.DataFrame(data)
    df_melt = df.melt(
        id_vars=ref,
        value_vars=data_dic.keys(),
        var_name=x_label,
        value_name=y_label,
    )

    sns.set(style="ticks")
    # sns.set_style("whitegrid")
    b = sns.boxplot(
        x=x_label,
        y=y_label,
        data=df_melt,
        hue=ref,  # different colors for different ref
        width=0.5,
        # y=y_label,
        palette=palette,
        order=data_dic.keys(),
    )
    if show_points:
        sns.stripplot(
            x=x_label,
            y=y_label,
            data=df_melt,
            hue=ref,
            size=7,
            edgecolor="black",
            linewidth=1,
            palette=palette,
            split=True,
        )
        b_handles, b_labels = b.get_legend_handles_labels()
        n_labels = len(data_dic.keys())
        l = plt.legend(b_handles[0:len(labels)], b_labels[0:len(labels)],
                       ncol=len(labels), mode="expand", loc="lower center", bbox_to_anchor=(0, 0.95, 1, 0.2),
                       )
    else:
        l = plt.legend(ncol=len(labels), mode="expand", loc="lower center",
                       bbox_to_anchor=(0, 0.95, 1, 0.2),
                       )

    if not show_legend:
        l = plt.legend([])
    sns.despine(offset=10, trim=True)

    # sns.set_style("whitegrid")

    return b


##
# Run t-test on two related samples of scores
# \date       2018-02-26 13:50:09+0000
#
# \param      x           array_like
# \param      y           array_like, same shape as x
# \param      axis        int, axis along which to compute test
# \param      nan_policy  {'propagate', 'raise', 'omit'}
#
# \return     t-statistic, two-tailed p-value
#
def run_t_test_related(x, y, axis=0, nan_policy='omit'):
    statistic, p_value = scipy.stats.ttest_rel(
        x, y, axis=axis, nan_policy=nan_policy)
    return statistic, p_value


##
# Run t-test for the means of two independent samples of scores
# \date       2018-02-26 13:52:41+0000
#
# \param      x           array_like
# \param      y           array_like
# \param      axis        int, axis along which to compute test
# \param      equal_var   bool, If True (default), perform a standard
#                         independent 2 sample test that assumes equal
#                         population variances
# \param      nan_policy  {'propagate', 'raise', 'omit'}
#
# \return     t-statistic, two-sided p-value for test
#
def run_t_test_independent(x, y, axis=0, equal_var=True, nan_policy='omit'):
    statistic, p_value = scipy.stats.ttest_ind(
        x, y, axis=axis, equal_var=equal_var, nan_policy=nan_policy)
    return statistic, p_value


##
# Calculate the Wilcoxon signed-rank test
# \date       2018-03-07 11:32:14+0000
#
# \param      x            array_like
# \param      y            array_like
# \param      zero_method  string, {"pratt", "wilcox", "zsplit"}
# \param      correction   bool; apply continuity correction by adjusting the
#                          Wilcoxon rank statistic
#
# \return     statistic, two-sided p-value for test
#
def run_wilkoxon_test(x, y=None, zero_method="wilcox", correction=False):
    statistic, p_value = scipy.stats.wilcoxon(
        x, y=y, zero_method=zero_method, correction=correction)
    return statistic, p_value


##
# Shows the scatter plot.
# \date       2018-02-02 19:23:51+0000
#
# \param      x        { parameter_description }
# \param      y        { parameter_description }
# \param      x_label  The x label
# \param      y_label  The y label
#
# \return     { description_of_the_return_value }
#
def show_scatter_plot(x, y, x_label, y_label):
    color_points = ph.COLORS_TABLEAU20[4]
    color_lines = ph.COLORS_TABLEAU20[2]
    marker_points = "o"
    markerfacecolor_points = "w"

    plt.plot(x, y,
             color=color_points,
             marker=marker_points,
             # markerfacecolor=markerfacecolor_points,
             linestyle="")

    # Plot identity as orientation line
    axes = plt.gca()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    window = [np.min([xmin, ymin]), np.max([xmax, ymax])]
    plt.plot(window, window,
             color=color_lines,
             linestyle="--",
             )
    plt.axis('equal')
    plt.xlabel(x_label)
    plt.ylabel(y_label)


##
# Shows the Bland Altman plot.
#
# \date       2018-02-02 19:24:03+0000
#
# \param      x        { parameter_description }
# \param      y        { parameter_description }
# \param      x_label  The x label
# \param      y_label  The y label
#
def show_bland_altman_plot(
        x,
        y,
        x_label,
        y_label,
        # color_points=ph.COLORS_TABLEAU20[4],  # green
        color_points=ph.COLORS_TABLEAU20[0],  # blue
        marker_points="o",
        color_lines=ph.COLORS_TABLEAU20[2],  # orange
        bubble_chart=False,
        annotations=True,
        non_parametric=False,
):

    # markerfacecolor_points="white",

    axes = plt.gca()

    # Plot zero line
    axes.axhline(y=0, color='k', linewidth=1, linestyle="-")

    # Plot data
    if not bubble_chart:
        plt.plot((x + y) / 2., x - y,
                 color=color_points,
                 marker=marker_points,
                 # markerfacecolor=markerfacecolor_points,
                 linestyle="",
                 )
    else:
        # Count frequency of repeating points in list
        points = [(i, j) for i, j in zip((x + y) / 2., x - y)]
        counter = collections.Counter(points)

        # Extract unique points for x and y in scatter plot
        a = [k[0] for k in counter.keys()]
        b = [k[1] for k in counter.keys()]

        # Extract frequency of unique points; scale for visualization
        s = np.array(counter.values())
        s *= 50

        plt.scatter(a, b, s=s,
                    color=color_points,
                    marker=marker_points,
                    )

    if non_parametric:
        labels = ["Median", "95% LoA"]
        mid = np.median(x - y)
        upper = np.percentile(x - y, 97.5)
        lower = np.percentile(x - y, 2.5)
    else:
        labels = ["Mean", "95% LoA"]
        # Mean and mean +- 1.96 sigma helper lines
        mid = np.mean(x - y)
        sigma = np.std(x - y)
        upper = mid + sigma * 1.96
        lower = mid - sigma * 1.96

    axes.axhline(y=upper, color=color_lines, linewidth=1, linestyle="--")
    axes.axhline(y=mid, color=color_lines, linewidth=1, linestyle="-.")
    axes.axhline(y=lower, color=color_lines, linewidth=1, linestyle="--")

    if annotations:
        xlim = np.array(axes.get_xlim())
        ylim = np.array(axes.get_ylim())
        center = np.mean(xlim)
        offset_x = (xlim[1] - xlim[0]) / 50.
        offset_y = (ylim[1] - ylim[0]) / 20.
        if mid != upper:
            plt.text(xlim[1] - offset_x, mid - offset_y, labels[0],
                     color=color_lines, va='center', ha='right', size='smaller')
            plt.text(xlim[1] - offset_x, upper - offset_y, labels[1],
                     color=color_lines, va='center', ha='right', size='smaller')
        elif mid != lower:
            plt.text(xlim[1] - offset_x, mid + offset_y, labels[0],
                     color=color_lines, va='center', ha='right', size='smaller')
            plt.text(xlim[1] - offset_x, lower + offset_y, labels[1],
                     color=color_lines, va='center', ha='right', size='smaller')
        else:
            plt.text(xlim[1] - offset_x, mid - offset_y, ", ".join(labels),
                     color=color_lines, va='center', ha='right', size='smaller')
        # plt.text(xlim[1] - offset_x, lower - offset_y, '$\mu-1.96\sigma$',
        #          color=color_lines, va='center', ha='right', size='smaller')

    plt.xlabel("(%s + %s) / 2" % (x_label, y_label))
    plt.ylabel("%s - %s" % (x_label, y_label))
