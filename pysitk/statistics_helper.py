
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

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


def write_array_to_latex(
        path_to_file,
        nda,
        nda_std=None,
        rows=None,
        cols=None,
        row_title=None,
        decimal_places=2,
        verbose=0,
):

    lines = []
    sep = " & "

    # \begin{tabular}{tabular_options}
    tabular_options = 'c' * nda.shape[1]
    if rows is not None:
        tabular_options = 'l' + tabular_options
    lines.append("\\begin{tabular}{%s}" % tabular_options)

    # Header: column titles of table
    line_args = ["\\bf %s" % c for c in cols]
    if rows is not None:
        if row_title is None:
            line_args.insert(0, "")
        else:
            line_args.insert(0, "\\bf %s" % row_title)
    lines.append(sep.join(line_args))
    lines.append("\\hline")

    # Entries of the table
    for i_row in range(nda.shape[0]):
        line_args = []
        if nda_std is None:
            line_args = ["\\num{%.2f}" % f for f in nda[i_row, :]]
        else:
            line_args = ["\\num{%.2f \\pm %.2f}" % (m, s)
                         for (m, s) in zip(nda[i_row, :], nda_std[i_row, :])]
        if rows is not None:
            line_args.insert(0, "\\bf %s" % rows[i_row])
        lines.append(sep.join(line_args))

    # \end{tabular}
    lines.append("\\end{tabular}")

    text = " \\\\\n".join(lines)
    ph.write_to_file(path_to_file, text, access_mode="w", verbose=True)
    if verbose:
        print(text)


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
def show_boxplot(data_dic, x_label, labels, ref="cls"):

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
    )

    b = sns.boxplot(
        data=df_melt,
        hue=ref,  # different colors for different ref
        x=x_label,
        y="value",  # only y="value" works!?!
        # y=y_label,
        # palette="Set1",
        order=sorted(data_dic.keys()),
    )
    # sns.set_style("whitegrid")

    return b


##
# Run t-test on two related samples of scores
# \date       2018-02-26 13:50:09+0000
#
# \param      x           array_like
# \param      y           array_like, same shape as x
# \param      axis        int, axis along which to compute test
# \param      nan_policy  {'propagate’, 'raise’, 'omit'}
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
# \param      nan_policy  {'propagate’, 'raise’, 'omit'}
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
def show_bland_altman_plot(x, y, x_label, y_label):

    marker_points = "o"
    color_points = ph.COLORS_TABLEAU20[4]
    color_lines = ph.COLORS_TABLEAU20[2]
    markerfacecolor_points = "white",

    # Plot data
    plt.plot((x + y) / 2., x - y,
             color=color_points,
             marker=marker_points,
             # markerfacecolor=markerfacecolor_points,
             linestyle="")

    axes = plt.gca()
    xmin, xmax = axes.get_xlim()

    # Plot mean and mean +- 1.96 sigma helper lines
    mu = np.mean(x - y)
    sigma = np.std(x - y)
    plt.plot([xmax, xmin], np.ones(2) * mu,
             color=color_lines, linestyle="-")
    plt.plot([xmax, xmin], np.ones(2) * (mu + sigma * 1.96),
             color=color_lines, linestyle="--")
    plt.plot([xmax, xmin], np.ones(2) * (mu - sigma * 1.96),
             color=color_lines, linestyle="--")
    plt.plot([xmax, xmin], np.ones(2) * mu,
             color=color_lines, linestyle="-")

    # Plot zero line
    axes.axhline(y=0, color='k')

    plt.xlabel("(%s + %s)/2" % (x_label, y_label))
    plt.ylabel("%s - %s" % (x_label, y_label))
