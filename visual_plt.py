import matplotlib
matplotlib.use('Agg')
# above to lines set the matplotlib backend to 'Agg', which
#  enables matplotlib-plots to also be generated if no X-server
#  is defined (e.g., when running in basic Docker-container)
import matplotlib.pyplot as plt


def plot_lines(list_with_lines, x_axes=None, line_names=None):
    '''Generates a figure containing multiple lines in one plot.

    :param list_with_lines: <list> of all lines to plot (with each line being a <list> as well)
    :param x_axes:          <list> containing the values for the x-axis
    :param line_names:      <list> containing the names of each line
    :return: f:             <figure>
    '''

    # if needed, generate default x-axis
    if x_axes == None:
        n_obs = len(list_with_lines[0])
        x_axes = list(range(n_obs))

    # if needed, generate default line-names
    if line_names == None:
        n_lines = len(list_with_lines)
        line_names = ["line " + str(line_id) for line_id in range(n_lines)]

    # make plot
    f, axarr = plt.subplots(1, 1, figsize=(12, 7))
    for task_id, name in enumerate(line_names):
        axarr.plot(x_axes, list_with_lines[task_id], label=name)
    axarr.legend()

    # return the figure
    return f