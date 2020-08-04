import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.ndimage import gaussian_filter1d


def value_grid(
    axes: plt.Axes,
    values: np.ndarray,
    show_values: bool = False,
    agent_position: Tuple[int, int] = None,
    abs_colors: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    fontsize: int = 6,
):
    # Colormap where blue means >0 and red means <0.
    cmap = plt.get_cmap('bwr_r')

    # Override mins/maxes.
    if min_val is None:
        min_val = values.min()
    if max_val is None:
        max_val = values.max()

    # If abs_colors is given, then min/max will be scaled so that 0 is the middle of the colormap. Otherwise,
    # they'll be relative.
    if abs_colors:
        o_min_val = min_val
        min_val = -max(abs(min_val), max(max_val, 0))
        max_val = max(abs(o_min_val), max(max_val, 0))

    # Calculate tick values and put them in 15 bins (16 ticks). Then, map them to the space of colors.
    levels = MaxNLocator(nbins=127).tick_values(min_val, max_val)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Plot the grid.
    im = axes.pcolormesh(values, cmap=cmap, norm=norm)

    # Formatting.
    axes.set_ylim(axes.get_ylim()[::-1])  # Y axis is inverted (0 at the top).
    axes.xaxis.tick_top()  # Put the x axis on the top (even though it's hidden.)
    axes.minorticks_on()  # Include minor ticks.
    axes.grid(True, color="black", lw=0.2, which='both')  # Show the major + minor grid.
    axes.xaxis.set_minor_locator(MultipleLocator(1))  # Grid at every integer (x axis).
    axes.yaxis.set_minor_locator(MultipleLocator(1))  # Grid at every integer (y axis).

    # Turn off all ticks. We don't need them.
    axes.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        right=False,
        labelleft=False,
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
    )

    # Optionally, draw values.
    if show_values:
        for i in range(len(values)):
            for j in range(len(values[i])):
                axes.text(
                    j + 0.5,
                    i + 0.5,
                    f'{values[i][j]:.2f}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=fontsize,
                )

    # Optionally, show the position of the agent.
    if agent_position is not None:
        if agent_position is not None:
            axes.text(
                agent_position[1] + 0.5,
                agent_position[0] + 0.5,
                'X',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10,
                color='black',
                fontstretch='ultra-expanded',
                fontweight='bold',
            )


class ValuePlotter:
    """Somewhat hacky wrapper that allows for very simple plotting and updating of tabular Q values.
    """

    def __init__(
        self,
        max_width: int = 3,
        abs_colors: bool = False,
        show_plots: bool = True,
        name: Optional[str] = None,
    ):
        self.plots = {}
        self.fig = plt.figure()
        self.show_plots = show_plots
        if self.show_plots:
            # plt.ion()
            self.fig.show()
            # plt.show()
        self.name = name
        if name is not None:
            self.fig.suptitle(name)
        self.max_width = max_width
        self.n_plots = 0  # Total number of plots in the graph.
        self.order = []
        self.abs_colors = abs_colors

    def _redraw(self, n_new_subplots: int):
        # delete all the axes.
        while len(self.fig.axes) > 0:
            self.fig.delaxes(self.fig.axes[0])

        current_fig = 1
        rows = n_new_subplots // self.max_width + 1
        columns = min(n_new_subplots, self.max_width)

        for name in self.order:
            plot_params = self.plots[name]
            if "all_q_axes" in plot_params:
                new_subplots = []
                for action in range(plot_params["q"].shape[-1]):
                    new_subplot = self.fig.add_subplot(rows, columns, current_fig)
                    current_fig += 1
                    new_subplots.append(new_subplot)
                self.plots[name]["axes"] = new_subplots
                self.update_q_func(name, self.plots[name]["q"])
            else:
                new_subplot = self.fig.add_subplot(rows, columns, current_fig)
                self.plots[name]["axes"] = new_subplot
                self.update_grid(name, self.plots[name]["value"])
                current_fig += 1

    def add_q_func(
        self,
        name: str,
        q: np.ndarray,
        action_labels: Optional[List[str]] = None,
        show_values: bool = False,
    ):

        if action_labels is not None:
            assert len(action_labels) == q.shape[-1]
        else:
            action_labels = [f"{name}_action_{i}" for i in range(q.shape[-1])]

        # Only 3D q functions.
        assert len(q.shape) == 3
        assert name not in self.plots

        new_n_plots = self.n_plots + q.shape[-1]
        row = new_n_plots // self.max_width + 1
        column = min(new_n_plots, self.max_width)

        # First, redraw the layout.
        self._redraw(new_n_plots)

        all_q_axes = []
        for action in range(q.shape[-1]):
            # Keep track.
            self.n_plots += 1

            # Create a new axis.
            print(
                f"adding q plot at index {self.n_plots} in a grid of: {row}, {column}"
            )
            q_axes = self.fig.add_subplot(row, column, self.n_plots)
            if self.show_plots:
                plt.pause(0.0001)

            # Append q axes.
            all_q_axes.append(q_axes)

        self.plots[name] = {
            "all_q_axes": all_q_axes,
            "labels": action_labels,
            "q": q,
            "show_values": show_values,
        }

        # Make sure that this plot is replotted in order.
        self.order.append(name)

        self.update_q_func(name, q)

    def add_grid(self, name: str, values: np.ndarray, show_values: bool = False):
        # Keep track.
        assert name not in self.plots

        # Redraw to take a new plot.
        self._redraw(self.n_plots + 1)

        self.n_plots += 1

        # Create a new axis.
        row = self.n_plots // self.max_width + 1
        column = min(self.n_plots, self.max_width)

        # since we're 1 indexed, the maxwidth-th element should be last, not first.
        if column == 0:
            column = self.max_width

        print(f"adding plot at index {self.n_plots} in a grid of: {row}, {column}")
        axes = self.fig.add_subplot(row, column, self.n_plots)

        self.plots[name] = {"axes": axes, "show_values": show_values, "value": values}

        # Make sure that this plot is replotted in order.
        self.order.append(name)

        self.update_grid(name, values)

    def update_grid(self, name: str, values: np.ndarray):
        plot = self.plots[name]
        axis = plot['axes']
        show_values = plot['show_values']
        self.plots[name]['value'] = values

        # Clear the action.
        axis.clear()

        # Set the value plot..
        value_grid(axis, values, show_values, abs_colors=self.abs_colors)
        axis.set_title(name)

        # Redraw.
        if self.show_plots:
            self.fig.canvas.draw()
            plt.pause(0.0001)

    def update_q_func(self, name: str, q: np.ndarray):
        plots = self.plots[name]

        all_q_axes = plots["all_q_axes"]
        labels = plots["labels"]
        show_values = plots["show_values"]
        self.plots[name]['q'] = q

        for action in range(q.shape[-1]):
            q_action = q[:, :, action]
            axis = all_q_axes[action]
            label = labels[action]

            # Clear the action.
            axis.clear()

            # Set the value plot, but with min/max scaled so that q colors are consistent across plots.
            value_grid(
                axis,
                q_action,
                show_values,
                abs_colors=self.abs_colors,
                min_val=q.min(),
                max_val=q.max(),
            )
            axis.set_title(label)
            # self.fig.canvas.draw()
            # plt.pause(0.01)

        # Redraw things. again?
        if self.show_plots:
            self.fig.canvas.draw()
            plt.pause(0.0001)

    def save(self, logdir: str, step: int, valname: Optional[str] = None):
        if valname is None:
            self.fig.savefig(os.path.join(logdir, f"plot_{step}.png"))
        else:
            # Select the correct subplot.
            plot = self.plots[valname]
            if "q" in plot:
                for label, ax in zip(plot['labels'], plot['all_q_axes']):
                    extent = ax.get_window_extent().transformed(
                        self.fig.dpi_scale_trans.inverted()
                    )
                    self.fig.savefig(
                        os.path.join(logdir, f"{valname}_{label}_{step}.png"),
                        bbox_inches=extent,
                    )
            else:
                extent = (
                    plot['axes']
                    .get_window_extent()
                    .transformed(self.fig.dpi_scale_trans.inverted())
                )
                self.fig.savefig(
                    os.path.join(logdir, f"{valname}_{step}.png"), bbox_inches=extent
                )

    def close(self):
        plt.close(self.fig)


def play_trajectory(
    r_grid: np.ndarray,
    positions: List[Tuple[int, int]],
    output_dir: Optional[str] = None,
    visualize: bool = True,
):
    fig = plt.figure()
    axes = fig.add_subplot()
    value_grid(axes, r_grid)

    # import pdb; pdb.set_trace()
    def update(iter):
        position = positions[iter]
        axes.clear()
        value_grid(axes, r_grid, agent_position=position)
        if visualize:
            plt.pause(0.0001)
            fig.canvas.draw()

    anim = animation.FuncAnimation(
        fig, update, frames=len(positions), blit=False, repeat=False, interval=100
    )
    if output_dir is not None:
        # Set up formatting for the movie files
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(output_dir)
    if visualize:
        plt.ion()
        plt.show()
    plt.close(fig)


def plot_training_runs(
    training_runs: List[List[float]],
    val_runs: Optional[List[List[float]]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Training Results",
    smooth: Optional[int] = 10,
    f: Optional[plt.Figure] = None,
):
    if f is None:
        f: plt.Figure = plt.figure()
    f.suptitle(title)

    if val_runs is not None:
        n_cols = 2
    else:
        n_cols = 1

    if labels is None:
        labels = [f"run_{i}" for i in range(len(training_runs))]

    def plot_runs(runs, run_labels, index, title):
        ax: plt.Axes = f.add_subplot(1, n_cols, index)
        n_eps = len(runs[0])
        x = [i for i in range(n_eps)]
        ax.set_title(title)
        if smooth is not None:
            runs = [gaussian_filter1d(run, sigma=smooth) for run in runs]

        for rewards, label in zip(runs, run_labels):
            ax.plot(x, rewards, label=label)
        ax.legend()

    # Plot train.
    plot_runs(training_runs, labels, 1, "Training")

    # Optionally, show validations.
    if val_runs is not None:
        plot_runs(val_runs, labels, 2, "Validation")

    f.show()


if __name__ == "__main__":
    # basic = np.random.random((30, 30))
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # value_grid(ax1, basic, show_values=False, agent_position=(5, 5))
    # value_grid(ax2, basic, show_values=False, agent_position=(5, 5))
    # value_grid(ax3, basic, show_values=False, agent_position=(5, 5))
    # value_grid(ax4, basic, show_values=False, agent_position=(5, 5))
    #
    # fig.show()
    #
    # input()
    # ax1.clear()
    # ax2.clear()
    # ax3.clear()
    # ax4.clear()
    # # fig.canvas.draw()
    # # input()
    # value_grid(ax1, basic, show_values=False, agent_position=(10, 5))
    # value_grid(ax2, basic, show_values=False, agent_position=(10, 5))
    # value_grid(ax3, basic, show_values=False, agent_position=(10, 5))
    # value_grid(ax4, basic, show_values=False, agent_position=(10, 5))
    # fig.canvas.draw()
    #
    # input()
    basic = np.random.random((30, 30))
    plotter = ValuePlotter(3)
    plotter.add_grid("random", basic)
    input()
    basic = np.random.random((30, 30))
    plotter.add_grid("random2", basic)
    input()
    basic = np.random.random((30, 30))
    plotter.add_grid("random3", basic)
    input()
    basic = np.random.random((30, 30))
    plotter.add_grid("random4", basic)
    input()
    basic = np.random.random((30, 30))
    plotter.update_grid("random", basic)
    input()
    basic_q = np.random.random((30, 30, 4))
    plotter.add_q_func("random_qs", basic_q, ["up", "down", "left", "right"])
    input()
    basic_q = np.random.random((30, 30, 4))
    plotter.update_q_func("random_qs", basic_q)
    input()

    # r_grid = np.zeros((30, 30))
    # positions = [(0, i) for i in range(30)] + [(i, 29) for i in range(1, 30)]
    # play_trajectory(r_grid, positions, "./mvie.mp4")
    # input()
