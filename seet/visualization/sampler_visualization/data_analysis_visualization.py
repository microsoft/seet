"""data_analysis_visualization.py

Tools for visualizing results of data analyses
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import matplotlib.pyplot as plt
import numpy


def convert_gazes_to_grid(gazes):
    gaze_dict = {0: set(), 1: set()}
    for gaze in gazes:
        gaze_dict[0].add(gaze[0])
        gaze_dict[1].add(gaze[1])

    return [len(gaze_dict[0]), len(gaze_dict[1])]


def plot_gaze_to_camera_histograms(
    visibility,
    title="Histogram of Gaze-to-Camera Angles",
    nominal_title="Histogram of Gaze-to-Camera Angles at Nominal Gaze",
    horiz_xlabel="Horizontal Gaze-to-Camera Angle [deg]",
    vert_xlabel="Vertical Gaze-to-Camera Angle [deg]",
    ylabel="Percentage of Gazes",
    path_prefix=None,
    **kwargs
):

    h_data = \
        (
            title,
            horiz_xlabel,
            (
                visibility.bin_edges_horizontal_angle_deg[:-1] +
                visibility.bin_edges_horizontal_angle_deg[1:]
            ) / 2,
            visibility.hist_horizontal_angle_deg,
            "horiz_histogram"
        )
    h_nominal_data = \
        (
            nominal_title,
            horiz_xlabel,
            (
                visibility.bin_edges_horizontal_angle_at_nominal_deg[:-1] +
                visibility.bin_edges_horizontal_angle_at_nominal_deg[1:]
            ) / 2,
            visibility.hist_horizontal_angle_at_nominal_deg,
            "horiz_histogram_at_nominal"
        )
    v_data = \
        (
            title,
            vert_xlabel,
            (
                visibility.bin_edges_vertical_angle_deg[:-1] +
                visibility.bin_edges_vertical_angle_deg[1:]
            ) / 2,
            visibility.hist_vertical_angle_deg,
            "vert_histogram"
        )
    v_nominal_data = \
        (
            nominal_title,
            vert_xlabel,
            (
                visibility.bin_edges_vertical_angle_at_nominal_deg[:-1] +
                visibility.bin_edges_vertical_angle_at_nominal_deg[1:]
            ) / 2,
            visibility.hist_vertical_angle_at_nominal_deg,
            "vert_histogram_at_nominal"
        )

    # Make sure limits of both horizontal or both vertical plots are same.
    width = min(h_data[2][1] - h_data[2][0], v_data[2][1] - v_data[2][0])
    xlim_min = min(h_data[2][0], v_data[2][0]) - width / 2
    xlim_max = max(h_data[2][-1], v_data[2][-1]) + width / 2
    xlim_min = min(xlim_min, -xlim_max)
    xlim_max = max(xlim_max, -xlim_min)

    groupped = (h_data, h_nominal_data, v_data, v_nominal_data)
    for data in groupped:
        width = data[2][1] - data[2][0]
        title = data[0]
        xlabel = data[1]
        x = data[2]
        y = data[3]
        name = data[4]

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([xlim_min, xlim_max])
        ax.grid(True)
        ax.bar(x, 100 * y, width=0.95 * width, **kwargs)
        plt.tight_layout()

        if path_prefix is not None:
            plot_name = path_prefix + " " + name + ".png"
            plt.savefig(plot_name, bbox_inches="tight")


def plot_glint_count_per_gaze(
    glint_counter,
    xlabel="Glint Counts",
    ylabel="Percentage of Images",
    path_prefix=None,
    **kwargs
):
    gaze_grid = convert_gazes_to_grid(glint_counter.gazes)

    which_eye = \
        {
            "Left": glint_counter.counts_per_gaze_left,
            "Right": glint_counter.counts_per_gaze_right}
    for key in which_eye.keys():
        counts_per_gaze = which_eye[key]

        fig, axs = \
            plt.subplots(
                nrows=gaze_grid[1],
                ncols=gaze_grid[0],
                figsize=numpy.array(gaze_grid) * 2
            )
        fig.suptitle(
            "Histograms of Glint Counts for {0:s}".format(key) +
            "Eye for Different (H, V) Gaze Directions"
        )

        for N, gaze in enumerate(glint_counter.gazes):
            i = N % gaze_grid[1]
            j = N // gaze_grid[1]
            title = r"${0:.1f}^\circ$, ${1:.1f}^\circ$".format(*gaze)
            axs[i, j].set_title(title)  # type: ignore
            axs[i, j].set_xlabel(xlabel)  # type: ignore
            axs[i, j].set_ylabel(ylabel)  # type: ignore
            axs[i, j].set_xlim(  # type: ignore
                [glint_counter.bin_edges[0], glint_counter.bin_edges[-1]]
            )
            axs[i, j].set_ylim([0, 100])  # type: ignore
            axs[i, j].grid(True)  # type: ignore
            axs[i, j].bar(  # type: ignore
                glint_counter.bins,
                100 * counts_per_gaze[N] / counts_per_gaze[N].sum(),
                label=title,
                **kwargs
            )
        plt.tight_layout()

        if path_prefix is not None:
            plot_name = path_prefix + " glint_counts_per_gaze_" + key + ".png"
            plt.savefig(plot_name, bbox_inches="tight")


def plot_overall_glint_count(
    glint_counter,
    title="Overall Glint Count Distribution",
    xlabel="Glint Counts",
    ylabel="Percentage of Images",
    plot_name=None,
    **kwargs
):

    gaze_grid = convert_gazes_to_grid(glint_counter.gazes)
    fig, ax = plt.subplots(figsize=numpy.array(gaze_grid) * 2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([glint_counter.bin_edges[0], glint_counter.bin_edges[-1]])
    ax.set_ylim([0, 100])
    ax.grid(True)
    ax.bar(
        glint_counter.bins,
        100 * glint_counter.all_counts / glint_counter.all_counts.sum(),
        label=title,
        **kwargs
    )
    plt.tight_layout()

    if plot_name is not None:
        plt.savefig(plot_name, bbox_inches="tight")


def plot_population_and_fov_coverage_per_min_glint_count(
    glint_counter,
    gaze_quantiles=numpy.array([0.5, 0.8, 0.9, 0.95, 1.0]),
    path_prefix=None,
    **kwargs
):
    gaze_grid = convert_gazes_to_grid(glint_counter.gazes)
    num_quantiles = len(gaze_quantiles)
    num_bins = len(glint_counter.bins)

    which_eye = \
        {
            "Both": glint_counter.num_gazes_with_min_count,
            "Right": glint_counter.num_gazes_with_min_count_right,
            "Left": glint_counter.num_gazes_with_min_count_left
        }
    all_tables = []
    for key in which_eye.keys():
        data = numpy.zeros((num_quantiles, num_bins))

        num_gazes = len(glint_counter.gazes)
        if key == "Both":
            num_gazes *= 2  # Times 2 because we have data for both eyes.

        for j in range(len(glint_counter.bins)):
            # Number of gazes for all users with at least min_count bins
            num_gazes_j = which_eye[key][:, j]
            for i, q in enumerate(gaze_quantiles):
                # Number of users with at least (q * num_gazes) gazes meeting
                # minimum glint criterion.
                data[i, j] = len(num_gazes_j[num_gazes_j >= q * num_gazes])

        _, ax = plt.subplots(figsize=numpy.array(gaze_grid) * 2)
        title = \
            "Percentage of Users Meeting Glint Requirements for Different " + \
            "FoV Quantiles, {0:s} Eye(s)".format(key)
        xlabel = "Minimum Glint Count"
        ylabel = "Percentage of Users"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([glint_counter.bin_edges[0], glint_counter.bin_edges[-1]])
        ax.set_ylim([0, 100])
        ax.grid(True)

        # Table with maximum value for minimum glint count jointly covering
        # (100%, 95%) of the population x (100%, 95%, 90%, 80%, 50%) FoV.
        table = r"|{0:s} eyes|||||".format(key) + "\n"
        table = table + "|-|-|-|-|-|\n"
        table = table + r"|Population coverage"
        for q in reversed(gaze_quantiles):
            table = table + r"|{:.0f}% FoV".format(q * 100)
        table = table + "|\n"

        one_hundred_coverage = numpy.zeros(len(gaze_quantiles), dtype=int)
        ninety_eight_coverage = numpy.zeros(len(gaze_quantiles), dtype=int)
        ninety_five_coverage = numpy.zeros(len(gaze_quantiles), dtype=int)
        bottom = numpy.zeros(num_bins)
        for i, q in enumerate(reversed(gaze_quantiles)):
            j = num_quantiles - i - 1
            label = r"$\geq$ {0:.0f}% FoV".format(q * 100)
            height = data[j, :] / glint_counter.num_scenes * 100 - bottom
            ax.bar(
                glint_counter.bins,
                height,
                bottom=bottom,
                label=label,
                **kwargs
            )
            bottom = height + bottom
            one_hundred_coverage_indices = numpy.nonzero(bottom >= 100)[0]
            one_hundred_coverage[i] = one_hundred_coverage_indices[-1]
            ninety_eight_coverage_indices = numpy.nonzero(bottom >= 98)[0]
            ninety_eight_coverage[i] = ninety_eight_coverage_indices[-1]
            ninety_five_coverage_indices = numpy.nonzero(bottom >= 95)[0]
            ninety_five_coverage[i] = ninety_five_coverage_indices[-1]

        ax.legend()

        plt.tight_layout()

        if path_prefix is not None:
            plot_name = path_prefix + " glint_KPI_" + key + ".png"
            plt.savefig(plot_name, bbox_inches="tight")

        table = table + r"|100%"
        for g in one_hundred_coverage:  # g for glints
            table = table + r"|{0:d}".format(g)
        table = table + "|\n"

        table = table + r"|98%"
        for g in ninety_eight_coverage:  # g for glints
            table = table + r"|{0:d}".format(g)
        table = table + "|\n"

        table = table + r"|95%"
        for g in ninety_five_coverage:  # g for glints
            table = table + r"|{0:d}".format(g)
        table = table + "|\n"

        all_tables = all_tables + [table, ]

    return all_tables
