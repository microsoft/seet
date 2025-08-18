"""sampler_utils.py

Utilities for sampling parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import numpy
import scipy.stats
import torch


class IrisVisibilityStatistics:
    """
    Computation of iris-visibility metrics.
    """

    def __init__(self, df):
        self.df = df
        self.generate_visibility_statistics()

    def generate_visibility_statistics(self):
        """generate_visibility_statistics.

        Generate histograms from iris-visibility data.
        """

        # Relevant headers.
        header = [
            "Subsystem",
            "Percentage visible",
            "Area in iris [mm^2]",
            "Area in image [pix^2]"
        ]

        # Summary statistics.
        table = r"|Parameter (N = {0:d})|Mean|Std|Min|Max|5%-ile|95%-ile"\
            .format(len(self.df)) + "\n"
        for col in header[1:]:
            table = table + \
                r"|{0:s}|{1:.1f}|{2:.1f}|{3:.1f}|{4:.1f}|{5:.1f}|{6:.1f}"\
                .format(
                    col,
                    self.df[col].mean(),
                    self.df[col].std(),
                    self.df[col].min(),
                    self.df[col].max(),
                    self.df[col].quantile(0.05),
                    self.df[col].quantile(0.95)
                ) + "\n"

        self.table = table


class EyeVisibilityStatistics:
    """
    Computation of eye-visibility metrics.
    """

    def __init__(self, df):
        self.df = df
        self.generate_visibility_statistics()

    def generate_visibility_statistics(self):
        """generate_visibility_statistics.

        Generate histograms from eye-visibility data.
        """

        # Relevant headers.
        header_camera_center_in_pupil = \
            [
                "Camera center in pupil {:s}".format(ax)
                for ax in ["x", "y", "z"]
            ]
        header_gaze_angles = ["Horiz. angle", "Vert. angle"]

        # Relevant data
        pupil_data = self.df.loc[:, header_camera_center_in_pupil].to_numpy()
        angle_data = self.df.loc[:, header_gaze_angles].to_numpy()
        ind_min_angle_data = numpy.argmin(numpy.abs(angle_data).sum(axis=1))
        min_abs_angle = numpy.abs(angle_data[ind_min_angle_data, :])

        horizontal_angle_deg = []
        vertical_angle_deg = []
        horizontal_angle_at_nominal_deg = []
        vertical_angle_at_nominal_deg = []
        for i in range(len(self.df)):
            # The horizontal angles for left and right eyes have different
            # meanings.
            if self.df.loc[i, "Subsystem"] == 0:
                sign = -1
            else:
                sign = 1

            camera_center_inPupil = torch.tensor(pupil_data[i, :])

            # Results across all gaze directions.
            h_angle_deg = \
                core.rad_to_deg(
                    torch.atan2(
                        camera_center_inPupil[0], camera_center_inPupil[-1]
                    )
                ).clone().detach().numpy() * sign  # Fix sign!
            horizontal_angle_deg = horizontal_angle_deg + [h_angle_deg, ]

            v_angle_deg = \
                core.rad_to_deg(
                    torch.atan2(
                        camera_center_inPupil[1], camera_center_inPupil[-1]
                    )
                ).clone().detach().numpy()
            vertical_angle_deg = vertical_angle_deg + [v_angle_deg, ]

            # Results for nominal gaze.
            if numpy.allclose(numpy.abs(angle_data[i, :]), min_abs_angle):
                horizontal_angle_at_nominal_deg = \
                    horizontal_angle_at_nominal_deg + [h_angle_deg, ]
                vertical_angle_at_nominal_deg = \
                    vertical_angle_at_nominal_deg + [v_angle_deg, ]

        horizontal_angle_deg = numpy.array(horizontal_angle_deg)
        horizontal_angle_at_nominal_deg = \
            numpy.array(horizontal_angle_at_nominal_deg)
        vertical_angle_deg = numpy.array(vertical_angle_deg)
        vertical_angle_at_nominal_deg = \
            numpy.array(vertical_angle_at_nominal_deg)

        # Summary statistics.
        table = \
            r"|($N = {0:d}$) | Mean | Std | Min | Max | 5%-ile | 95%-ile"\
            .format(len(self.df)) + "\n"
        table = table + "|-" * 7 + "\n"

        for i, angles in enumerate(
            [horizontal_angle_deg, vertical_angle_deg]
        ):
            mean = angles.mean()
            std = angles.std()
            min = angles.min()
            max = angles.max()
            q_angles = numpy.quantile(angles, [0.05, 0.95])

            table = table + \
                r"|{0:s}|{1:.1f}|{2:.1f}|{3:.1f}|{4:.1f}|{5:.1f}|{6:.1f}"\
                .format(header_gaze_angles[i], mean, std, min, max, *q_angles)\
                + "\n"

        self.table = table
        self.hist_horizontal_angle_deg, self.bin_edges_horizontal_angle_deg = \
            numpy.histogram(horizontal_angle_deg, density=True)
        self.hist_vertical_angle_deg, self.bin_edges_vertical_angle_deg = \
            numpy.histogram(vertical_angle_deg, density=True)

        self.hist_horizontal_angle_at_nominal_deg, \
            self.bin_edges_horizontal_angle_at_nominal_deg = \
            numpy.histogram(horizontal_angle_at_nominal_deg, density=True)
        self.hist_vertical_angle_at_nominal_deg, \
            self.bin_edges_vertical_angle_at_nominal_deg = \
            numpy.histogram(vertical_angle_at_nominal_deg, density=True)


class GlintCountStatistics:
    """
    Structure to hold output of computation of glint count statistics.
    """

    def __init__(self, df):
        self.df = df
        self.generate_glint_count_statistics()

    def generate_glint_count_statistics(self):

        # Get the header for the LED visibility columns.
        header_LEDs = []
        for col in self.df.columns:
            if col.lower()[:4] == "led ":
                header_LEDs = header_LEDs + [col, ]

        # Allocate space to store the histograms
        max_num_glints = len(header_LEDs)
        # For N glints, we have N + 1 bins, because we want to consider the
        # possibility of having 0 visible glints. And N + 1 bints have N + 2
        # edges.
        bin_edges = \
            numpy.linspace(-0.5, max_num_glints + 0.5, num=max_num_glints + 2)
        bins = numpy.array(range(max_num_glints + 1))

        # Results across all gaze directions.
        data = self.df.loc[:, header_LEDs].sum(axis=1).to_numpy()
        all_counts, _ = numpy.histogram(data, bins=bin_edges)

        # Results grouped by gaze directions.
        gazes = []
        counts_per_gaze_left = []
        counts_per_gaze_right = []
        grouped = self.df.groupby(by=["Horiz. angle", "Vert. angle"])
        for group in grouped:
            gazes = gazes + [group[0], ]

            grouped_per_eye = group[1].groupby(by="Subsystem")
            for eye_group in grouped_per_eye:
                data = eye_group[1].loc[:, header_LEDs].sum(axis=1).to_numpy()
                counts = numpy.histogram(data, bins=bin_edges)[0]
                if eye_group[0] == 0:
                    counts_per_gaze_left = counts_per_gaze_left + [counts, ]
                if eye_group[0] == 1:
                    counts_per_gaze_right = counts_per_gaze_right + [counts, ]

        # Results grouped by user.
        grouped = self.df.groupby(by="Scene index")
        num_scenes = len(grouped)
        num_gazes_with_min_count = numpy.zeros((num_scenes, len(bins)))
        num_gazes_with_min_count_left = numpy.zeros((num_scenes, len(bins)))
        num_gazes_with_min_count_right = numpy.zeros((num_scenes, len(bins)))
        for i, group in enumerate(grouped):
            # Counts per user at each gaze direction
            data = group[1].loc[:, header_LEDs].sum(axis=1).to_numpy()

            # Number of gaze directions with at least 0, 1, ..., N visible
            # glints.
            for j, min_counts in enumerate(bins):
                num_gazes_with_min_count[i, j] = len(data[data >= min_counts])

            # Same but for each eye.
            grouped_per_eye = group[1].groupby(by="Subsystem")
            for eye_group in grouped_per_eye:
                data = eye_group[1].loc[:, header_LEDs].sum(axis=1).to_numpy()
                for j, min_counts in enumerate(bins):
                    if eye_group[0] == 0:
                        num_gazes_with_min_count_left[i, j] = \
                            len(data[data >= min_counts])
                    if eye_group[0] == 1:
                        num_gazes_with_min_count_right[i, j] = \
                            len(data[data >= min_counts])

            self.num_scenes = num_scenes
            self.bins = bins
            self.bin_edges = bin_edges
            self.all_counts = all_counts
            self.gazes = gazes
            self.counts_per_gaze_left = counts_per_gaze_left
            self.counts_per_gaze_right = counts_per_gaze_right
            self.num_gazes_with_min_count_left = num_gazes_with_min_count_left
            self.num_gazes_with_min_count_right = \
                num_gazes_with_min_count_right
            self.num_gazes_with_min_count = num_gazes_with_min_count


class Sampler:
    """
    A basic sampling class that generates samples according to a specified
    distribution.
    """

    def __init__(self, dist_type="truncated normal", **kwargs):
        """__init__.

        Create a sampler from which to draw samples.

        Args:
            dist_type (str, optional): distribution family. It may be "normal",
            "uniform", or "truncated normal". Defaults to "truncated normal".

            **kwargs (dict, optional): additional parameters to define the
            distribution.
        """

        self.kwargs = kwargs
        if dist_type == "uniform":
            # For the uniform distribution,
            #
            # kwargs = {"loc": min, "scale": max-min}.
            self.random_variable = scipy.stats.uniform

        elif dist_type == "normal":
            # For the normal distribution,
            #
            # kwargs = {"loc": mean, "scale": std}.
            self.random_variable = scipy.stats.norm

        else:  # dist_type == "truncated normal"
            # For the truncated normal distribution,
            #
            # kwargs = {"min": min, "max": max, "loc": mean, "scale": std}.
            #
            # Note that mean and std are for the non-truncated distribution.
            # The actual mean and std will depend on the truncation parameters.
            # Note also that the parameters "a" and "b" of the truncnorm RV in
            # scipy have a non-intuitive meaning, different from "min" and
            # "max".
            tmp_kwargs = {"loc": 0.0, "scale": 1.0, "min": -2.0, "max": 2.0}
            if kwargs is None:
                kwargs = tmp_kwargs
            else:
                for key in tmp_kwargs.keys():
                    tmp_kwargs[key] = kwargs.get(key, tmp_kwargs[key])

            self.kwargs = tmp_kwargs

            # We need to convert min/max to a b
            a = (self.kwargs["min"] - self.kwargs["loc"]) / \
                self.kwargs["scale"]
            b = (self.kwargs["max"] - self.kwargs["loc"]) / \
                self.kwargs["scale"]
            self.kwargs["a"] = a
            self.kwargs["b"] = b
            self.kwargs.pop("min")
            self.kwargs.pop("max")

            self.random_variable = scipy.stats.truncnorm

    @staticmethod
    def read_SE3_parameters(dictionary, key):
        """read_SE3_parameters.

        This helper function reads parameters of a rotational or
        translational distribution.

        Args:
            dictionary (dict): input dictionary. Must be formatted as
            {
                key: {
                    "apply": True or False, (optional)
                    x: {dictionary of distribution parameters},
                    y: {dictionary of distribution parameters},
                    z: {dictionary of distribution parameters},
                },

                etc.
            }

            key (string): key from which to extract the rotation or translation
            parameters for sampling in SE3.
        """
        group = dictionary[key]
        apply = group.get("apply", False)
        samplers = []
        if apply:
            for axis in ("x", "y", "z"):
                dist_type = group[axis]["distribution type"]
                samplers = \
                    samplers + [Sampler(dist_type=dist_type, **group[axis]), ]

        return samplers

    def generate_sample(self):
        """generate_sample.

        Generate a sample from the random variable. The presumed use of this
        sample is to be added to some canonical value to generate an additive
        perturbation.
        """

        return self.random_variable.rvs(**self.kwargs).astype(core.NUMPY_DTYPE)
