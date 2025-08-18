"""App for eye-tracking sensitivity analysis.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import matplotlib.pyplot as plt
import PySimpleGUI as sg
import sensitivity_analysis_app_utils
import sys


class SensitivityAnalysisAPP():
    """Class with GUI elements for eye-tracking sensitivity analysis.
    """

    def __init__(self):
        """Create the app and initialize the event loop.
        """
        self.analysis_utils = \
            sensitivity_analysis_app_utils.SensitivityAnalysisAppUtils()

        self.create_window()

        # We stay here until process_events returns
        self.process_events()

        self.close()

    def close(self):
        """Graciously exit.
        """
        self.window.close()
        sys.exit(0)

    @staticmethod
    def _labeled_slider(text, max, min=0.0, key=None):
        """Create a PySimpleGUI named frame holding a slider.

        Resolution is (max - min) / 10, orientation is vertical, and size is
        (10, 10).

        Args:
            text (string): name of parameter controlled by slider.

            max (float): maximum value of slider.

            min (float, optional): minimum value of slider. Defaults to 0.0.

            key (str, optional): key used to access value of slider. Defaults
            to None, in which case there is no key.

        Returns:
            sg.Frame: borderless PySimpleGUI frame holding slider.
        """
        slider = \
            sg.Slider(
                range=(min, max),
                default_value=max/2,
                resolution=(max - min) / 20,
                orientation="vertical",
                size=(10, 10),
                key=key
            )

        layout = [[slider]]

        return sg.Frame(text, layout, border_width=0)

    @staticmethod
    def _multi_slider_widget(name, text, max, min=None, keys=None):
        """Creates a named frame of named side-by-side labeled sliders.

        Args:
            name (str): name of frame.

            text (list of str): name of each slider in the frame.

            max (list of float): maximum value of each slider in the frame.

            min (list of float, optional): minimum value of each slider in the
            frame. Defaults to None, in which case the minimum value is zero.

            keys (list of str, optional): string key for each slider in the
            frame. Defaults to None, in which case the sliders will not have
            keys.

        Returns:
            sg.Frame: PySimpleGUI frame holding the sliders.
        """
        num_sliders = len(text)
        assert (num_sliders == len(max))
        if min is None:
            min = len(max) * [0.0]
        if keys is None:
            keys = len(max) * [None]
        else:
            assert (len(keys) == len(max))

        layout = \
            [
                [
                    SensitivityAnalysisAPP._labeled_slider(
                            text[i], max[i], min=min[i], key=keys[i]
                    )
                    for i in range(num_sliders)
                ]
            ]

        return sg.Frame(name, layout)

    def create_tab_widget(self, name, slider_text, max_vals):
        layout = self._multi_slider_widget(name, slider_text, max_vals)

        return sg.Tab("LEDs", [[layout]])

    def create_window(self):
        """Create GUI.
        """

        #######################################################################
        # Frame with controls for generating derivatives.
        generate_derivatives_frame = \
            sg.Frame(
                "Generate data",
                [
                    [
                        # Input number of samples.
                        sg.Text("Number of samples:", size=(20, 1)),
                        sg.InputText(
                            sg.user_settings_get_entry("-num samples-", ""),
                            size=(4, 1),
                            key="-NUMSAMPLES-"
                        )
                    ],
                    [
                        # Input sampling configuration file name.
                        sg.Text("Sampling-parameters file:", size=(20, 1)),
                        sg.Input(
                            sg.user_settings_get_entry(
                                "-sampling file name-", ""),
                            size=(36, 1),
                            key="-SAMPLINGFILENAME-"
                        ),
                        sg.FileBrowse(target="-SAMPLINGFILENAME-"),

                        # Trigger generation of derivatives.
                        sg.Button(
                            "OK",
                            size=(7, 1),
                            enable_events=True,
                            key="-GENERATEDERIVATIVES-"
                        )
                    ]
                ]
            )
        #######################################################################

        #######################################################################
        # Frame with controls for loading derivatives.
        load_derivatives_frame = \
            sg.Frame(
                "Load data",
                [
                    [
                        # Input derivatives file name.
                        sg.Text("Derivatives file:", size=(20, 1)),
                        sg.Input(
                            sg.user_settings_get_entry(
                                "-derivatives file name-", ""
                            ),
                            size=(36, 1),
                            key="-DERIVATIVESFILENAME-"
                        ),
                        sg.FileBrowse(target="-DERIVATIVESFILENAME-"),

                        # Trigger data loading.
                        sg.Button(
                            "OK",
                            size=(7, 1),
                            enable_events=True,
                            key="-LOADDERIVATIVES-"
                        )
                    ]
                ]
            )
        #######################################################################

        #######################################################################
        # Frame with controls for loading input covariances.
        input_covariances_frame = \
            sg.Frame(
                "Load input covariances",
                [
                    [
                        # Input folder with covariance files.
                        sg.Text("Path to covariance files:", size=(20, 1)),
                        sg.Input(
                            sg.user_settings_get_entry(
                                "-covariances folder name-", ""),
                            size=(36, 1),
                            key="-COVARIANCESFOLDERNAME-"
                        ),
                        sg.FolderBrowse(target="-COVARIANCESFOLDERNAME-"),

                        # Trigger loading of covariances.
                        sg.Button(
                            "OK",
                            size=(7, 1),
                            enable_events=True,
                            key="-LOADCOVARIANCES-"
                        )
                    ]
                ]
            )
        #######################################################################

        #######################################################################
        # Frame with fine tunning of standard deviation of input parameters.

        # Standard deviation of LED position
        led_tab = \
            sg.Tab(
                "LEDs",
                [
                    [
                        self._multi_slider_widget(
                            "Standard dev. for coordinates [mm]:",
                            ["X", "Y", "Z"],
                            [1.0] * 3,
                            keys=["LED X", "LED Y", "LED Z"]
                        )
                    ]
                ]
            )

        # Standard deviation of camera extrinsic parameters
        extrinsics_tab = \
            sg.Tab(
                "Camera extrinsics",
                [
                    [
                        # Standard deviation or rotation.
                        self._multi_slider_widget(
                            "Standard dev. for rotation [mrad]:",
                            ["Pitch", "Yaw", "Roll"],
                            [5.0] * 3,
                            keys=["Pitch", "Yaw", "Roll"],
                        ),
                        sg.Push(),

                        # Standard deviation of translation
                        self._multi_slider_widget(
                            "Standard dev. for translation [mm]:",
                            ["X", "Y", "Z"],
                            [1.0] * 3,
                            keys=["Camera X", "Camera Y", "Camera Z"]
                        )
                    ]
                ]
            )

        # Standard deviation of camera intrinsic parameters.
        intrinsics_tab = \
            sg.Tab(
                "Camera intrinsics",
                [
                    [
                        # Standard deviation of pinhole parameters.
                        self._multi_slider_widget(
                            "Standard dev. for pinhole parameters [pix]:",
                            ["px", "py", "fx", "fy"],
                            max=[0.5] * 4,
                            keys=["px", "py", "fx", "fy"]
                        ),
                        sg.Push(),

                        # Standard deviation of radial distortion parameters.
                        self._multi_slider_widget(
                            "Standard dev. for distortion parameters [adim.]:",
                            ["cx", "cy", "k0", "k1", "k2"],
                            [0.2] * 5,
                            keys=["cx", "cy", "k0", "k1", "k2"]
                        )
                    ]
                ]
            )

        # Standard deviation of location of image features.
        features_tab = \
            sg.Tab(
                "Image features",
                [
                    [
                        self._multi_slider_widget(
                            "Standard dev. for features [pix]:",
                            ["Glint", "Pupil", "Limbus"],
                            [1.0] * 3,
                            keys=["Glint", "Pupil", "Limbus"]
                        )
                    ]
                ]
            )

        # Standard deviation of all inputs.
        input_standard_deviation_frame = \
            sg.Frame(
                "Noise fine tunning",
                [
                    [
                        sg.TabGroup(
                            [
                                [
                                    led_tab,
                                    extrinsics_tab,
                                    intrinsics_tab,
                                    features_tab
                                ]
                            ]
                        )
                    ]
                ]
            )

        #######################################################################
        # Frame with controls for plots.
        plot_parameters_frame = \
            sg.Frame(
                "Standard deviations at KPIs",
                [
                    [
                        sg.Text("Gaze [deg]:"),
                        sg.Input("0.75", key="gaze KPI", size=(5, 1)),
                        sg.Push(),
                        sg.Text("X, Y, Z position [mm]:"),
                        sg.Input("0.75", key="x position KPI", size=(5, 1)),
                        sg.Input("0.75", key="y position KPI", size=(5, 1)),
                        sg.Input("", key="z position KPI", size=(5, 1))
                    ]
                ]
            )

        plot_generation_frame = \
            sg.Frame(
                "Plot controls",
                [
                    [
                        sg.Check("All plots", key="all plots"),
                        sg.Button("Plot", size=(7, 1), key="-PLOT-")
                    ]
                ]
            )

        layout = [
            [generate_derivatives_frame],
            [sg.Text("OR")],
            [load_derivatives_frame],
            [sg.HorizontalSeparator()],
            [input_covariances_frame],
            [input_standard_deviation_frame],
            [plot_parameters_frame, sg.Push(), plot_generation_frame]
        ]

        self.derivatives_file_name = None
        self.covariances_folder_name = None

        # Fix misbehaving scale. PySimpleGUI is using TkInter scale, pyplot
        # uses Qt.
        tmp = sg.tk.Tk()
        fig, _ = plt.subplots(1, 1)
        tmp.tk.call("tk", "scaling", fig.dpi/36)
        plt.close(fig)
        tmp.destroy()

        self.window = \
            sg.Window(
                "Sensitivity Analysis Tool",
                layout
            )

    def process_events(self):
        """Event processing loop.
        """

        while True:
            self.event, self.values = self.window.read()

            if self.event == sg.WIN_CLOSED:
                self.analysis_utils.close()
                return

            # Read in available configuration files and folders.
            self.derivatives_file_name = self.values["-DERIVATIVESFILENAME-"]
            self.covariances_folder_name = \
                self.values["-COVARIANCESFOLDERNAME-"]

            if self.event == "-GENERATEDERIVATIVES-":
                # Get the number of samples.
                num_samples_str = self.values["-NUMSAMPLES-"]
                sg.user_settings_set_entry("-num samples-", num_samples_str)
                num_samples = \
                    int(num_samples_str) if num_samples_str != "" else None

                # Get the configuration file for the sampler.
                sampling_file_name = self.values["-SAMPLINGFILENAME-"]
                sg.user_settings_set_entry(
                    "-sampling file name-", sampling_file_name
                )

                # Generate the derivatives. This is very expensive!
                generator = self.analysis_utils.generate_data(
                    sampling_file_name,
                    num_samples=num_samples
                )
                while True:
                    try:
                        counter = next(generator)
                        sg.one_line_progress_meter(
                            "Computing derivatives of data",
                            counter + 1,
                            num_samples * 2 * 20  # 2 eyes, 5x4 grid.
                        )

                    except StopIteration:
                        # At the end of the iterations, the object
                        # self.analysis_utils has its member data
                        # "data_dictionary" populated with the values of the
                        # derivatives.
                        break

                # Save the results.
                output_dir_name = \
                    sg.popup_get_folder(
                        "Save derivatives for future use.",
                        default_path=sg.user_settings_get_entry(
                            "-last saved derivative path-", ""
                        ),
                        history=True
                    )

                if output_dir_name is not None:
                    sg.user_settings_set_entry(
                        "-saved derivative path-",
                        list(
                            set(
                                sg.user_settings_get_entry(
                                    "-saved derivative path-", []
                                ) +
                                [output_dir_name, ]
                            )
                        )
                    )

                    sg.user_settings_set_entry(
                        "-last saved derivative path-", output_dir_name
                    )

                    self.derivatives_file_name = \
                        self.analysis_utils.save_data(output_dir_name)
                    self.values["-DERIVATIVESFILENAME-"] = \
                        self.derivatives_file_name
                    sg.user_settings_set_entry(
                        "-derivatives file name-",
                        self.values["-DERIVATIVESFILENAME-"]
                    )

            if self.event == "-LOADDERIVATIVES-":
                # Load pre-computed derivatives used in sensitivity analysis.
                derivatives_file_name = self.values["-DERIVATIVESFILENAME-"]
                self.derivatives_file_name = derivatives_file_name
                sg.user_settings_set_entry(
                    "-derivatives file name-",
                    self.values["-DERIVATIVESFILENAME-"]
                )
                self.analysis_utils.load_derivatives(
                    self.derivatives_file_name
                )

            if self.event == "-LOADCOVARIANCES-":
                # Load configuration files for sensitivity analysis.
                covariances_folder_name = \
                    self.values["-COVARIANCESFOLDERNAME-"]
                self.covariances_folder_name = covariances_folder_name
                sg.user_settings_set_entry(
                    "-covariances folder name-",
                    self.values["-COVARIANCESFOLDERNAME-"]
                )
                self.analysis_utils.load_configuration(
                    self.covariances_folder_name
                )

            if self.event == "-PLOT-":
                # Create pyplots with results of sensitivity analysis.
                if self.covariances_folder_name is None:
                    sg.Popup("Noise configuration is missing.")

                else:
                    self.analysis_utils.load_configuration(
                        self.covariances_folder_name
                    )

                    if self.derivatives_file_name is None:
                        sg.Popup("Derivatives data are missing.")
                    else:
                        self.analysis_utils.load_derivatives(
                            self.derivatives_file_name
                        )

                        gaze_kpi_deg = self.window["gaze KPI"].get()
                        if gaze_kpi_deg != "":
                            gaze_kpi_deg = float(gaze_kpi_deg)
                        else:
                            gaze_kpi_deg = None
                        position_kpi_mm = \
                            [
                                self.window["x position KPI"].get(),
                                self.window["y position KPI"].get(),
                                self.window["z position KPI"].get()
                            ]
                        for i in range(len(position_kpi_mm)):
                            if position_kpi_mm[i] != "":
                                position_kpi_mm[i] = float(position_kpi_mm[i])
                            else:
                                position_kpi_mm[i] = None

                        self.generate_plots(
                            [gaze_kpi_deg],
                            position_kpi_mm,
                            self.window["all plots"].get()
                        )

    def generate_plots(self, gaze_kpi_deg, pose_kpi_mm, all_plots):
        """Generate pyplots for visualization of sensitivity-analysis results.
        """

        std_names = [
            "LED X",
            "LED Y",
            "LED Z",
            "Pitch",
            "Yaw",
            "Roll",
            "Camera X",
            "Camera Y",
            "Camera Z",
            "px",
            "py",
            "fx",
            "fy",
            "cx",
            "cy",
            "k0",
            "k1",
            "k2",
            "Glint",
            "Pupil",
            "Limbus"
        ]
        std_dict = dict()
        for name in std_names:
            std_dict[name] = self.values[name]

        self.analysis_utils.load_stds(std_names, std_dict)
        self.analysis_utils.compute_covariances()
        self.analysis_utils.compute_covariances_for_KPIs()
        self.analysis_utils.compute_contributions_to_KPIs()
        self.analysis_utils.plot_results(
            gaze_kpi_deg=gaze_kpi_deg,
            pose_kpi_mm=pose_kpi_mm,
            all_plots=all_plots
        )


if __name__ == "__main__":
    sensitivity_analysis_app = SensitivityAnalysisAPP()
