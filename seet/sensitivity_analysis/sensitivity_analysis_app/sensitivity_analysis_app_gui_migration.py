<<<<<<< HEAD
"""App for eye-tracking sensitivity analysis - PySide6 Version.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import matplotlib.pyplot as plt
import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QTabWidget, QLabel, QLineEdit, QPushButton, QSlider,
    QCheckBox, QFrame, QSpacerItem, QSizePolicy, QFileDialog,
    QMessageBox, QProgressDialog, QComboBox
)
from PySide6.QtCore import Qt, QSettings, Signal, QObject
from PySide6.QtGui import QCloseEvent
import sensitivity_analysis_app_utils
import torch


class SensitivityAnalysisAPP(QMainWindow):
    """Class with GUI elements for eye-tracking sensitivity analysis - PySide6 Version.
    """

    def __init__(self):
        """Create the app and initialize the event loop.
        """
        super().__init__()
        
        # Initialize QSettings for user preferences
        self.settings = QSettings("SEET", "SensitivityAnalysisApp")
        
        self.analysis_utils = \
            sensitivity_analysis_app_utils.SensitivityAnalysisAppUtils()

        initial_device = self.settings.value("compute_device", "cpu")
        self._update_device(initial_device)

        self.create_window()
        
        current_index = self.device_combo.findText(initial_device)
        if current_index >= 0:
            self.device_combo.setCurrentIndex(current_index)
        
        # Show the window
        self.show()

    def _update_device(self, device_name):
        """Update the compute device for PyTorch operations."""
        if device_name == "cuda":
            if torch.cuda.is_available():
                selected_device = torch.device('cuda')
            else:
                # Fall back to CPU if CUDA not available
                selected_device = torch.device('cpu')
                if hasattr(self, 'device_combo'):  # Only show warning if GUI exists
                    QMessageBox.warning(
                        self, "CUDA Not Available",
                        "CUDA is not available. Falling back to CPU."
                    )
                    current_index = self.device_combo.findText('cpu')
                    self.device_combo.setCurrentIndex(current_index)
        else:
            selected_device = torch.device('cpu')
        
        self.analysis_utils.device = selected_device
        self.settings.setValue("compute_device", device_name)
        
        return selected_device

    def _on_device_changed(self, device_name):
        """Handle device combo box changes."""
        self.device_combo.hidePopup()
        self._update_device(device_name)

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event (replaces sg.WIN_CLOSED).
        """
        self.analysis_utils.close()
        event.accept()

    def close_application(self):
        """Graciously exit.
        """
        self.close()

    def _labeled_slider(self, text, max_val, min_val=0.0, key=None):
        """Create a PySide6 labeled slider in a group box.

        Args:
            text (string): name of parameter controlled by slider.
            max_val (float): maximum value of slider.
            min_val (float, optional): minimum value of slider. Defaults to 0.0.
            key (str, optional): object name used to access slider. Defaults to None.

        Returns:
            QGroupBox: borderless PySide6 group box holding slider.
        """
        group_box = QGroupBox()
        group_box.setStyleSheet("border: 0px;")

        slider = QSlider(Qt.Vertical)
        slider.setRange(int(min_val * 20), int(max_val * 20))  # Scale for resolution
        slider.setValue(int(max_val * 10))  # Default to middle
        slider.setFixedSize(40, 100)

        if key:
            slider.setObjectName(key)

        # Parameter label above everything
        param_label = QLabel(text)
        param_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        param_label.setStyleSheet("font-size: 14px;")

        # Max and min labels
        max_label = QLabel(f"{max_val:.2f}")
        max_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        min_label = QLabel(f"{min_val:.2f}")
        min_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Value label beside slider, updates dynamically
        value_label = QLabel(f"{slider.value() / 20.0:.2f}")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("font-weight: bold;")
        slider.valueChanged.connect(lambda val: value_label.setText(f"{val / 20.0:.2f}"))

        # Layout: param label, max label, slider+value label (overlap), min label
        layout = QVBoxLayout()
        layout.addWidget(param_label)
        layout.addWidget(max_label)

        # Stack slider and value label using QStackedLayout or QHBoxLayout
        slider_value_layout = QHBoxLayout()
        slider_value_layout.addStretch()
        slider_value_layout.addWidget(slider)
        slider_value_layout.addWidget(value_label)
        slider_value_layout.addStretch()
        layout.addLayout(slider_value_layout)

        layout.addWidget(min_label)
        group_box.setLayout(layout)

        return group_box

    def _multi_slider_widget(self, name, text, max_vals, min_vals=None, keys=None):
        """Creates a named group box of side-by-side labeled sliders.

        Args:
            name (str): name of group box.
            text (list of str): name of each slider in the group box.
            max_vals (list of float): maximum value of each slider.
            min_vals (list of float, optional): minimum value of each slider.
                Defaults to None, in which case the minimum value is zero.
            keys (list of str, optional): object name for each slider.
                Defaults to None, in which case the sliders will not have object names.

        Returns:
            QGroupBox: PySide6 group box holding the sliders.
        """
        num_sliders = len(text)
        assert (num_sliders == len(max_vals))
        if min_vals is None:
            min_vals = len(max_vals) * [0.0]
        if keys is None:
            keys = len(max_vals) * [None]
        else:
            assert (len(keys) == len(max_vals))

        # Create horizontal layout for sliders
        layout = QHBoxLayout()
        
        for i in range(num_sliders):
            slider_widget = self._labeled_slider(
                text[i], max_vals[i], min_val=min_vals[i], key=keys[i]
            )
            layout.addWidget(slider_widget)

        group_box = QGroupBox(name)
        group_box.setStyleSheet("QGroupBox { font-size: 14px; }")
        group_box.setLayout(layout)
        return group_box

    # Signal handlers for file browse functionality
    def _browse_sampling_file(self):
        """Browse for sampling parameters file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sampling Parameters File",
            self.sampling_filename_input.text(),
            "All Files (*)"
        )
        if file_path:
            self.sampling_filename_input.setText(file_path)

    def _browse_derivatives_file(self):
        """Browse for derivatives file."""
        # Default to examples directory first, then user's last path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        default_examples_dir = os.path.join(project_root, "seet", "sensitivity_analysis", "default_derivatives")
        
        start_dir = self.derivatives_filename_input.text()
        if not start_dir or not os.path.exists(start_dir):
            start_dir = self.settings.value("-derivatives file name-", default_examples_dir)
        if not start_dir or not os.path.exists(start_dir):
            start_dir = default_examples_dir
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Derivatives File (Examples available in default_derivatives folder)",
            start_dir,
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if file_path:
            self.derivatives_filename_input.setText(file_path)

    def _browse_covariances_folder(self):
        """Browse for covariances folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Covariances Folder",
            self.covariances_folder_input.text()
        )
        if folder_path:
            self.covariances_folder_input.setText(folder_path)

    # Action button handlers (placeholders for now)
    def _generate_derivatives(self):
        """Handle generate derivatives button click."""
        try:
            # Get inputs from GUI
            num_samples_text = self.num_samples_input.text().strip()
            sampling_file = self.sampling_filename_input.text().strip()
            
            # Validate inputs
            if not sampling_file:
                QMessageBox.warning(
                    self, "Input Error",
                    "Please specify a sampling parameters file."
                )
                return
                
            if not os.path.exists(sampling_file):
                QMessageBox.warning(
                    self, "File Error",
                    f"Sampling parameters file not found: {sampling_file}"
                )
                return
                
            # Parse number of samples
            num_samples = None
            if num_samples_text:
                try:
                    num_samples = int(num_samples_text)
                    if num_samples <= 0:
                        raise ValueError("Number of samples must be positive")
                except ValueError:
                    QMessageBox.warning(
                        self, "Input Error",
                        "Please enter a valid positive number for samples."
                    )
                    return
            
            # Save settings
            self.settings.setValue("-num samples-", num_samples_text)
            self.settings.setValue("-sampling file name-", sampling_file)
            
            # Create progress dialog
            progress = QProgressDialog(
                "Generating derivatives...", "Cancel", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            # Generate data using the generator
            generator = self.analysis_utils.generate_data(
                sampling_file, num_samples
            )
            sample_count = 0
            
            try:
                while True:
                    try:
                        sample_count = next(generator)
                        # Update progress (rough estimate since we don't know
                        # total in advance)
                        if num_samples:
                            progress_percent = min(
                                int((sample_count / num_samples) * 100), 99
                            )
                        else:
                            # Just show activity
                            progress_percent = min(sample_count, 99)
                        progress.setValue(progress_percent)
                        
                        # Check if user cancelled
                        if progress.wasCanceled():
                            QMessageBox.information(
                                self, "Cancelled",
                                "Data generation was cancelled."
                            )
                            return
                            
                        # Process events to keep GUI responsive
                        QApplication.processEvents()
                        
                    except StopIteration:
                        # Generation completed successfully
                        progress.setValue(100)
                        break
                        
            except Exception as e:
                progress.close()
                QMessageBox.critical(
                    self, "Generation Error",
                    f"Error during data generation: {str(e)}"
                )
                return
            
            progress.close()
            
            # Prompt to save derivatives
            reply = QMessageBox.question(
                self, "Save Derivatives",
                f"Data generation completed! Processed {sample_count} samples.\n\n"
                "Would you like to save the derivatives for future use?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Get save directory - default to project results folder
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                default_results_dir = os.path.join(project_root, "results", "derivatives")
                
                last_saved_path = self.settings.value("-last saved derivative path-", default_results_dir)
                output_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Save derivatives for future use",
                    last_saved_path
                )
                
                if output_dir:
                    try:
                        # Save the derivatives
                        derivatives_file = self.analysis_utils.save_data(output_dir)
                        
                        # Update the derivatives filename input field
                        self.derivatives_filename_input.setText(derivatives_file)
                        
                        # Update settings
                        self.settings.setValue("-last saved derivative path-", output_dir)
                        self.settings.setValue("-derivatives file name-", derivatives_file)
                        saved_paths = self.settings.value("-saved derivative path-", [])
                        if not isinstance(saved_paths, list):
                            saved_paths = []
                        if output_dir not in saved_paths:
                            saved_paths.append(output_dir)
                            self.settings.setValue("-saved derivative path-", saved_paths)
                        
                        QMessageBox.information(
                            self, "Save Successful",
                            f"Derivatives saved successfully to:\n{derivatives_file}"
                        )
                        
                    except Exception as e:
                        QMessageBox.critical(
                            self, "Save Error",
                            f"Error saving derivatives: {str(e)}"
                        )
            else:
                QMessageBox.information(
                    self, "Complete",
                    f"Data generation completed! Processed {sample_count} samples."
                )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An unexpected error occurred: {str(e)}"
            )
            return

    def _load_derivatives(self):
        """Handle load derivatives button click."""
        try:
            # Get derivatives file path from GUI
            derivatives_file = self.derivatives_filename_input.text().strip()
            
            # Validate input
            if not derivatives_file:
                QMessageBox.warning(
                    self, "Input Error",
                    "Please specify a derivatives file."
                )
                return
                
            if not os.path.exists(derivatives_file):
                QMessageBox.warning(
                    self, "File Error",
                    f"Derivatives file not found: {derivatives_file}"
                )
                return
            
            # Save setting
            self.settings.setValue("-derivatives file name-", derivatives_file)
            
            # Load derivatives
            self.analysis_utils.load_derivatives(derivatives_file)
            self.derivatives_file_name = derivatives_file
            
            QMessageBox.information(
                self, "Success",
                "Derivatives loaded successfully!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error loading derivatives: {str(e)}"
            )
            return

    def _load_covariances(self):
        """Handle load covariances button click."""
        try:
            # Get covariances folder path from GUI
            covariances_folder = self.covariances_folder_input.text().strip()
            
            # Validate input
            if not covariances_folder:
                QMessageBox.warning(
                    self, "Input Error",
                    "Please specify a covariances folder."
                )
                return
                
            if not os.path.exists(covariances_folder):
                QMessageBox.warning(
                    self, "Folder Error",
                    f"Covariances folder not found: {covariances_folder}"
                )
                return
                
            if not os.path.isdir(covariances_folder):
                QMessageBox.warning(
                    self, "Folder Error",
                    f"Path is not a directory: {covariances_folder}"
                )
                return
            
            # Save setting
            self.settings.setValue(
                "-covariances folder name-", covariances_folder
            )
            
            # Load configuration
            self.analysis_utils.load_configuration(covariances_folder)
            self.covariances_folder_name = covariances_folder
            
            QMessageBox.information(
                self, "Success",
                "Covariances loaded successfully!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error loading covariances: {str(e)}"
            )
            return

    def _collect_slider_values(self):
        """Collect standard deviation values from all sliders."""
        std_dict = {}
        std_names = []
        
        # Define slider keys in the order expected by the utility class
        slider_keys = [
            # LEDs (3 components)
            "LED X", "LED Y", "LED Z",
            # Camera extrinsics (6 components)
            "Pitch", "Yaw", "Roll", "Camera X", "Camera Y", "Camera Z",
            # Camera intrinsics (9 components)
            "px", "py", "fx", "fy", "cx", "cy", "k0", "k1", "k2",
            # Image features (3 components)
            "Glint", "Pupil", "Limbus"
        ]
        
        for key in slider_keys:
            slider = self.findChild(QSlider, key)
            if slider:
                # Convert slider value back to float (remember we scaled by 20)
                value = slider.value() / 20.0
                std_dict[key] = value
                std_names.append(key)
            else:
                # Default value if slider not found
                std_dict[key] = 0.5
                std_names.append(key)
                
        return std_names, std_dict

    def _plot_results(self):
        """Handle plot button click."""
        try:
            # Check if derivatives are loaded
            if self.analysis_utils.data_dictionary is None:
                QMessageBox.warning(
                    self, "Data Error",
                    "Please load or generate derivatives first."
                )
                return
                
            # Check if covariances are loaded
            if (self.analysis_utils.leds_covariance_calculator is None or
                    self.analysis_utils.camera_covariance_calculator is None or
                    self.analysis_utils.features_covariance_calculator is
                    None):
                QMessageBox.warning(
                    self, "Configuration Error",
                    "Please load input covariances first."
                )
                return
            
            # Get KPI values from GUI
            try:
                gaze_kpi_text = self.gaze_kpi_input.text().strip()
                gaze_kpi = float(gaze_kpi_text) if gaze_kpi_text else None
                
                x_kpi_text = self.x_position_kpi_input.text().strip()
                y_kpi_text = self.y_position_kpi_input.text().strip()
                z_kpi_text = self.z_position_kpi_input.text().strip()
                
                x_kpi = float(x_kpi_text) if x_kpi_text else None
                y_kpi = float(y_kpi_text) if y_kpi_text else None
                z_kpi = float(z_kpi_text) if z_kpi_text else None
                
                # Structure KPIs correctly for the plotting function
                # Gaze KPI should be a list with 1 element (or None)
                gaze_kpi_list = [gaze_kpi] if gaze_kpi is not None else [None]
                # Position KPI should be a list with 3 elements  
                pose_kpi_list = [x_kpi, y_kpi, z_kpi]
                
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error",
                    "Please enter valid numeric values for KPIs."
                )
                return
            
            # Get all plots checkbox state
            all_plots = self.all_plots_checkbox.isChecked()
            
            # Collect slider values
            std_names, std_dict = self._collect_slider_values()
            
            # Load standard deviations into utility class
            self.analysis_utils.load_stds(std_names, std_dict)
            
            # Create progress dialog for computation
            progress = QProgressDialog(
                "Computing covariances...", "Cancel", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            progress.setValue(20)
            QApplication.processEvents()
            
            if progress.wasCanceled():
                return
                
            # Compute covariances
            self.analysis_utils.compute_covariances()
            progress.setValue(60)
            QApplication.processEvents()
            
            if progress.wasCanceled():
                return
                
            # Compute KPI covariances
            self.analysis_utils.compute_covariances_for_KPIs()
            progress.setValue(80)
            QApplication.processEvents()
            
            if progress.wasCanceled():
                return
                
            # Compute contributions
            self.analysis_utils.compute_contributions_to_KPIs()
            progress.setValue(90)
            QApplication.processEvents()
            
            if progress.wasCanceled():
                return
                
            # Plot results
            self.analysis_utils.plot_results(gaze_kpi_list, pose_kpi_list, all_plots)
            progress.setValue(100)
            progress.close()
            
            QMessageBox.information(
                self, "Success",
                "Plots generated successfully!"
            )
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(
                self, "Error",
                f"Error generating plots: {str(e)}"
            )
            return

    def create_tab_widget(self, name, slider_text, max_vals):
        """Create a tab widget with sliders.
        
        Args:
            name (str): Tab name
            slider_text (list): Labels for sliders
            max_vals (list): Maximum values for sliders
            
        Returns:
            QWidget: Tab widget containing the sliders
        """
        tab_widget = QWidget()
        layout = QVBoxLayout()
        
        slider_group = self._multi_slider_widget(name, slider_text, max_vals)
        layout.addWidget(slider_group)
        
        tab_widget.setLayout(layout)
        return tab_widget

    def create_window(self):
        """Create GUI.
        """

        #######################################################################
        # Frame with controls for generating derivatives.
        generate_derivatives_frame = QGroupBox("Generate data")
        generate_derivatives_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        generate_layout = QVBoxLayout()
        
        # Row 1: Number of samples
        samples_row = QHBoxLayout()
        samples_label = QLabel("Number of samples:")
        samples_label.setMinimumWidth(150)
        self.num_samples_input = QLineEdit()
        self.num_samples_input.setObjectName("-NUMSAMPLES-")
        self.num_samples_input.setMaximumWidth(60)
        self.num_samples_input.setText(
            self.settings.value("-num samples-", "")
        )
        samples_row.addWidget(samples_label)
        samples_row.addWidget(self.num_samples_input)
        samples_row.addStretch()

        # Row 2: Compute device selection
        device_row = QHBoxLayout()
        device_label = QLabel("Compute device:")
        device_label.setMinimumWidth(150)
        self.device_combo = QComboBox()
        self.device_combo.setObjectName("-DEVICE-")
        self.device_combo.addItems(["cpu", "cuda"])
        self.device_combo.setMaximumWidth(100)
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        device_row.addWidget(device_label)
        device_row.addWidget(self.device_combo)
        device_row.addStretch()

        
        # Row 3: Sampling parameters file
        sampling_row = QHBoxLayout()
        sampling_label = QLabel("Sampling-parameters file:")
        sampling_label.setMinimumWidth(150)
        self.sampling_filename_input = QLineEdit()
        self.sampling_filename_input.setObjectName("-SAMPLINGFILENAME-")
        self.sampling_filename_input.setMinimumWidth(300)
        self.sampling_filename_input.setText(
            self.settings.value("-sampling file name-", "")
        )
        
        self.sampling_browse_btn = QPushButton("Browse")
        self.sampling_browse_btn.clicked.connect(self._browse_sampling_file)
        
        self.generate_derivatives_btn = QPushButton("OK")
        self.generate_derivatives_btn.setObjectName("-GENERATEDERIVATIVES-")
        self.generate_derivatives_btn.setMaximumWidth(70)
        self.generate_derivatives_btn.clicked.connect(
            self._generate_derivatives
        )
        
        sampling_row.addWidget(sampling_label)
        sampling_row.addWidget(self.sampling_filename_input)
        sampling_row.addWidget(self.sampling_browse_btn)
        sampling_row.addWidget(self.generate_derivatives_btn)
        
        generate_layout.addLayout(samples_row)
        generate_layout.addLayout(device_row)
        generate_layout.addLayout(sampling_row)
        generate_derivatives_frame.setLayout(generate_layout)
        #######################################################################

        #######################################################################
        # Frame with controls for loading derivatives.
        load_derivatives_frame = QGroupBox("Load data")
        load_derivatives_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        load_layout = QVBoxLayout()
        
        # Derivatives file row
        derivatives_row = QHBoxLayout()
        derivatives_label = QLabel("Derivatives file:")
        derivatives_label.setMinimumWidth(150)
        self.derivatives_filename_input = QLineEdit()
        self.derivatives_filename_input.setObjectName("-DERIVATIVESFILENAME-")
        self.derivatives_filename_input.setMinimumWidth(300)
        self.derivatives_filename_input.setText(
            self.settings.value("-derivatives file name-", "")
        )
        
        self.derivatives_browse_btn = QPushButton("Browse")
        self.derivatives_browse_btn.clicked.connect(
            self._browse_derivatives_file
        )
        
        self.load_derivatives_btn = QPushButton("OK")
        self.load_derivatives_btn.setObjectName("-LOADDERIVATIVES-")
        self.load_derivatives_btn.setMaximumWidth(70)
        self.load_derivatives_btn.clicked.connect(self._load_derivatives)
        
        derivatives_row.addWidget(derivatives_label)
        derivatives_row.addWidget(self.derivatives_filename_input)
        derivatives_row.addWidget(self.derivatives_browse_btn)
        derivatives_row.addWidget(self.load_derivatives_btn)
        
        load_layout.addLayout(derivatives_row)
        load_derivatives_frame.setLayout(load_layout)
        #######################################################################

        #######################################################################
        # Frame with controls for loading input covariances.
        input_covariances_frame = QGroupBox("Load input covariances")
        input_covariances_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        covariances_layout = QVBoxLayout()
        
        # Covariances folder row
        covariances_row = QHBoxLayout()
        covariances_label = QLabel("Path to covariance files:")
        covariances_label.setMinimumWidth(150)
        self.covariances_folder_input = QLineEdit()
        self.covariances_folder_input.setObjectName("-COVARIANCESFOLDERNAME-")
        self.covariances_folder_input.setMinimumWidth(300)
        self.covariances_folder_input.setText(
            self.settings.value("-covariances folder name-", "")
        )
        
        self.covariances_browse_btn = QPushButton("Browse")
        self.covariances_browse_btn.clicked.connect(
            self._browse_covariances_folder
        )
        
        self.load_covariances_btn = QPushButton("OK")
        self.load_covariances_btn.setObjectName("-LOADCOVARIANCES-")
        self.load_covariances_btn.setMaximumWidth(70)
        self.load_covariances_btn.clicked.connect(self._load_covariances)
        
        covariances_row.addWidget(covariances_label)
        covariances_row.addWidget(self.covariances_folder_input)
        covariances_row.addWidget(self.covariances_browse_btn)
        covariances_row.addWidget(self.load_covariances_btn)
        
        covariances_layout.addLayout(covariances_row)
        input_covariances_frame.setLayout(covariances_layout)
        #######################################################################

        #######################################################################
        # Frame with fine tuning of standard deviation of input parameters.

        # Standard deviation of LED position
        led_tab = QWidget()
        led_layout = QVBoxLayout()
        led_slider_group = self._multi_slider_widget(
            "Standard dev. for coordinates [mm]:",
            ["X", "Y", "Z"],
            [1.0] * 3,
            keys=["LED X", "LED Y", "LED Z"]
        )
        led_layout.addWidget(led_slider_group)
        led_tab.setLayout(led_layout)

        # Standard deviation of camera extrinsic parameters
        extrinsics_tab = QWidget()
        extrinsics_layout = QVBoxLayout()
        extrinsics_row = QHBoxLayout()
        
        # Standard deviation for rotation
        rotation_group = self._multi_slider_widget(
            "Standard dev. for rotation [mrad]:",
            ["Pitch", "Yaw", "Roll"],
            [5.0] * 3,
            keys=["Pitch", "Yaw", "Roll"]
        )
        
        # Standard deviation of translation
        translation_group = self._multi_slider_widget(
            "Standard dev. for translation [mm]:",
            ["X", "Y", "Z"],
            [1.0] * 3,
            keys=["Camera X", "Camera Y", "Camera Z"]
        )
        
        extrinsics_row.addWidget(rotation_group)
        extrinsics_row.addStretch()
        extrinsics_row.addWidget(translation_group)
        extrinsics_layout.addLayout(extrinsics_row)
        extrinsics_tab.setLayout(extrinsics_layout)

        # Standard deviation of camera intrinsic parameters.
        intrinsics_tab = QWidget()
        intrinsics_layout = QVBoxLayout()
        intrinsics_row = QHBoxLayout()
        
        # Standard deviation of pinhole parameters
        pinhole_group = self._multi_slider_widget(
            "Standard dev. for pinhole parameters [pix]:",
            ["px", "py", "fx", "fy"],
            max_vals=[0.5] * 4,
            keys=["px", "py", "fx", "fy"]
        )
        
        # Standard deviation of radial distortion parameters
        distortion_group = self._multi_slider_widget(
            "Standard dev. for distortion parameters [adim.]:",
            ["cx", "cy", "k0", "k1", "k2"],
            [0.2] * 5,
            keys=["cx", "cy", "k0", "k1", "k2"]
        )
        
        intrinsics_row.addWidget(pinhole_group)
        intrinsics_row.addStretch()
        intrinsics_row.addWidget(distortion_group)
        intrinsics_layout.addLayout(intrinsics_row)
        intrinsics_tab.setLayout(intrinsics_layout)

        # Standard deviation of location of image features.
        features_tab = QWidget()
        features_layout = QVBoxLayout()
        features_slider_group = self._multi_slider_widget(
            "Standard dev. for features [pix]:",
            ["Glint", "Pupil", "Limbus"],
            [1.0] * 3,
            keys=["Glint", "Pupil", "Limbus"]
        )
        features_layout.addWidget(features_slider_group)
        features_tab.setLayout(features_layout)

        # Standard deviation of all inputs.
        input_standard_deviation_frame = QGroupBox("Noise fine tuning")
        input_standard_deviation_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        noise_layout = QVBoxLayout()
        
        # Create tab widget and add tabs
        self.noise_tab_widget = QTabWidget()
        self.noise_tab_widget.addTab(led_tab, "LEDs")
        self.noise_tab_widget.addTab(extrinsics_tab, "Camera extrinsics")
        self.noise_tab_widget.addTab(intrinsics_tab, "Camera intrinsics")
        self.noise_tab_widget.addTab(features_tab, "Image features")
        
        noise_layout.addWidget(self.noise_tab_widget)
        input_standard_deviation_frame.setLayout(noise_layout)

        #######################################################################
        # Frame with controls for plots.
        plot_parameters_frame = QGroupBox("Standard deviations at KPIs")
        plot_parameters_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        plot_params_layout = QVBoxLayout()
        plot_params_row = QHBoxLayout()
        
        # Gaze KPI input
        gaze_label = QLabel("Gaze [deg]:")
        self.gaze_kpi_input = QLineEdit("0.75")
        self.gaze_kpi_input.setObjectName("gaze KPI")
        self.gaze_kpi_input.setMaximumWidth(60)
        
        # Position KPI inputs
        position_label = QLabel("X, Y, Z position [mm]:")
        self.x_position_kpi_input = QLineEdit("0.75")
        self.x_position_kpi_input.setObjectName("x position KPI")
        self.x_position_kpi_input.setMaximumWidth(60)
        
        self.y_position_kpi_input = QLineEdit("0.75")
        self.y_position_kpi_input.setObjectName("y position KPI")
        self.y_position_kpi_input.setMaximumWidth(60)
        
        self.z_position_kpi_input = QLineEdit("")
        self.z_position_kpi_input.setObjectName("z position KPI")
        self.z_position_kpi_input.setMaximumWidth(60)
        
        plot_params_row.addWidget(gaze_label)
        plot_params_row.addWidget(self.gaze_kpi_input)
        plot_params_row.addStretch()
        plot_params_row.addWidget(position_label)
        plot_params_row.addWidget(self.x_position_kpi_input)
        plot_params_row.addWidget(self.y_position_kpi_input)
        plot_params_row.addWidget(self.z_position_kpi_input)
        
        plot_params_layout.addLayout(plot_params_row)
        plot_parameters_frame.setLayout(plot_params_layout)

        plot_generation_frame = QGroupBox("Plot controls")
        plot_generation_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        plot_gen_layout = QVBoxLayout()
        plot_gen_row = QHBoxLayout()
        
        self.all_plots_checkbox = QCheckBox("All plots")
        self.all_plots_checkbox.setObjectName("all plots")
        
        self.plot_button = QPushButton("Plot")
        self.plot_button.setObjectName("-PLOT-")
        self.plot_button.setMaximumWidth(70)
        self.plot_button.clicked.connect(self._plot_results)
        
        plot_gen_row.addWidget(self.all_plots_checkbox)
        plot_gen_row.addWidget(self.plot_button)
        
        plot_gen_layout.addLayout(plot_gen_row)
        plot_generation_frame.setLayout(plot_gen_layout)

        # Create main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Add widgets to main layout
        main_layout.addWidget(generate_derivatives_frame)
        
        # "OR" separator
        or_label = QLabel("OR")
        or_label.setStyleSheet("QLabel { font-size: 16px; }")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(or_label)
        
        main_layout.addWidget(load_derivatives_frame)
        
        # Horizontal separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(3)
        separator.setMidLineWidth(6)
        separator.setMinimumHeight(8)
        separator.setStyleSheet("QFrame { border-top: 4px solid #444; }")
        main_layout.addWidget(separator)
        
        main_layout.addWidget(input_covariances_frame)
        main_layout.addWidget(input_standard_deviation_frame)
        
        # Bottom row with plot controls
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(plot_parameters_frame)
        bottom_row.addStretch()
        bottom_row.addWidget(plot_generation_frame)
        main_layout.addLayout(bottom_row)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.derivatives_file_name = None
        self.covariances_folder_name = None

        # Set window properties
        self.setWindowTitle("Sensitivity Analysis Tool")
        self.setMinimumSize(800, 600)
        
        # Show the window
        self.show()


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    sensitivity_app = SensitivityAnalysisAPP()
    sys.exit(app.exec())

=======
"""App for eye-tracking sensitivity analysis - PySide6 Version.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import matplotlib.pyplot as plt
import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QTabWidget, QLabel, QLineEdit, QPushButton, QSlider,
    QCheckBox, QFrame, QSpacerItem, QSizePolicy, QFileDialog,
    QMessageBox, QProgressDialog
)
from PySide6.QtCore import Qt, QSettings, Signal, QObject
from PySide6.QtGui import QCloseEvent
import sensitivity_analysis_app_utils


class SensitivityAnalysisAPP(QMainWindow):
    """Class with GUI elements for eye-tracking sensitivity analysis - PySide6 Version.
    """

    def __init__(self):
        """Create the app and initialize the event loop.
        """
        super().__init__()
        
        # Initialize QSettings for user preferences
        self.settings = QSettings("SEET", "SensitivityAnalysisApp")
        
        self.analysis_utils = \
            sensitivity_analysis_app_utils.SensitivityAnalysisAppUtils()

        self.create_window()
        
        # Show the window
        self.show()

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event (replaces sg.WIN_CLOSED).
        """
        self.analysis_utils.close()
        event.accept()

    def close_application(self):
        """Graciously exit.
        """
        self.close()

    def _labeled_slider(self, text, max_val, min_val=0.0, key=None):
        """Create a PySide6 labeled slider in a group box.

        Args:
            text (string): name of parameter controlled by slider.
            max_val (float): maximum value of slider.
            min_val (float, optional): minimum value of slider. Defaults to 0.0.
            key (str, optional): object name used to access slider. Defaults to None.

        Returns:
            QGroupBox: borderless PySide6 group box holding slider.
        """
        group_box = QGroupBox()
        group_box.setStyleSheet("border: 0px;")

        slider = QSlider(Qt.Vertical)
        slider.setRange(int(min_val * 20), int(max_val * 20))  # Scale for resolution
        slider.setValue(int(max_val * 10))  # Default to middle
        slider.setFixedSize(40, 100)

        if key:
            slider.setObjectName(key)

        # Parameter label above everything
        param_label = QLabel(text)
        param_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        param_label.setStyleSheet("font-size: 14px;")

        # Max and min labels
        max_label = QLabel(f"{max_val:.2f}")
        max_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        min_label = QLabel(f"{min_val:.2f}")
        min_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Value label beside slider, updates dynamically
        value_label = QLabel(f"{slider.value() / 20.0:.2f}")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("font-weight: bold;")
        slider.valueChanged.connect(lambda val: value_label.setText(f"{val / 20.0:.2f}"))

        # Layout: param label, max label, slider+value label (overlap), min label
        layout = QVBoxLayout()
        layout.addWidget(param_label)
        layout.addWidget(max_label)

        # Stack slider and value label using QStackedLayout or QHBoxLayout
        slider_value_layout = QHBoxLayout()
        slider_value_layout.addStretch()
        slider_value_layout.addWidget(slider)
        slider_value_layout.addWidget(value_label)
        slider_value_layout.addStretch()
        layout.addLayout(slider_value_layout)

        layout.addWidget(min_label)
        group_box.setLayout(layout)

        return group_box

    def _multi_slider_widget(self, name, text, max_vals, min_vals=None, keys=None):
        """Creates a named group box of side-by-side labeled sliders.

        Args:
            name (str): name of group box.
            text (list of str): name of each slider in the group box.
            max_vals (list of float): maximum value of each slider.
            min_vals (list of float, optional): minimum value of each slider.
                Defaults to None, in which case the minimum value is zero.
            keys (list of str, optional): object name for each slider.
                Defaults to None, in which case the sliders will not have object names.

        Returns:
            QGroupBox: PySide6 group box holding the sliders.
        """
        num_sliders = len(text)
        assert (num_sliders == len(max_vals))
        if min_vals is None:
            min_vals = len(max_vals) * [0.0]
        if keys is None:
            keys = len(max_vals) * [None]
        else:
            assert (len(keys) == len(max_vals))

        # Create horizontal layout for sliders
        layout = QHBoxLayout()
        
        for i in range(num_sliders):
            slider_widget = self._labeled_slider(
                text[i], max_vals[i], min_val=min_vals[i], key=keys[i]
            )
            layout.addWidget(slider_widget)

        group_box = QGroupBox(name)
        group_box.setStyleSheet("QGroupBox { font-size: 14px; }")
        group_box.setLayout(layout)
        return group_box

    # Signal handlers for file browse functionality
    def _browse_sampling_file(self):
        """Browse for sampling parameters file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sampling Parameters File",
            self.sampling_filename_input.text(),
            "All Files (*)"
        )
        if file_path:
            self.sampling_filename_input.setText(file_path)

    def _browse_derivatives_file(self):
        """Browse for derivatives file."""
        # Default to examples directory first, then user's last path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        default_examples_dir = os.path.join(project_root, "seet", "sensitivity_analysis", "default_derivatives")
        
        start_dir = self.derivatives_filename_input.text()
        if not start_dir or not os.path.exists(start_dir):
            start_dir = self.settings.value("-derivatives file name-", default_examples_dir)
        if not start_dir or not os.path.exists(start_dir):
            start_dir = default_examples_dir
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Derivatives File (Examples available in default_derivatives folder)",
            start_dir,
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if file_path:
            self.derivatives_filename_input.setText(file_path)

    def _browse_covariances_folder(self):
        """Browse for covariances folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Covariances Folder",
            self.covariances_folder_input.text()
        )
        if folder_path:
            self.covariances_folder_input.setText(folder_path)

    # Action button handlers (placeholders for now)
    def _generate_derivatives(self):
        """Handle generate derivatives button click."""
        try:
            # Get inputs from GUI
            num_samples_text = self.num_samples_input.text().strip()
            sampling_file = self.sampling_filename_input.text().strip()
            
            # Validate inputs
            if not sampling_file:
                QMessageBox.warning(
                    self, "Input Error",
                    "Please specify a sampling parameters file."
                )
                return
                
            if not os.path.exists(sampling_file):
                QMessageBox.warning(
                    self, "File Error",
                    f"Sampling parameters file not found: {sampling_file}"
                )
                return
                
            # Parse number of samples
            num_samples = None
            if num_samples_text:
                try:
                    num_samples = int(num_samples_text)
                    if num_samples <= 0:
                        raise ValueError("Number of samples must be positive")
                except ValueError:
                    QMessageBox.warning(
                        self, "Input Error",
                        "Please enter a valid positive number for samples."
                    )
                    return
            else:
                QMessageBox.warning(
                        self, "Input Error",
                        "Please enter a valid positive number for samples."
                    )
                return
            
            # Save settings
            self.settings.setValue("-num samples-", num_samples_text)
            self.settings.setValue("-sampling file name-", sampling_file)
            
            # Create progress dialog
            progress = QProgressDialog(
                "Generating derivatives...", "Cancel", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setValue(0)
            progress.show()
            
            # Generate data using the generator
            generator = self.analysis_utils.generate_data(
                sampling_file, num_samples
            )
            # Approximate 40 measurements per sample (2 eyes, 20 "gaze" directions each)
            measure_count = num_samples * 40 
            sample_count = 0
            
            try:
                while True:
                    try:
                        sample_count = next(generator)
                        # Update progress (rough estimate since we don't know
                        # total in advance)
                        if num_samples:
                            progress_percent = min(
                                int((sample_count / measure_count) * 100), 99
                            )
                        else:
                            # Just show activity
                            progress_percent = min(sample_count, 99)
                        progress.setValue(progress_percent)
                        
                        # Check if user cancelled
                        if progress.wasCanceled():
                            progress.close()
                            QMessageBox.information(
                                self, "Cancelled",
                                "Data generation was cancelled."
                            )
                            return
                            
                        # Process events to keep GUI responsive
                        QApplication.processEvents()
                        
                    except StopIteration:
                        # Generation completed successfully
                        progress.setValue(100)
                        break
                        
            except Exception as e:
                progress.close()
                QMessageBox.critical(
                    self, "Generation Error",
                    f"Error during data generation: {str(e)}"
                )
                return
            
            progress.close()
            
            # Prompt to save derivatives
            reply = QMessageBox.question(
                self, "Save Derivatives",
                f"Data generation completed! Processed {sample_count} samples.\n\n"
                "Would you like to save the derivatives for future use?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Get save directory - default to project results folder
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                default_results_dir = os.path.join(project_root, "results", "derivatives")
                
                last_saved_path = self.settings.value("-last saved derivative path-", default_results_dir)
                output_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Save derivatives for future use",
                    last_saved_path
                )
                
                if output_dir:
                    try:
                        # Save the derivatives
                        derivatives_file = self.analysis_utils.save_data(output_dir)
                        
                        # Update the derivatives filename input field
                        self.derivatives_filename_input.setText(derivatives_file)
                        
                        # Update settings
                        self.settings.setValue("-last saved derivative path-", output_dir)
                        self.settings.setValue("-derivatives file name-", derivatives_file)
                        saved_paths = self.settings.value("-saved derivative path-", [])
                        if not isinstance(saved_paths, list):
                            saved_paths = []
                        if output_dir not in saved_paths:
                            saved_paths.append(output_dir)
                            self.settings.setValue("-saved derivative path-", saved_paths)
                        
                        QMessageBox.information(
                            self, "Save Successful",
                            f"Derivatives saved successfully to:\n{derivatives_file}"
                        )
                        
                    except Exception as e:
                        QMessageBox.critical(
                            self, "Save Error",
                            f"Error saving derivatives: {str(e)}"
                        )
            else:
                QMessageBox.information(
                    self, "Complete",
                    f"Data generation completed! Processed {sample_count} samples."
                )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An unexpected error occurred: {str(e)}"
            )
            return

    def _load_derivatives(self):
        """Handle load derivatives button click."""
        try:
            # Get derivatives file path from GUI
            derivatives_file = self.derivatives_filename_input.text().strip()
            
            # Validate input
            if not derivatives_file:
                QMessageBox.warning(
                    self, "Input Error",
                    "Please specify a derivatives file."
                )
                return
                
            if not os.path.exists(derivatives_file):
                QMessageBox.warning(
                    self, "File Error",
                    f"Derivatives file not found: {derivatives_file}"
                )
                return
            
            # Save setting
            self.settings.setValue("-derivatives file name-", derivatives_file)
            
            # Load derivatives
            self.analysis_utils.load_derivatives(derivatives_file)
            self.derivatives_file_name = derivatives_file
            
            QMessageBox.information(
                self, "Success",
                "Derivatives loaded successfully!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error loading derivatives: {str(e)}"
            )
            return

    def _load_covariances(self):
        """Handle load covariances button click."""
        try:
            # Get covariances folder path from GUI
            covariances_folder = self.covariances_folder_input.text().strip()
            
            # Validate input
            if not covariances_folder:
                QMessageBox.warning(
                    self, "Input Error",
                    "Please specify a covariances folder."
                )
                return
                
            if not os.path.exists(covariances_folder):
                QMessageBox.warning(
                    self, "Folder Error",
                    f"Covariances folder not found: {covariances_folder}"
                )
                return
                
            if not os.path.isdir(covariances_folder):
                QMessageBox.warning(
                    self, "Folder Error",
                    f"Path is not a directory: {covariances_folder}"
                )
                return
            
            # Save setting
            self.settings.setValue(
                "-covariances folder name-", covariances_folder
            )
            
            # Load configuration
            self.analysis_utils.load_configuration(covariances_folder)
            self.covariances_folder_name = covariances_folder
            
            QMessageBox.information(
                self, "Success",
                "Covariances loaded successfully!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error loading covariances: {str(e)}"
            )
            return

    def _collect_slider_values(self):
        """Collect standard deviation values from all sliders."""
        std_dict = {}
        std_names = []
        
        # Define slider keys in the order expected by the utility class
        slider_keys = [
            # LEDs (3 components)
            "LED X", "LED Y", "LED Z",
            # Camera extrinsics (6 components)
            "Pitch", "Yaw", "Roll", "Camera X", "Camera Y", "Camera Z",
            # Camera intrinsics (9 components)
            "px", "py", "fx", "fy", "cx", "cy", "k0", "k1", "k2",
            # Image features (3 components)
            "Glint", "Pupil", "Limbus"
        ]
        
        for key in slider_keys:
            slider = self.findChild(QSlider, key)
            if slider:
                # Convert slider value back to float (remember we scaled by 20)
                value = slider.value() / 20.0
                std_dict[key] = value
                std_names.append(key)
            else:
                # Default value if slider not found
                std_dict[key] = 0.5
                std_names.append(key)
                
        return std_names, std_dict

    def _plot_results(self):
        """Handle plot button click."""
        try:
            # Check if derivatives are loaded
            if self.analysis_utils.data_dictionary is None:
                QMessageBox.warning(
                    self, "Data Error",
                    "Please load or generate derivatives first."
                )
                return
                
            # Check if covariances are loaded
            if (self.analysis_utils.leds_covariance_calculator is None or
                    self.analysis_utils.camera_covariance_calculator is None or
                    self.analysis_utils.features_covariance_calculator is
                    None):
                QMessageBox.warning(
                    self, "Configuration Error",
                    "Please load input covariances first."
                )
                return
            
            # Get KPI values from GUI
            try:
                gaze_kpi_text = self.gaze_kpi_input.text().strip()
                gaze_kpi = float(gaze_kpi_text) if gaze_kpi_text else None
                
                x_kpi_text = self.x_position_kpi_input.text().strip()
                y_kpi_text = self.y_position_kpi_input.text().strip()
                z_kpi_text = self.z_position_kpi_input.text().strip()
                
                x_kpi = float(x_kpi_text) if x_kpi_text else None
                y_kpi = float(y_kpi_text) if y_kpi_text else None
                z_kpi = float(z_kpi_text) if z_kpi_text else None
                
                # Structure KPIs correctly for the plotting function
                # Gaze KPI should be a list with 1 element (or None)
                gaze_kpi_list = [gaze_kpi] if gaze_kpi is not None else [None]
                # Position KPI should be a list with 3 elements  
                pose_kpi_list = [x_kpi, y_kpi, z_kpi]
                
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error",
                    "Please enter valid numeric values for KPIs."
                )
                return
            
            # Get all plots checkbox state
            all_plots = self.all_plots_checkbox.isChecked()
            
            # Collect slider values
            std_names, std_dict = self._collect_slider_values()
            
            # Load standard deviations into utility class
            self.analysis_utils.load_stds(std_names, std_dict)
            
            # Create progress dialog for computation
            progress = QProgressDialog(
                "Computing covariances...", "Cancel", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            progress.setValue(20)
            QApplication.processEvents()
            
            if progress.wasCanceled():
                return
                
            # Compute covariances
            self.analysis_utils.compute_covariances()
            progress.setValue(60)
            QApplication.processEvents()
            
            if progress.wasCanceled():
                return
                
            # Compute KPI covariances
            self.analysis_utils.compute_covariances_for_KPIs()
            progress.setValue(80)
            QApplication.processEvents()
            
            if progress.wasCanceled():
                return
                
            # Compute contributions
            self.analysis_utils.compute_contributions_to_KPIs()
            progress.setValue(90)
            QApplication.processEvents()
            
            if progress.wasCanceled():
                return
                
            # Plot results
            self.analysis_utils.plot_results(gaze_kpi_list, pose_kpi_list, all_plots)
            progress.setValue(100)
            progress.close()
            
            QMessageBox.information(
                self, "Success",
                "Plots generated successfully!"
            )
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(
                self, "Error",
                f"Error generating plots: {str(e)}"
            )
            return

    def create_tab_widget(self, name, slider_text, max_vals):
        """Create a tab widget with sliders.
        
        Args:
            name (str): Tab name
            slider_text (list): Labels for sliders
            max_vals (list): Maximum values for sliders
            
        Returns:
            QWidget: Tab widget containing the sliders
        """
        tab_widget = QWidget()
        layout = QVBoxLayout()
        
        slider_group = self._multi_slider_widget(name, slider_text, max_vals)
        layout.addWidget(slider_group)
        
        tab_widget.setLayout(layout)
        return tab_widget

    def create_window(self):
        """Create GUI.
        """

        #######################################################################
        # Frame with controls for generating derivatives.
        generate_derivatives_frame = QGroupBox("Generate data")
        generate_derivatives_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        generate_layout = QVBoxLayout()
        
        # Row 1: Number of samples
        samples_row = QHBoxLayout()
        samples_label = QLabel("Number of samples:")
        samples_label.setMinimumWidth(150)
        self.num_samples_input = QLineEdit()
        self.num_samples_input.setObjectName("-NUMSAMPLES-")
        self.num_samples_input.setMaximumWidth(60)
        self.num_samples_input.setText(
            self.settings.value("-num samples-", "")
        )
        samples_row.addWidget(samples_label)
        samples_row.addWidget(self.num_samples_input)
        samples_row.addStretch()
        
        # Row 2: Sampling parameters file
        sampling_row = QHBoxLayout()
        sampling_label = QLabel("Sampling-parameters file:")
        sampling_label.setMinimumWidth(150)
        self.sampling_filename_input = QLineEdit()
        self.sampling_filename_input.setObjectName("-SAMPLINGFILENAME-")
        self.sampling_filename_input.setMinimumWidth(300)
        self.sampling_filename_input.setText(
            self.settings.value("-sampling file name-", "")
        )
        
        self.sampling_browse_btn = QPushButton("Browse")
        self.sampling_browse_btn.clicked.connect(self._browse_sampling_file)
        
        self.generate_derivatives_btn = QPushButton("OK")
        self.generate_derivatives_btn.setObjectName("-GENERATEDERIVATIVES-")
        self.generate_derivatives_btn.setMaximumWidth(70)
        self.generate_derivatives_btn.clicked.connect(
            self._generate_derivatives
        )
        
        sampling_row.addWidget(sampling_label)
        sampling_row.addWidget(self.sampling_filename_input)
        sampling_row.addWidget(self.sampling_browse_btn)
        sampling_row.addWidget(self.generate_derivatives_btn)
        
        generate_layout.addLayout(samples_row)
        generate_layout.addLayout(sampling_row)
        generate_derivatives_frame.setLayout(generate_layout)
        #######################################################################

        #######################################################################
        # Frame with controls for loading derivatives.
        load_derivatives_frame = QGroupBox("Load data")
        load_derivatives_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        load_layout = QVBoxLayout()
        
        # Derivatives file row
        derivatives_row = QHBoxLayout()
        derivatives_label = QLabel("Derivatives file:")
        derivatives_label.setMinimumWidth(150)
        self.derivatives_filename_input = QLineEdit()
        self.derivatives_filename_input.setObjectName("-DERIVATIVESFILENAME-")
        self.derivatives_filename_input.setMinimumWidth(300)
        self.derivatives_filename_input.setText(
            self.settings.value("-derivatives file name-", "")
        )
        
        self.derivatives_browse_btn = QPushButton("Browse")
        self.derivatives_browse_btn.clicked.connect(
            self._browse_derivatives_file
        )
        
        self.load_derivatives_btn = QPushButton("OK")
        self.load_derivatives_btn.setObjectName("-LOADDERIVATIVES-")
        self.load_derivatives_btn.setMaximumWidth(70)
        self.load_derivatives_btn.clicked.connect(self._load_derivatives)
        
        derivatives_row.addWidget(derivatives_label)
        derivatives_row.addWidget(self.derivatives_filename_input)
        derivatives_row.addWidget(self.derivatives_browse_btn)
        derivatives_row.addWidget(self.load_derivatives_btn)
        
        load_layout.addLayout(derivatives_row)
        load_derivatives_frame.setLayout(load_layout)
        #######################################################################

        #######################################################################
        # Frame with controls for loading input covariances.
        input_covariances_frame = QGroupBox("Load input covariances")
        input_covariances_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        covariances_layout = QVBoxLayout()
        
        # Covariances folder row
        covariances_row = QHBoxLayout()
        covariances_label = QLabel("Path to covariance files:")
        covariances_label.setMinimumWidth(150)
        self.covariances_folder_input = QLineEdit()
        self.covariances_folder_input.setObjectName("-COVARIANCESFOLDERNAME-")
        self.covariances_folder_input.setMinimumWidth(300)
        self.covariances_folder_input.setText(
            self.settings.value("-covariances folder name-", "")
        )
        
        self.covariances_browse_btn = QPushButton("Browse")
        self.covariances_browse_btn.clicked.connect(
            self._browse_covariances_folder
        )
        
        self.load_covariances_btn = QPushButton("OK")
        self.load_covariances_btn.setObjectName("-LOADCOVARIANCES-")
        self.load_covariances_btn.setMaximumWidth(70)
        self.load_covariances_btn.clicked.connect(self._load_covariances)
        
        covariances_row.addWidget(covariances_label)
        covariances_row.addWidget(self.covariances_folder_input)
        covariances_row.addWidget(self.covariances_browse_btn)
        covariances_row.addWidget(self.load_covariances_btn)
        
        covariances_layout.addLayout(covariances_row)
        input_covariances_frame.setLayout(covariances_layout)
        #######################################################################

        #######################################################################
        # Frame with fine tuning of standard deviation of input parameters.

        # Standard deviation of LED position
        led_tab = QWidget()
        led_layout = QVBoxLayout()
        led_slider_group = self._multi_slider_widget(
            "Standard dev. for coordinates [mm]:",
            ["X", "Y", "Z"],
            [1.0] * 3,
            keys=["LED X", "LED Y", "LED Z"]
        )
        led_layout.addWidget(led_slider_group)
        led_tab.setLayout(led_layout)

        # Standard deviation of camera extrinsic parameters
        extrinsics_tab = QWidget()
        extrinsics_layout = QVBoxLayout()
        extrinsics_row = QHBoxLayout()
        
        # Standard deviation for rotation
        rotation_group = self._multi_slider_widget(
            "Standard dev. for rotation [mrad]:",
            ["Pitch", "Yaw", "Roll"],
            [5.0] * 3,
            keys=["Pitch", "Yaw", "Roll"]
        )
        
        # Standard deviation of translation
        translation_group = self._multi_slider_widget(
            "Standard dev. for translation [mm]:",
            ["X", "Y", "Z"],
            [1.0] * 3,
            keys=["Camera X", "Camera Y", "Camera Z"]
        )
        
        extrinsics_row.addWidget(rotation_group)
        extrinsics_row.addStretch()
        extrinsics_row.addWidget(translation_group)
        extrinsics_layout.addLayout(extrinsics_row)
        extrinsics_tab.setLayout(extrinsics_layout)

        # Standard deviation of camera intrinsic parameters.
        intrinsics_tab = QWidget()
        intrinsics_layout = QVBoxLayout()
        intrinsics_row = QHBoxLayout()
        
        # Standard deviation of pinhole parameters
        pinhole_group = self._multi_slider_widget(
            "Standard dev. for pinhole parameters [pix]:",
            ["px", "py", "fx", "fy"],
            max_vals=[0.5] * 4,
            keys=["px", "py", "fx", "fy"]
        )
        
        # Standard deviation of radial distortion parameters
        distortion_group = self._multi_slider_widget(
            "Standard dev. for distortion parameters [adim.]:",
            ["cx", "cy", "k0", "k1", "k2"],
            [0.2] * 5,
            keys=["cx", "cy", "k0", "k1", "k2"]
        )
        
        intrinsics_row.addWidget(pinhole_group)
        intrinsics_row.addStretch()
        intrinsics_row.addWidget(distortion_group)
        intrinsics_layout.addLayout(intrinsics_row)
        intrinsics_tab.setLayout(intrinsics_layout)

        # Standard deviation of location of image features.
        features_tab = QWidget()
        features_layout = QVBoxLayout()
        features_slider_group = self._multi_slider_widget(
            "Standard dev. for features [pix]:",
            ["Glint", "Pupil", "Limbus"],
            [1.0] * 3,
            keys=["Glint", "Pupil", "Limbus"]
        )
        features_layout.addWidget(features_slider_group)
        features_tab.setLayout(features_layout)

        # Standard deviation of all inputs.
        input_standard_deviation_frame = QGroupBox("Noise fine tuning")
        input_standard_deviation_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        noise_layout = QVBoxLayout()
        
        # Create tab widget and add tabs
        self.noise_tab_widget = QTabWidget()
        self.noise_tab_widget.addTab(led_tab, "LEDs")
        self.noise_tab_widget.addTab(extrinsics_tab, "Camera extrinsics")
        self.noise_tab_widget.addTab(intrinsics_tab, "Camera intrinsics")
        self.noise_tab_widget.addTab(features_tab, "Image features")
        
        noise_layout.addWidget(self.noise_tab_widget)
        input_standard_deviation_frame.setLayout(noise_layout)

        #######################################################################
        # Frame with controls for plots.
        plot_parameters_frame = QGroupBox("Standard deviations at KPIs")
        plot_parameters_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        plot_params_layout = QVBoxLayout()
        plot_params_row = QHBoxLayout()
        
        # Gaze KPI input
        gaze_label = QLabel("Gaze [deg]:")
        self.gaze_kpi_input = QLineEdit("0.75")
        self.gaze_kpi_input.setObjectName("gaze KPI")
        self.gaze_kpi_input.setMaximumWidth(60)
        
        # Position KPI inputs
        position_label = QLabel("X, Y, Z position [mm]:")
        self.x_position_kpi_input = QLineEdit("0.75")
        self.x_position_kpi_input.setObjectName("x position KPI")
        self.x_position_kpi_input.setMaximumWidth(60)
        
        self.y_position_kpi_input = QLineEdit("0.75")
        self.y_position_kpi_input.setObjectName("y position KPI")
        self.y_position_kpi_input.setMaximumWidth(60)
        
        self.z_position_kpi_input = QLineEdit("")
        self.z_position_kpi_input.setObjectName("z position KPI")
        self.z_position_kpi_input.setMaximumWidth(60)
        
        plot_params_row.addWidget(gaze_label)
        plot_params_row.addWidget(self.gaze_kpi_input)
        plot_params_row.addStretch()
        plot_params_row.addWidget(position_label)
        plot_params_row.addWidget(self.x_position_kpi_input)
        plot_params_row.addWidget(self.y_position_kpi_input)
        plot_params_row.addWidget(self.z_position_kpi_input)
        
        plot_params_layout.addLayout(plot_params_row)
        plot_parameters_frame.setLayout(plot_params_layout)

        plot_generation_frame = QGroupBox("Plot controls")
        plot_generation_frame.setStyleSheet("QGroupBox { font-size: 18px; }")
        plot_gen_layout = QVBoxLayout()
        plot_gen_row = QHBoxLayout()
        
        self.all_plots_checkbox = QCheckBox("All plots")
        self.all_plots_checkbox.setObjectName("all plots")
        
        self.plot_button = QPushButton("Plot")
        self.plot_button.setObjectName("-PLOT-")
        self.plot_button.setMaximumWidth(70)
        self.plot_button.clicked.connect(self._plot_results)
        
        plot_gen_row.addWidget(self.all_plots_checkbox)
        plot_gen_row.addWidget(self.plot_button)
        
        plot_gen_layout.addLayout(plot_gen_row)
        plot_generation_frame.setLayout(plot_gen_layout)

        # Create main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Add widgets to main layout
        main_layout.addWidget(generate_derivatives_frame)
        
        # "OR" separator
        or_label = QLabel("OR")
        or_label.setStyleSheet("QLabel { font-size: 16px; }")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(or_label)
        
        main_layout.addWidget(load_derivatives_frame)
        
        # Horizontal separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(3)
        separator.setMidLineWidth(6)
        separator.setMinimumHeight(8)
        separator.setStyleSheet("QFrame { border-top: 4px solid #444; }")
        main_layout.addWidget(separator)
        
        main_layout.addWidget(input_covariances_frame)
        main_layout.addWidget(input_standard_deviation_frame)
        
        # Bottom row with plot controls
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(plot_parameters_frame)
        bottom_row.addStretch()
        bottom_row.addWidget(plot_generation_frame)
        main_layout.addLayout(bottom_row)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.derivatives_file_name = None
        self.covariances_folder_name = None

        # Set window properties
        self.setWindowTitle("Sensitivity Analysis Tool")
        self.setMinimumSize(800, 600)
        
        # Show the window
        self.show()


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    sensitivity_app = SensitivityAnalysisAPP()
    sys.exit(app.exec())

>>>>>>> origin
