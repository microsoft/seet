# SEET

SEET (Simulation Engine for Eye Tracking) is a Python library for modeling, simulating, and analyzing synthetic eye tracking systems. It provides effective tools for geometry, device modeling, optimization, sensitivity analysis, and visualization, supporting research and development in eye tracking.

## Features

- Modular core for geometric primitives and mathematical operations
- Device modeling (cameras, LEDs, occluders, etc.)
- User modeling (eye, cornea, pupil, eyelids, limbus)
- Scene and sampler utilities for synthetic data generation
- Optimization routines for ellipse fitting and error analysis
- Sensitivity analysis for model robustness
- Visualization tools for primitives, devices, and scenes
- Extensible architecture for custom models and analysis

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Optionally, create a virtual environment using the provided `SEET_env.yaml`:

```bash
conda env create -f SEET_env.yaml
conda activate seet
```

Install the SEET package:
```bash
pip install -e .
```

## Usage

Import SEET modules in your Python scripts:

```python
from seet.core import geometry, node
from seet.device import device_model, leds
from seet.user import eye_model, cornea_model
```

Explore the `notebooks/` directory for example workflows and analysis.

## Testing

Run all unittests:

```bash
python -m unittest discover tests
```

## Repository Structure

- `seet/` — Main library modules
- `tests/` — Unit tests for all modules
- `notebooks/` — Example Jupyter notebooks
- `requirements.txt` — Python dependencies
- `SEET_env.yaml` — Conda environment specification

## Contributing

Contributions are welcome! Please submit issues and pull requests via GitHub.

## License

This project is licensed under the MIT License.
