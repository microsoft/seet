# Default Derivatives

This directory contains pre-computed derivative examples for the SEET sensitivity analysis tool.

## Available Examples

### small_example_derivatives.pkl
- **Sample count**: 10-50 samples
- **Purpose**: Quick testing and demonstration
- **Computation time**: < 1 minute
- **Use case**: Initial exploration, debugging, tutorials

### medium_example_derivatives.pkl  
- **Sample count**: 100-500 samples
- **Purpose**: Moderate accuracy analysis
- **Computation time**: 5-15 minutes
- **Use case**: Development testing, moderate precision analysis

### full_example_derivatives.pkl
- **Sample count**: 1000+ samples
- **Purpose**: High accuracy analysis
- **Computation time**: 30+ minutes
- **Use case**: Production analysis, research results

## Usage

1. Load any of these files using the "Load data" section in the sensitivity analysis GUI
2. Browse to this directory and select the appropriate .pkl file
3. Continue with loading covariances and running analysis

## Generating New Examples

To create new example derivatives:

1. Use the "Generate data" section in the GUI
2. Configure appropriate sampling parameters
3. Choose the number of samples based on desired accuracy vs. computation time
4. Save the results to this directory for future use

## Configuration

These derivatives were generated using the default sampling parameters from:
- `seet/sampler/default_sampler/default_scene_sampler.json`
- `seet/sampler/default_sampler/default_device_sampler.json` 
- `seet/sampler/default_sampler/default_user_sampler.json`

And default covariances from:
- `seet/sensitivity_analysis/default_covariances/`