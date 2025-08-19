"""
Utility tools for experiments with depth-perception errors.

Collection of functions to generate and analyze data to evaluate
depth-perception errors.
"""


import kiruna
import torch


def sample_points_on_plane(
    error_simulator,
    depth_mm,
    azimuth_samples_deg,
    elevation_samples_deg
):
    """
    Create a meshgrid X, Y, Z in 3D with points sampled from a rectangle.

    Sample points on a 3D rectangular planar patch with boundary points given
    by the arguments.

    Args:
        depth_mm (float): depth of plane in user coordinate system.

        azimuth_samples_deg (iterable): iterable with N azimuth angles in
        degrees at which to sample points.

        elevation_range_deg (iterable): iterable with M elevation angles in
        degrees at which to sample points.

    Returns:
        list: (M, N) list of (3,) torch.Tensor corresponding to 3D points at
        given azimuths, elevations, and depth.
    """
    translation_toUser_fromPlane = torch.tensor((0.0, 0.0, depth_mm))
    transform_toUser_fromPlane = kiruna.core.SE3(translation_toUser_fromPlane)
    plane = \
        kiruna.primitives.Plane(
            error_simulator.user, transform_toUser_fromPlane
        )
    origin_inUser = torch.zeros(3)

    all_grid_points_inUser = list()
    for elevation_deg in elevation_samples_deg:
        rotation_elevation = kiruna.core.rotation_around_x(elevation_deg)
        grid_points_inUser_row = list()

        for azimuth_deg in azimuth_samples_deg:
            rotation_azimuth = kiruna.core.rotation_around_y(-azimuth_deg)
            rotation = rotation_azimuth @ rotation_elevation
            direction_inUser = rotation @ torch.tensor((0.0, 0.0, 1.0))
            grid_point_inUser = \
                plane.intersect_from_origin_and_direction_inOther(
                    error_simulator.user, origin_inUser, direction_inUser
                )
            grid_points_inUser_row.append(grid_point_inUser)

        all_grid_points_inUser.append(grid_points_inUser_row)

    # Clean up the pose graph.
    error_simulator.user.remove_child(plane)

    return all_grid_points_inUser


def compute_errors_at_grid(error_simulator, grid):
    """
    Compute position and vergence errors for user setup and points in grid.

    For the user and camera setup represented by error_simulator, compute the
    perceived 3D position error and vertical disparity for points in the grid.

    Args:
        error_simulator (StereoPerceptionErrors): experimental setup.

        grid (list): (M, N) list of (3,) tensor corresponding to an M x N grid
        of 3D points for which the depth perception and vergence errors is to
        be computed.
    """
    observed_points_inUser_at_grid = list()
    position_errors_at_grid = list()
    disparity_mrad_at_grid = list()
    nrows = len(grid)
    for row in range(nrows):
        observed_points_inUser_row = list()
        position_errors_at_grid_row = list()
        disparity_mrad_at_grid_row = list()
        ncols = len(grid[row])
        for col in range(ncols):
            # Best case scenario: users are verging at true point.
            point_inUser = grid[row][col]

            # Compute and store the observed point and the eyes disparity.
            observed_point_inUser, disparity_mrad = \
                error_simulator.run_rendering_pipeline(point_inUser)
            observed_points_inUser_row.append(observed_point_inUser)
            disparity_mrad_at_grid_row.append(disparity_mrad)

        # Store all data.
        observed_points_inUser_at_grid.append(observed_points_inUser_row)
        position_errors_at_grid.append(position_errors_at_grid_row)
        disparity_mrad_at_grid.append(disparity_mrad_at_grid_row)

    return observed_points_inUser_at_grid, disparity_mrad_at_grid
