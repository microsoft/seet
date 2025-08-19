"""ellipse_fitting.py

Ellipse-fitting class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
import scipy.optimize as optimize
import torch


class EllipseFitting():
    """EllipseFitting.

    Class for ellipse fitting using minimization of geometric distance.
    """

    def __init__(self, parent_plane, points_inParent2DPlane, std=None):
        """__init__.

        Initialize ellipse-fitting class.

        Args:
            parent_pane (Plane): plane that serves as parent node to the
            ellipse to be fitted to the points.

            points_inParent2DPlane (list of (2,) torch.Tensors): list with
            torch.Tensors corresponding to 2D point to which ellipse should be
            fitted. The points are in the coordinate system of the ellipse's
            parent plane.

            std (None | list of floats ): standard deviation of noise in
            points. If None, noise for all points is assumed to have the same
            (unity) standard deviation. If a list of floats, noise
            heteroscedastic, i.e., noise in each point will have a different
            standard deviation.
        """

        self.parent_plane = parent_plane
        self.points_inParent2DPlane = points_inParent2DPlane

        try:
            iter(std)  # type: ignore
            self.var = std**2  # type: ignore
        except TypeError:
            self.var = torch.ones(len(points_inParent2DPlane))

        # Initial ellipse. The initial center is the centroid of the points.
        center_inParent2DPlane = torch.zeros(2, requires_grad=True)
        for single_point_inParent2DPlane in self.points_inParent2DPlane:
            center_inParent2DPlane = \
                center_inParent2DPlane + single_point_inParent2DPlane

        # Initial value
        self.center_inParent2DPlane = \
            center_inParent2DPlane / len(self.points_inParent2DPlane)
        self.center_inParent2DPlane = \
            self.center_inParent2DPlane.clone().detach()

        # To initialize the axes and angle we could do SVD, but we use
        # something simpler.
        distance_to_center = torch.tensor(0.0)
        for single_point_inParent2DPlane in self.points_inParent2DPlane:
            distance_to_center = \
                distance_to_center + \
                torch.linalg.norm(
                    single_point_inParent2DPlane - self.center_inParent2DPlane
                )
        distance_to_center = \
            distance_to_center / len(self.points_inParent2DPlane)

        # Initial value.
        self.angle_deg = torch.tensor(0.0, requires_grad=True)

        # Initial values.
        self.x_radius = distance_to_center.clone().detach()
        self.y_radius = distance_to_center.clone().detach()

        # Random cost value.
        self.cost = torch.tensor(0.0, requires_grad=True)

        # Random derivative values.
        self.d_cost_d_center = torch.zeros(2, requires_grad=True)
        self.d_cost_d_angle = torch.tensor(0.0, requires_grad=True)
        self.d_cost_d_x_radius = torch.tensor(0.0, requires_grad=True)
        self.d_cost_d_y_radius = torch.tensor(0.0, requires_grad=True)
        self.d_cost_d_parameter = torch.zeros(5, requires_grad=True)

        self._require_grad()

    def _require_grad(self):
        """_require_grad.

        Force gradients of relevant parameters to be required.
        """

        self.center_inParent2DPlane.requires_grad = True
        self.angle_deg.requires_grad = True
        self.x_radius.requires_grad = True
        self.y_radius.requires_grad = True

    def _array_to_torch(self, parameter):
        """_array_to_torch.

        Convert a numpy array to the center, angle, and radii parameters
        that we seek to optimize.

        Args:
            parameter (numpy.array): (5,) numpy array with ellipse parameters.
        """

        # Type specification is required because scipy optimize internal type
        # is float64 regardless of the type of the initial value of the
        # parameter to be optimized.
        self.center_inParent2DPlane = \
            torch.tensor(
                parameter[:2],
                dtype=self.center_inParent2DPlane.dtype,
                requires_grad=True
            )
        self.angle_deg = \
            torch.tensor(
                parameter[2], dtype=self.angle_deg.dtype, requires_grad=True
            )
        self.x_radius = \
            torch.tensor(
                parameter[3], dtype=self.x_radius.dtype, requires_grad=True
            )
        self.y_radius = \
            torch.tensor(
                parameter[4], dtype=self.y_radius.dtype, requires_grad=True
            )

    def _torch_to_array(self):
        """_torch_to_array.

        Convert ellipse parameters to numpy array for optimization.

        Returns:
            numpy.array: (5,) numpy array with ellipse parameters.
        """
        torch_parameter = \
            torch.hstack(
                (
                    self.center_inParent2DPlane,
                    self.angle_deg,
                    self.x_radius,
                    self.y_radius
                )
            )

        return torch_parameter.clone().detach().numpy()

    def initialize(self, method="algebraic"):
        """initialize.

        Initialize ellipse parameters. Two methods are available:
        'algebraic,' and 'svd'. Algebraic performs closed-form minimization of
        algebraic distance. 'svd' finds the centroid and covariance matrix of
        the data.

        Args:
            method (str, optional): initialization method. Defaults to
            "algebraic".
        """

        method = method.lower()

        if method == "algebraic":
            self.algebraic_initialization()
        else:
            assert (method == "svd"), "Unknown initialization method."
            self.svd_initialization()

    def svd_initialization(self):
        """svd_initialization.

        Initialization of ellipse parameters through svd.

        Returns:
            (torch.Tensor, torch.Tensor): eigenvalues and eigenvectors of
            covariance matrix of data.
        """

        with torch.no_grad():
            center_inParent2DPlane = self.points_inParent2DPlane.mean(axis=0)
            cov = torch.cov(self.points_inParent2DPlane.T)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            x_radius, y_radius = \
                torch.sqrt(eigenvalues) * torch.sqrt(torch.tensor(2.0))
            angle_rad = torch.atan2(eigenvectors[0, 1], eigenvectors[0, 0])

        self.center_inParent2DPlane = center_inParent2DPlane.clone().detach()
        self.angle_deg = core.rad_to_deg(angle_rad.clone().detach())
        self.x_radius = x_radius.clone().detach()
        self.y_radius = y_radius.clone().detach()

        self._require_grad()

        return eigenvalues, eigenvectors

    def algebraic_initialization(self):
        """algebraic_initialization.

        Initialization of ellipse parameters through minimization of
        algebraic distances.
        """

        # Setup linear system.
        A = torch.zeros((0, 9))
        for datum in self.points_inParent2DPlane:
            h_datum = core.homogenize(datum)
            A = torch.vstack((A, torch.outer(h_datum, h_datum).view(1, 9)))

        # Get ellipse "C" matrix.
        U, d, Vt = torch.linalg.svd(A)
        C = Vt[-1, :].reshape((3, 3))
        C = C + C.T

        # Get center.
        infinity_x = torch.tensor([1.0, 0.0, 0.0])
        infinity_y = torch.tensor([0.0, 1.0, 0.0])
        homogeneous_center = torch.cross(C @ infinity_x, C @ infinity_y)
        center = core.dehomogenize(homogeneous_center)

        # Get translated C matrix.
        translation = \
            torch.vstack(
                (
                    torch.hstack((torch.eye(2), center.view((2, 1)))),
                    torch.tensor([0.0, 0.0, 1.0]).view((1, 3))
                )
            )
        C_ = translation.T @ C @ translation
        C_ = -C_ / C_[-1, -1]

        # Get axes and angle.
        eigenvalues, eigenvectors = torch.linalg.eigh(C_[:2, :2])
        if eigenvalues[0] <= 0 or eigenvalues[1] <= 0:
            # Give up. Use svd initialization instead.
            return

        self.center_inParent2DPlane = center
        angle_rad = torch.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
        self.angle_deg = core.rad_to_deg(angle_rad)
        self.x_axis = 1 / torch.sqrt(eigenvalues[0])
        self.y_axis = 1 / torch.sqrt(eigenvalues[1])

        self._require_grad()

    def set_initial_values(
        self, center_inParent2DPlane, angle_deg, x_radius, y_radius
    ):
        """
        Manually set initial values for optimization

        Args:
            center_inParent2DPlane (torch.Tensor): (2,) tensor representing
            coordinates of initial ellipse center in the coordinate system of
            the ellipse's parent plane node.

            angle_deg (torch.Tensor): angle of x-axis of initial ellipse with
            respect to x-axis of the ellipse's parent plane node.

            x_radius (torch.Tensor): length of semi x-axis of ellipse.

            y_radius (torch.Tensor): length of semi y-axis of ellipse.
        """
        self.center_inParent2DPlane = center_inParent2DPlane
        self.center_inParent2DPlane.requires_grad = True

        self.angle_deg = angle_deg
        self.angle_deg.requires_grad = True

        self.x_radius = x_radius
        self.x_radius.requires_grad = True

        self.y_radius = y_radius
        self.y_radius.requires_grad = True

    def _cost_function(self, create_graph=False):
        """_cost_function.

        Internal-use cost function to be optimized. Uses torch tensors, and
        propagates derivatives.

        Args:
            create_graph (bool, optional): flag indicating whether the
            computational graph for the derivatives is created. If higher-order
            derivatives are required, this should be set to True. Defaults to
            False.
        """

        # This adds an ellipse to the pose graph. Remember to remove it
        # afterwards!
        ellipse = \
            primitives.Ellipse.create_from_origin_angle_and_axes_inPlane(
                self.parent_plane,
                self.center_inParent2DPlane,
                self.angle_deg,
                self.x_radius,
                self.y_radius
            )

        T_toEllipse_fromPlane = \
            ellipse.get_transform_toParent_fromSelf().create_inverse()
        cost = torch.tensor(0.0, requires_grad=True)
        for single_point_inParent2DPlane in self.points_inParent2DPlane:
            # Transform the point to the ellipse's coordinate system.
            flat_point_inParent2DPlane = \
                single_point_inParent2DPlane.flatten()
            point_inParent3DPlane = \
                torch.hstack(
                    (flat_point_inParent2DPlane, torch.tensor(0.0))
                )
            point_inEllipse = \
                T_toEllipse_fromPlane.transform(point_inParent3DPlane)

            # Find the closest point on the ellipse.
            angle_rad = \
                ellipse.compute_angle_of_closest_point(point_inEllipse)
            closest_point_inEllipse = \
                ellipse.get_points_at_angles_inEllipse(angle_rad)

            # Compute the squared distances between the point and the
            # closest point on the ellipse.
            difference = point_inEllipse - closest_point_inEllipse

            cost = cost + difference @ difference

        # Remove ellipse from pose graph.
        self.parent_plane.remove_child(ellipse)

        # Result is differentiable with respect to the relevant parameters.
        self.cost = cost

        # Compute required derivatives.
        self.d_cost_d_center = \
            core.compute_auto_jacobian_from_tensors(
                self.cost,
                self.center_inParent2DPlane,
                create_graph=create_graph
            )
        self.d_cost_d_angle = \
            core.compute_auto_jacobian_from_tensors(
                self.cost, self.angle_deg, create_graph=create_graph
            )
        self.d_cost_d_x_radius = \
            core.compute_auto_jacobian_from_tensors(
                self.cost, self.x_radius, create_graph=create_graph
            )
        self.d_cost_d_y_radius = \
            core.compute_auto_jacobian_from_tensors(
                self.cost, self.y_radius, create_graph=create_graph
            )

        self.d_cost_d_parameter = \
            torch.hstack(
                (
                    self.d_cost_d_center,
                    self.d_cost_d_angle,
                    self.d_cost_d_x_radius,
                    self.d_cost_d_y_radius
                )
            )

    def cost_function(self, parameter):
        """cost_function.

        External use cost function to be optimized.

        Args:
            parameter (numpy.array): (5,) array with ellipse parameters.
        """

        # Set the parameters of the ellipse as torch tensors. This updates the
        # parameters
        #
        # self.center_inParent2DPlane,
        # self.angle_deg,
        # self.x_radius, and
        # self.y_radius.
        self._array_to_torch(parameter)

        # Compute cost with derivatives. This updates the parameters
        #
        # self.cost,
        # self.d_cost_d_center,
        # self.d_cost_d_angle,
        # self.d_cost_d_x_radius,
        # self.d_cost_d_y_radius, and
        # self.d_cost_d_parameter
        self._cost_function()

        return self.cost.clone().detach().item()

    def jacobian_cost_function(self, parameter):
        """jacobian_cost_function.

        External use Jacobian of cost function to be optimized. The input
        parameter is a dummy variable to match the signature of the cost
        function.

        Args:
            parameter (numpy.array): (5,) array with ellipse parameters. This
            is a dummy variable, the Jacobian has already been computed during
            computation of the cost function.
        """

        return self.d_cost_d_parameter.clone().detach().numpy()

    def fit(self, options={"disp": False}):
        """fit.

        Actual ellipse fitting.
        """

        # Optimize parameters.
        initial_parameter = self._torch_to_array()
        optimization_result = \
            optimize.minimize(
                self.cost_function,
                initial_parameter,
                method="BFGS",
                jac=self.jacobian_cost_function,
                options=options
            )
        if not optimization_result.success:
            print(optimization_result)
            raise Exception("Ellipse fitting failed.")

        # Create ellipse with optimized parameters.
        self._array_to_torch(optimization_result.x)
        ellipse = \
            primitives.Ellipse.create_from_origin_angle_and_axes_inPlane(
                self.parent_plane,
                self.center_inParent2DPlane,
                self.angle_deg,
                self.x_radius,
                self.y_radius
            )

        return ellipse
