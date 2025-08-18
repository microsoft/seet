"""ray.py

User-defined package for ray type.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
import torch


class Ray(core.Node):
    """Ray.

    Represents a ray, which has an origin and a pointing direction.
    In its own coordinate system, the node is (0, 0, 0), pointing in the
    direction (0, 0, 1).
    """

    def __init__(
        self,
        parent,
        transform_toParent_fromSelf,
        scale=core.T1.clone(),
        name="",
        requires_grad=False
    ):
        super().__init__(
            parent,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

        self.scale = scale

        identity = core.SE3.create_identity()

        self.origin = \
            primitives.Point(self, identity, name=f"point in {self.name}")
        self.direction = \
            primitives.Direction(
                self,
                identity,
                scale=self.scale,
                name=f"direction in {self.name}"
            )

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            list: list with required arguments in the order of init.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"scale": self.scale}

        return {**base_kwargs, **this_kwargs}

    @classmethod
    def create_from_origin_and_dir_inParent(
        cls,
        parent,
        origin_inParent,
        dir_inParent,
        name=""
    ):
        """
        Creates a ray object in parent from the origin and pointing direction.
        """
        scale = torch.linalg.norm(dir_inParent)
        normalized_dir_inParent = dir_inParent / scale
        z_axis = torch.tensor([0.0, 0, 1])
        R = \
            core.rotation_matrix_from_u_to_v(
                z_axis, normalized_dir_inParent
            )
        transform_toParent_fromSelf = \
            core.SE3(torch.hstack((R, origin_inParent.view(3, 1))))

        return Ray(parent, transform_toParent_fromSelf, scale=scale, name=name)

    @classmethod
    def create_from_params_inParent(cls, parent, params, name=""):
        """create_from_params_inParent.

        Create a ray from a an origin (x, y, z), angles (theta, phi), and
        scale (s) in parent coordinate system.

        Args:
            parent (Node): parent node of ray, in which parameters are defined.

            params (torch.Tensor): (5,) or (6,) tensor. If length is 6,
            parameters are (x, y, z, theta, phi, s). If length is 5, scale s is
            assumed to be 1.

            name (string, optional): name of Ray object. Defaults to "".

        Returns:
            Ray: Ray object with prescribed parameters attached to the input
            parent node.
        """
        # tensor=(x,y,z,theta,phi), where (x,y,z) is the ray start, and
        # (theta,phi) define spherical polar coordinates for the ray direction.

        #             ^ z
        #             |\
        #             | \
        #             |  \
        #        theta|  /|
        #             |\/ |
        #             |/__|___________\  y
        #          __/ \  |           /
        #       __/\___/\ |
        #    __/    phi  \|
        #  |/_
        # x

        theta = params[3]
        phi = params[4]

        ray_start = params[:3]

        scale = params[5] if len(params) == 6 else core.T1
        ray_dir = \
            torch.hstack(
                (
                    torch.sin(theta) * torch.cos(phi),
                    torch.sin(theta) * torch.sin(phi),
                    torch.cos(theta)
                )
            ) * scale

        return Ray.create_from_origin_and_dir_inParent(
            parent, ray_start, ray_dir, name=name
        )

    @staticmethod
    def intersect_rays_inOther(other, ray_a, ray_b):
        """Intersect two rays.

        This method generates the least-squares intersection of two rays and
        outputs the result as a point in the coordinate system of node other

        Args:
            other (Node): node in which to represent the coordinate system of
            the intersection.

            ray_a (Ray): first input ray.

            ray_b (Ray): second input ray.

        Returns:
            torch.Tensor: (3,) tensor representing the point that is jointly
            closest to both input, represented in the coordinate system of node
            other.
        """
        a_origin_inOther, \
            a_dir_inOther = ray_a.get_origin_and_direction_inOther(other)
        b_origin_inOther, \
            b_dir_inOther = ray_b.get_origin_and_direction_inOther(other)

        # Solve the following problem in x (3D), t1 (scalar), and t2 (scalar):
        #
        # min_{x, t1, t2} |x - (o1 + v1 * t1)|^2 + |x - (o2 + v2 * t2)|^2
        #
        # This yields a linear system of equations of type A @ (x, t1, t2) = b.
        o1 = a_origin_inOther.view((3, 1))
        v1 = a_dir_inOther.view((3, 1))
        v1Tv1 = (v1.T @ v1).view((1, 1))
        o2 = b_origin_inOther.view((3, 1))
        v2 = b_dir_inOther.view((3, 1))
        v2Tv2 = (v2.T @ v2).view((1, 1))
        top = torch.hstack((2 * torch.eye(3), -v1, -v2))
        mid = torch.hstack((v1.T, -v1Tv1, core.T0.view((1, 1))))
        bot = torch.hstack((v2.T, core.T0.view((1, 1)), -v2Tv2))
        A = torch.vstack((top, mid, bot))
        b = torch.vstack((o1 + o2, (v1.T @ o1), (v2.T @ o2)))

        return torch.linalg.solve(A, b).flatten()[:3]

    @staticmethod
    def get_disparity_mrad(ray_a, ray_b):
        """Compute the disparity between rays.

        A vergence point is computed as the least-squares approximation to the
        point closest to the rays. If the vergence point is exact, i.e., the
        rays intersect, the disparity is zero. Otherwise we compute the
        disparity as twice the angle between one of the rays and the ray from
        the origin of that center and the vergence point. This angle is the
        same for either ray.

        Arga:
            ray_a, ray_b: (Ray) rays betwewen which to disparity is computed.

        Returns:
            torch.Tensor: (1,) torch.Tensor corresponding to disparity angle in
            mrad.
        """
        vergence_point_inRayA = Ray.intersect_rays_inOther(ray_a, ray_a, ray_b)
        vergence_point_inRayA = core.normalize(vergence_point_inRayA)
        # In the coordinate system of ray_a, the origin and direction of ray_a
        # itself are (0, 0, 0) and (0, 0, 1).
        sin_angle_rad = \
            torch.sqrt(
                vergence_point_inRayA[1]**2 + vergence_point_inRayA[0]**2
            )

        return torch.arcsin(sin_angle_rad) * 1_000 * 2

    def get_origin_and_direction_inParent(self):
        """get_origin_and_direction_inParent.

        Returns (origin, dir) in parent."""
        origin_inParent = self.origin.get_coordinates_inOther(self.parent)
        dir_inParent = self.direction.get_components_inOther(self.parent)

        return origin_inParent, dir_inParent

    def get_params_inParent(self):
        """get_params_inParent.

        tensor=(x,y,z,theta,phi), where (x,y,z) is the ray start,
        and (theta,phi) define spherical polar coordinates for the ray
        direction.
        """
        ray_origin, ray_dir = self.get_origin_and_direction_inParent()
        theta = torch.acos(ray_dir[2])
        phi = torch.atan2(ray_dir[1], ray_dir[0])

        return torch.hstack((ray_origin, theta, phi, self.direction.scale))

    def get_origin_and_direction_inOther(self, other):
        """get_origin_and_direction_inOther.

        Get origin and direction of ray in an arbitrary coordinate system.

        Args:
            other (Node): Node to which we want to transform the origin
            and direction of the ray.

        Returns:
            torch.Tensor, torch.Tensor: (3,) tensors specifying the origin and
            direction of the ray in the coordinate system of other_node.
        """
        origin_inOther = self.origin.get_coordinates_inOther(other)
        dir_inOther = self.direction.get_components_inOther(other)

        return origin_inOther, dir_inOther

    def project_to_ray_inRay(self, point_inRay):
        """project_to_ray_inRay.

        Computes in the coordinates of the ray the coordinates of the point
        along the ray closest to the input point.

        Args:
            point_inRay (torch.Tensor): (3,) tensor corresponding to
            coordinates of input point, in coordinate system of ray's parent.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the coordinates of the
            output in the ray's coordinate system.
        """
        return torch.tensor([0.0, 0.0, 1.0]) * point_inRay

    def project_to_ray_inOther(self, point_inOther, other):
        """project_to_ray_inOther.

        Computes in the coordinates of the ray's parent the coordinates of
        the point along the ray closest to the input point.

        Args:
            point_inParent (torch.Tensor): (3,) tensor corresponding to
            coordinates of input point, in coordinate system of other node.

            other (Node): node in which the coordinates of point_inOther are
            represented.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the coordinates of the
            output in the ray's coordinate system.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)
        point_inRay = \
            transform_toOther_fromSelf.inverse_transform(point_inOther)
        projection_inRay = self.project_to_ray_inRay(point_inRay)

        return transform_toOther_fromSelf.transform(projection_inRay)

    def project_to_ray_inParent(self, point_inParent):
        """project_to_ray_inParent.

        Computes in the coordinates of the ray's parent the coordinates of
        the point along the ray closest to the input point.

        Args:
            point_inParent (torch.Tensor): (3,) tensor corresponding to
            coordinates of input point, in coordinate system of ray's parent.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the coordinates of the
            output in the ray's coordinate system.
        """
        # We could have done this:
        #
        # return self.project_to_ray_inOther(point_inParent, self.parent)
        #
        # but that would be inefficient, because this would require the
        # concatenation of several transformations.
        point_inRay = \
            self.transform_toParent_fromSelf.inverse_transform(point_inParent)
        projection_inRay = self.project_to_ray_inRay(point_inRay)

        return self.transform_toParent_fromSelf.transform(projection_inRay)

    def compute_distance_to_point_inRay(self, point_inRay):
        """compute_distance_to_point_inRay.

        Compute the orthogonal distance from the ray to a Point object.

        Args:
            point_inRay (torch.Tensor): (3,) tensor corresponding to the
            coordinates of input point, represented in coordinate system of
            ray.

        Returns:
            torch.float: orthogonal distance between input point and ray.
        """
        return torch.linalg.norm(point_inRay[:2])

    def compute_distance_to_point_inOther(self, point_inOther, other):
        """compute_distance_to_point_inOther.

        Compute the orthogonal distance form the ray to a point represented
        in the coordinates of another node.

        Args:
            point_inOther (torch.Tensor): (3,) tensor corresponding to
            coordinates of input point represented in the coordinate system of
            other node.

            other (Node): node in which coordinates of point_inOther are
            represented.

        Returns:
            torch.float: orthogonal distance between input point and ray.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)
        point_inRay = \
            transform_toOther_fromSelf.inverse_transform(point_inOther)

        return self.compute_distance_to_point_inRay(point_inRay)

    def compute_distance_to_point_inParent(self, point_inParent):
        """compute_distance_to_point_inParent.

        Compute the orthogonal distance from the ray to a point represented
        in the coordinate system of the ray's parent.

        Args:
            point_inParent (torch.Tensor): (3,) tensor corresponding to
            coordinates of input point represented in the coordinate system of
            the ray's parent.

        Returns:
            torch.float: orthogonal distance between input point and ray.
        """
        # We could have done this:
        #
        # return \
        #  self.compute_distance_to_point_inOther(point_inParent, self.parent),
        #
        # but that would be inefficient, because this would require the
        # concatenation of several transformations.
        point_inRay = \
            self.transform_toParent_fromSelf.inverse_transform(point_inParent)

        return self.compute_distance_to_point_inRay(point_inRay)

    def is_point_on_ray_inRay(self, point_inRay, tol=core.TEPS * 100):
        """is_point_on_ray_inRay.

        Test whether input point in ray's coordinate system is along ray,
        within tolerance.

        Args:
            point_inRay (torch.Tensor): (3,) tensor representing coordinates of
            input point in ray's coordinate system.

            tol (torch.float, optional): distance tolerance below which point
            is considered to be on ray. Defaults to core.TEPS*100.

        Returns:
            bool: True if distance from point to ray is less than tolerance,
            False otherwise.
        """
        distance = self.compute_distance_to_point_inRay(point_inRay)

        return distance < tol

    def is_point_on_ray_inOther(
        self, point_inOther, other, tol=core.TEPS * 100
    ):
        """
        Test whether input point represented in coordinate system of another
        node is along ray, within tolerance.

        Args:
            point_inOther (torch.Tensor): (3,) tensor representing coordinates
            of input point in coordinate system of other node.

            other (Node): node in which coordinates of input point are
            represented.

            tol (torch.float, optional): distance tolerance below which point
            is considered to be on ray. Defaults to core.TEPS*100.

        Returns:
            bool: True if distance from point to ray is less than tolerance,
            False otherwise.
        """
        return \
            self.compute_distance_to_point_inOther(point_inOther, other) < tol

    def is_point_on_ray_inParent(self, point_inParent, tol=core.TEPS * 100):
        """is_point_on_ray_inParent.

        Test whether input point represented in coordinate system of ray's
        parent is along ray, within tolerance.

        Args:
            point_inParent (torch.Tensor): (3,) tensor representing coordinates
            of input point in coordinate system of ray's parent.

            tol (torch.float, optional): distance tolerance below which point
            is considered to be on ray. Defaults to core.TEPS*100.
        """
        return self.compute_distance_to_point_inParent(point_inParent) < tol
