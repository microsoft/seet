"""plane.py

User-defined package defining a plane primitives.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import torch


class Plane(core.Node):
    """Plane.

    Represents a plane, which has an origin and normal.
    """

    def __init__(
        self,
        parent,
        transform_toParent_fromSelf,
        name="",
        requires_grad=False
    ):
        """Initialize object.

        In its own coordinate system, the origin is (0, 0, 0), and the
        plane is the z=0 plane.

        Args:
            parent (Node): Node object representing the parent of the plane in
            the pose graph.

            transform_toParent_fromSelf (SE3): transformation from the plane
            coordinate system to the coordinate system of the plane's parent.

            name (string, optional): Name of the plane. Defaults to "".
        """
        super().__init__(
            parent,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

        self.origin = torch.zeros(3)
        self.normal = torch.tensor([0.0, 0, 1])
        self.orthonormal = torch.eye(3, 2)

    @classmethod
    def create_from_origin_and_normal_inParent(
        cls, parent, origin_inParent, normal_inParent, name=""
    ):
        """
        Creates a plane object in parent from the origin and normal
        direction.

        Args:
            parent (core.Node): parent node of plane object.

            origin_inParent (torch.Tensor): (3,) tensor corresponding to origin
            of plane in parent's coordinate system.

            normal_inParent (torch.Tensor): (3,) tensor corresponding to normal
            of plane in parent's coordinate system.

            name (str, optional): _description_. Defaults to "".

        Returns:
            Plane: plane object attached to parent in pose graph.
        """
        norm = torch.norm(normal_inParent)
        if torch.abs(norm - 1) > core.TEPS:
            dir_normalized = normal_inParent / norm
        else:
            dir_normalized = normal_inParent

        z_axis = torch.tensor([0.0, 0, 1])
        R = core.rotation_matrix_from_u_to_v(z_axis, dir_normalized)
        if R is None:
            R = torch.eye(3)

        transform_toParent_fromNode = \
            core.SE3(
                torch.hstack((R, origin_inParent.reshape(3, 1)))
            )

        return Plane(parent, transform_toParent_fromNode, name=name)

    @classmethod
    def create_from_homogeneous_coordinates_inParent(
        cls, parent, plane_inParent_h
    ):
        """
        Create plane from vector [n.T, -d] representing plane in homogeneous
        coordinates in the coordinate system of parent.

        Args:
            parent (core.node): parent node of plane
            plane_inParent_h (torch.Tensor): Homogeneous coordinates [n.T, -d]
            of plane in parent core. n is the normal direction, -d is the
            normal distance to the origin.

        Returns:
            plane: plane object.
        """
        scaled_direction_inParent = plane_inParent_h[:3]
        scale = torch.linalg.norm(scaled_direction_inParent)
        sign = torch.sign(plane_inParent_h[-1])
        normal_inParent = -sign * scaled_direction_inParent / scale
        distance_to_origin = sign * plane_inParent_h[-1] / scale
        origin_inParent = distance_to_origin * normal_inParent

        return cls.create_from_origin_and_normal_inParent(
            parent, origin_inParent, normal_inParent
        )

    @staticmethod
    def normalize_homogeneous_coordinates(homogeneous_coordinates):
        """normalize_homogeneous_coordinates.

        Normalize the homogeneous coordinates (dir, scale) of the plane
        given as input to produce coordinates (n, -d), where d is non-negative
        and n is a unit vector.

        Args:
            homogeneous_coordinates (torch.Tensor): (4,) or (4, 1) torch tensor
            representing the homogeneous coordinates of a plane.

        Returns:
            torch.Tensor: (4,) or (4, 1) torch Tensor representing the
            normalized coordinates of the input.
        """
        norm = torch.linalg.norm(homogeneous_coordinates[:3])
        sign = torch.sign(homogeneous_coordinates[-1])

        return -sign * homogeneous_coordinates / norm

    def get_origin_and_normal_inOther(self, other):
        """get_origin_and_normal_inOther.

        Get coordinates of origin and normal to the plane in the coordinate
        system of the input core.

        Args:
            other (Node): Node in whose coordinate system the origin and normal
            to the plane are to be returned.

        Returns:
            tuple: tuple holding two (3,) torch tensors corresponding to the
            coordinates of the plane's origin and normal in the coordinate
            system of Node other.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)
        origin_inOther = transform_toOther_fromSelf.transform(self.origin)
        normal_inOther = \
            transform_toOther_fromSelf.rotation.transform(self.normal)

        return origin_inOther, normal_inOther

    def get_orthonormal_inOther(self, other):
        """get_orthonormal_inOther.

        Get basis of orthonormal coordinate system for points on plane,
        with the basis represented in the coordinate system of the input core.

        Args:
            other (Node): Node object in whose coordinate system the
            orthonormal basis of points on the plane is sought.

        Returns:
            torch.Tensor: (3, 2) torch tensor representing orthonormal basis
            on the plane's own coordinate system, represented in the coordinate
            system of other.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)

        return transform_toOther_fromSelf.rotation.transform(self.orthonormal)

    def get_origin_and_normal_inParent(self):
        """get_origin_and_normal_inParent.

        Get coordinates of origin and normal to the plane in the coordinate
        system of the plane's parent core.

        Returns:
            tuple: tuple holding two (3,) torch tensors corresponding to the
            coordinates of the plane's origin and normal in the plane's
            coordinate system.
        """
        return self.get_origin_and_normal_inOther(self.parent)

    def get_homogeneous_coordinates_inOther(self, other):
        """get_homogeneous_coordinates_inOther.

        Get the homogeneous coordinates of the plane in the coordinate
        system of other.

        Args:
            other (Node): Node in whose coordinate system the origin and normal
            to the plane are to be returned.

        Returns:
            torch.Tensor: (4,) torch tensor corresponding to the homogeneous
            coordinates of the plane in the coordinate system of other.
        """
        origin_inOther, normal_inOther = \
            self.get_origin_and_normal_inOther(other)
        signed_distance_to_origin = origin_inOther @ normal_inOther

        return \
            torch.sign(signed_distance_to_origin) * \
            torch.hstack((normal_inOther, -signed_distance_to_origin))

    def get_homogeneous_coordinates_inParent(self):
        """get_homogeneous_coordinates_inParent.

        Get the homogeneous coordinates of plane in the coordinate system
        of its parent.

        Returns:
            torch.Tensor: (4,) torch tensor corresponding to the homogeneous
            coordinates of the plane in the coordinate system of its parent.
        """
        return self.get_homogeneous_coordinates_inOther(self.parent)

    def compute_signed_distance_to_point_inPlane(self, point_inPlane):
        """compute_signed_distance_to_point_inPlane.

        Computes the signed distance from the plane to a point in the
        plane's coordinate system.

        In the coordinate system of the plane, the signed distance is the z
        coordinate of a point. So a point "in front" of the plane (z > 0) will
        have a positive distance to the plane. A point "behind" the plane
        (z < 0) will have a negative distance to the plane. A point on the
        plane will have zero distance to the plane.

        Args:
            point_inPlane (torch.Tensor): (3,) or (3,1) torch.Tensor
            representing the coordinates of the input point in the coordinate
            system of the plane.

        Returns:
            torch.Tensor: (1,) or (1,1) torch.Tensor corresponding to the
            signed distance from the point to the plane.
        """
        # Distance is z coordinate.
        return point_inPlane[-1]

    def compute_signed_distance_to_point_inOther(self, point_inOther, other):
        """compute_signed_distance_to_point_inOther.

        Computes the signed distance from the plane to a point in the
        coordinate system of the other core.

        In the coordinate system of the plane, the signed distance is the z
        coordinate of a point. So a point "in front" of the plane (z > 0) will
        have a positive distance to the plane. A point "behind" the plane
        (z < 0) will have a negative distance to the plane. A point on the
        plane will have zero distance to the plane.

        Args:
            point_inOther (torch.Tensor): (3,) or (3,1) torch.Tensor
            representing the coordinates of the input point in the coordinate
            system of the other core.

            other (Node): node in which coordinates of point_inOther are
            represented.

        Returns:
            torch.Tensor: (1,) or (1,1) torch.Tensor corresponding to the
            signed distance from the point to the plane.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)
        point_inPlane = \
            transform_toOther_fromSelf.inverse_transform(point_inOther)

        return self.compute_signed_distance_to_point_inPlane(point_inPlane)

    def compute_signed_distance_to_point_inParent(self, point_inParent):
        """compute_signed_distance_to_point_inParent.

        Computes the signed distance from the plane to a point in the
        coordinate system of the plane's parent.

        In the coordinate system of the plane, the signed distance is the z
        coordinate of a point. So a point "in front" of the plane (z > 0) will
        have a positive distance to the plane. A point "behind" the plane
        (z < 0) will have a negative distance to the plane. A point on the
        plane will have zero distance to the plane.

        Args:
            point_inParent (torch.Tensor): (3,) or (3,1) torch.Tensor
            representing the coordinates of the input point in the coordinate
            system of the plane's parent.

        Returns:
            torch.Tensor: (1,) or (1,1) torch.Tensor corresponding to the
            signed distance from the point to the plane.
        """
        point_inPlane = \
            self.transform_toParent_fromSelf.inverse_transform(point_inParent)

        return self.compute_signed_distance_to_point_inPlane(point_inPlane)

    def is_point_on_plane_inPlane(self, point_inPlane, tol=core.TEPS * 100):
        """is_point_on_plane_inPlane.

        Returns True if point is on plane, within tolerance.

        Args:
            point_inPlane (torch.Tensor): (3,) tensor represeting point in the
            coordinate system of the plane.

            tol (float or torch.Tensor, optional): Proximity tolerance in
            determining whether point is on the plane. Defaults to
            core.EPS*100.

        Returns:
            bool: True if absolute distance from input point to plane is
            smaller than tolerance, False otherwise.
        """
        algebraic_distance = \
            self.compute_signed_distance_to_point_inPlane(point_inPlane)

        return torch.abs(algebraic_distance) < tol

    def is_point_on_plane_inOther(
        self, point_inOther, other, tol=core.TEPS * 100
    ):
        """
        Returns True if point is on plane, within tolerance.

        Args:
            point_inOther (torch.Tensor): (3,) tensor representing coordinates
            of point in the coordinate system of other core.

            other (Node): node in the pose graph in which the coordinates of
            the input point are represented.

            tol (float or torch.Tensor, optional): Proximity tolerance in
            determining whether point is on the plane. Defaults to
            core.EPS*100.

        Returns:
            bool: True if absolute distance from input point to plane is
            smaller than tolerance, False otherwise.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)
        point_inPlane = \
            transform_toOther_fromSelf.inverse_transform(point_inOther)

        return self.is_point_on_plane_inPlane(point_inPlane, tol=tol)

    def is_point_on_plane_inParent(
        self, point_inParent, tol=core.TEPS * 100
    ):
        """
        Returns True if point is on plane, within tolerance.

        Args:
            point_inParent (torch.Tensor): (3,) tensor representing coordinates
            of point in the coordinate system of other plane's parent.

            tol (float or torch.Tensor, optional): Proximity tolerance in
            determining whether point is on the plane. Defaults to
            core.EPS*100.

        Returns:
            bool: True if absolute distance from input point to plane is
            smaller than tolerance, False otherwise.
        """
        point_inPlane = \
            self.transform_toParent_fromSelf.inverse_transform(point_inParent)

        return self.is_point_on_plane_inPlane(point_inPlane, tol=tol)

    def intersect_from_origin_and_direction_inPlane(
        self, origin_inPlane, direction_inPlane
    ):
        """
        Intersect a ray with the plane. The ray is described by an origin
        and a direction in the coordinate system of the plane.

        Args:
            origin_inPlane (torch.Tensor): (3,) tensor corresponding to
            coordinates of origin of ray in coordinate system of the plane.

            direction_inPlane (torch.Tensor): (3,) tensor corresponding to
            coordinates of ray direction in coordinate system of the plane.

        Returns:
            torch.Tensor: (3,) tensor corresponding to intersection of ray with
            plane, in plane's coordinate system.
        """
        # In plane's own coordinate system, plane is the xy plane, i.e., z = 0.
        # Therefore the intersection is the point p(t) = origin + t*direction
        # with third component equal to zero.
        t = -origin_inPlane[-1] / direction_inPlane[-1]  # Test before?

        return origin_inPlane + t * direction_inPlane

    def intersect_from_origin_and_direction_inParent(
        self, origin_inParent, direction_inParent
    ):
        """
        Intersect a ray with the plane. The ray is described by an origin
        and a direction in the coordinate system of the plane's parent.

        Args:
            origin_inParent (torch.Tensor): (3,) tensor corresponding to
            coordinates of origin of ray in coordinate system of the plane's
            parent core.

            direction_inParent (torch.Tensor): (3,) tensor corresponding to
            coordinates of ray direction in coordinate system of the plane's
            parent core.
        """
        origin_inPlane = \
            self.transform_toParent_fromSelf.inverse_transform(origin_inParent)
        direction_inPlane = \
            self.transform_toParent_fromSelf.rotation.inverse_transform(
                direction_inParent
            )
        intersection_inPlane = \
            self.intersect_from_origin_and_direction_inPlane(
                origin_inPlane, direction_inPlane
            )

        return self.transform_toParent_fromSelf.transform(intersection_inPlane)

    def intersect_from_origin_and_direction_inOther(
        self, other, origin_inOther, direction_inOther
    ):
        """
        Intersect a ray with the plane. The ray is described by an origin
        and a direction in the coordinate system of the input node other.

        Args:
            other (Node): node in the same pose graph as the plane.

            origin_inOther (torch.Tensor): (3,) tensor corresponding to
            coordinates of origin of ray in coordinate system of the node
            other.

            direction_inOther (torch.Tensor): (3,) tensor corresponding to
            coordinates of ray direction in coordinate system of the the node
            other.

        Returns:
            torch.Tensor: (3,) torch tensor representing intersection of ray
            with plane in coordinate system of other.
        """

        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)
        origin_inPlane = \
            transform_toOther_fromSelf.inverse_transform(origin_inOther)
        direction_inPlane = \
            transform_toOther_fromSelf.rotation.inverse_transform(
                direction_inOther
            )
        intersection_inPlane = \
            self.intersect_from_origin_and_direction_inPlane(
                origin_inPlane, direction_inPlane
            )

        return transform_toOther_fromSelf.transform(intersection_inPlane)

    def intersect_ray_inParent(self, input_ray):
        """intersect_ray_inParent.

        Intersect a ray with the plane. The plane and the ray must have the
        same parent.

        Args:
            input_ray (primitives.Ray): ray in the same pose graph as plane.

        Returns:
            torch.Tensor: (3,) torch.Tensor representing intersection of point
            and plane in coordinate system of their common parent.
        """

        # Transform the ray to the coordinate system of the plane.
        ray_origin_inPlane, ray_dir_inPlane = \
            input_ray.get_origin_and_direction_inOther(self)

        if ray_origin_inPlane[2] == 0:
            intersection_inPlane = ray_origin_inPlane
        else:
            if torch.abs(ray_dir_inPlane[2]) < core.TEPS:
                return None  # Ray is parallel with the plane.

            scale = -ray_origin_inPlane[2] / ray_dir_inPlane[2]
            intersection_inPlane = ray_origin_inPlane + scale * ray_dir_inPlane

        return self.transform_toParent_fromSelf.transform(intersection_inPlane)
