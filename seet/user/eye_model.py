"""eye_model.py

Eye model class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com) and Chris Aholt"


import json
import seet.core as core
import seet.primitives as primitives
from seet.user import cornea_model
from seet.user import eyelids_model
from seet.user import limbus_model
from seet.user import pupil_model
from seet.user import user_configs
import os
import torch


class EyeModel(core.Node):
    """EyeModel.

    Object representing an eye in 3D. An eye is a node in the pose graph
    consisting of an ellipsoidal cornea, a circular pupil, and a circular
    limbus. Typically, the cornea, pupil, and limbus are anchored as children
    to the a Point node in a UserModel object, corresponding to the eye's
    rotation center.

    The origin of the eye coordinate system is the rotation center of the eye.
    The z axis corresponds to the eye's optical axis, and the x and y axes are
    horizontal (positive is user's right to left) and vertical (positive is
    user's feet to head), respectively.

    Typically, the parent of the eye node will be a UserModel object.
    """

    def __init__(
        self,
        eye_rotation_center,
        transform_toParent_fromSelf,
        name="",
        parameter_file_name=os.path.join(
            user_configs.USER_DIR, r"default_user/default_left_eye.json"
        ),
        requires_grad=None
    ):
        """
        Initialize EyeModel object.

        An EyeModel object is a node in the pose graph. Parameters of the eye
        model are defined either through a json parameter file or through a
        parameter dictionary. If a parameter dictionary is available, it takes
        precedence over the parameter file.

        Args:
            eye_rotation_center (Node, optional): Node corresponding to parent
            of the eye node, typically a UserModel object. Defaults to None.

            transform_toParent_fromSelf (core.SE3): transformation from eye
            coordinate system to the coordinate system of the rotation center.
            It should be the identity transform, but we keep it flexible.

            name (str, optional): name of object.

            parameter_file_name (str, optional): json file containing
            parameters of eye model. Defaults to
            "default_user/default_left_eye.json".

            requires_grad (bool): flag indicating if parameters of eye model
            should be differentiable.

        Returns:
            EyeModel: EyeModel object.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            eye_parameters = json.load(parameter_file_stream)

        if requires_grad is None:
            if "requires grad" in eye_parameters.keys() and \
                    eye_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        # Name the eye model.
        if "name" in eye_parameters.keys():
            name = eye_parameters["name"]
        else:
            name = name

        super().__init__(
            eye_rotation_center,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

        # Easy reference to rotation center from eye.
        self.rotation_center = eye_rotation_center

        #######################################################################
        # Cornea.
        key = "distance from rotation center to cornea center"
        self.distance_from_rotation_center_to_cornea_center = \
            z = torch.tensor(eye_parameters[key], requires_grad=requires_grad)
        z_translation = core.translation_in_z(z)
        transform_toRotationCenter_fromCornea = core.SE3(z_translation)
        self.cornea = \
            cornea_model.CorneaModel(
                self,
                transform_toRotationCenter_fromCornea,
                parameter_file_name=parameter_file_name,
                requires_grad=requires_grad
            )

        #######################################################################
        # Eyelids.
        key = "distance from rotation center to eyelids axis"
        z = torch.tensor(eye_parameters[key], requires_grad=requires_grad)
        z_translation = core.translation_in_z(z)
        transform_toRotationCenter_fromEyelids = core.SE3(z_translation)
        self.eyelids = \
            eyelids_model.EyelidsModel(
                self,
                transform_toRotationCenter_fromEyelids,
                parameter_file_name=parameter_file_name,
                requires_grad=requires_grad
            )

        # Pupil and limbus are not children of the eye, but rather of their own
        # planes. Those planes, in turn are indeed children of the eye.
        #######################################################################
        # Pupil.
        key = "distance from rotation center to pupil plane"
        self.distance_from_rotation_center_to_pupil_plane = \
            z = torch.tensor(eye_parameters[key], requires_grad=requires_grad)
        z_translation = core.translation_in_z(z)
        transform_toRotationCenter_fromPupilPlane = core.SE3(z_translation)
        self.pupil_plane = \
            primitives.Plane(self, transform_toRotationCenter_fromPupilPlane)
        self.pupil = \
            pupil_model.PupilModel(
                self.pupil_plane,
                core.SE3.create_identity(),
                parameter_file_name=parameter_file_name,
                requires_grad=requires_grad
            )

        #######################################################################
        # Limbus.
        key = "distance from rotation center to limbus plane"
        self.distance_from_rotation_center_to_limbus_plane = \
            z = torch.tensor(eye_parameters[key], requires_grad=requires_grad)
        z_translation = core.translation_in_z(z)
        transform_toRotationCenter_fromLimbusPlane = core.SE3(z_translation)
        self.limbus_plane = \
            primitives.Plane(self, transform_toRotationCenter_fromLimbusPlane)
        self.limbus = \
            limbus_model.LimbusModel(
                self.limbus_plane,
                core.SE3.create_identity(requires_grad=requires_grad),
                parameter_file_name=parameter_file_name,
                requires_grad=requires_grad
            )

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary with keyword arguments.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}

    @staticmethod
    def create_secrets_eye_file(eye_blob, requires_grad=False):
        """create_secrets_eye_file.

        Read user calibration blob for a single eye and generate secrets
        dictionary for eye model.

        Args:
            eye_blob (str or dict): path and name of file for user calibration
            of a single eye. May also be a dictionary with the eye parameters.

            requires_grad (bool, optional): whether parameters of eye model are
            differentiable. Defaults to False.

        Returns:
            dict: dictionary with parameters of eye model in secrets format.

            SE3: transformation to camera coordinates from eye coordinates.
        """

        with core.Node.open(eye_blob) as eye_stream:
            eye_data = json.load(eye_stream)

        eye_model_dict = eye_data["EyeModel"]

        # Transformation to camera from eye coordinates.
        translation_vector_toCamera_fromEye = \
            torch.tensor(
                eye_model_dict["rotationCenterInETCameraFrame"]["v"],
                requires_grad=requires_grad
            ).view((3, 1))
        rotation_matrix_toEye_fromCamera = \
            torch.tensor(
                eye_data["RotationCam2Eye"]["v"],
                requires_grad=requires_grad
            ).view((3, 3))

        transform_toCamera_fromEye = \
            core.SE3(
                torch.hstack(
                    (
                        rotation_matrix_toEye_fromCamera.T,  # Needs transpose.
                        translation_vector_toCamera_fromEye
                    )
                )
            )

        # Overall eye.
        distance_from_rotation_center_to_cornea_center = \
            eye_model_dict["rotationCenterToCorneaCenterDistance"]
        distance_from_cornea_center_to_pupil_plane = \
            eye_model_dict["corneaCenterToPupilPlaneDistance"]
        distance_from_rotation_center_to_limbus_plane = \
            distance_from_rotation_center_to_cornea_center + \
            eye_model_dict["limbusCenterInQuadricFrame"]["v"][-1]

        secrets_eye_dict = dict()
        secrets_eye_dict["name"] = "Created from " + eye_blob
        secrets_eye_dict["requires grad"] = requires_grad
        secrets_eye_dict["distance from rotation center to cornea center"] = \
            distance_from_rotation_center_to_cornea_center
        secrets_eye_dict["distance from rotation center to pupil plane"] = \
            distance_from_rotation_center_to_cornea_center + \
            distance_from_cornea_center_to_pupil_plane
        secrets_eye_dict["distance from rotation center to limbus plane"] = \
            distance_from_rotation_center_to_limbus_plane
        secrets_eye_dict["distance from rotation center to eyelids axis"] = 0.0

        # Cornea
        secrets_cornea_dict = dict()
        secrets_cornea_dict["name"] = secrets_eye_dict["name"]
        secrets_cornea_dict["requires grad"] = requires_grad

        shape_matrix = eye_model_dict["corneaModel"]["corneaQuadricShape"]["v"]
        lengths_semi_axes = \
            1 / torch.sqrt(torch.diag(torch.tensor(shape_matrix).view((3, 3))))
        # We compute the radii factors from the mean curvature at the apex.
        radius_of_curvature = lengths_semi_axes[:2].mean()
        radii_factors = lengths_semi_axes / radius_of_curvature

        secrets_cornea_dict["radius of curvature"] = \
            radius_of_curvature.clone().detach().item()
        secrets_cornea_dict["radii factors"] = \
            radii_factors.clone().detach().tolist()
        secrets_cornea_dict["num level sets"] = 7
        secrets_cornea_dict["num points per level set"] = 30

        # Eyelids. There are no eyelids in the calibration blob, so we create
        # our own.
        with open(
            os.path.join(
                user_configs.USER_DIR, r"default_user/default_left_eye.json"
            )
        ) as left_eye_stream:
            left_eye_dict = json.load(left_eye_stream)

        secrets_eyelids_dict = left_eye_dict["eyelids"]

        # Pupil.
        secrets_pupil_dict = \
            {
                "name": "Created from " + eye_blob,
                "requires grad": requires_grad,
                "radius": eye_data["CalibrationSessionParameters"][
                    "medianPupilRadiusFromCalibration"
                ],
                "num_points": 30
            }

        # Limbus.
        secrets_limbus_dict = \
            {
                "name": "Created from " + eye_blob,
                "requires grad": requires_grad,
                "radius": eye_model_dict["limbusRadius"],
                "num_points": 30
            }

        #######################################################################
        # Put it all together as an eye.
        secrets_eye_dict["cornea"] = secrets_cornea_dict
        secrets_eye_dict["eyelids"] = secrets_eyelids_dict
        secrets_eye_dict["pupil"] = secrets_pupil_dict
        secrets_eye_dict["limbus"] = secrets_limbus_dict

        return secrets_eye_dict, transform_toCamera_fromEye

    @classmethod
    def create_eye_from_real_data(
        cls,
        eye_rotation_center,
        transform_toParent_fromSelf,
        eye_blob,
        requires_grad=False
    ):
        """
        Create an eye model from a user-calibration blob.

        Args:
            eye_rotation_center (Node, optional): Node corresponding to parent
            of the eye node, typically a UserModel object. Defaults to None.

            transform_toParent_fromSelf (core.SE3): transformation from eye
            coordinate system to the coordinate system of the rotation center.
            It should be the identity transform, but we keep it flexible.

            eye_blob (str or dict): path and name of file for user calibration
            of a single eye. May also be a dictionary with the eye parameters.

            requires_grad (bool, optional): whether parameters of eye model are
            differentiable. Defaults to False.
        """

        secrets_eye_blob, _ = \
            EyeModel.create_secrets_eye_file(
                eye_blob, requires_grad=requires_grad
            )

        return \
            cls(
                eye_rotation_center,
                transform_toParent_fromSelf,
                parameter_file_name=secrets_eye_blob,  # type: ignore
                requires_grad=requires_grad
            )

    def get_cornea_apex_inOther(self, other):
        """get_cornea_apex_inOther.

        Get the coordinates of the cornea apex in the coordinate system of the
        node other

        Args:
            other (Node): node in which to represent the coordinates of the
            cornea's apex.

        Returns:
            torch.Tensor: (3,) tensor corresponding to coordinates of the
            cornea apex in the coordinate system of the eye's parent.
        """

        apex_inEye = self.cornea.get_apex_inEye()

        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)

        return transform_toOther_fromSelf.transform(apex_inEye)

    def get_gaze_direction_inParent(self):
        """Get the gaze vector in the coordinate system of the eye's parent.

        Returns:
            torch.Tensor: (3,) torch.Tensor with coordinates of a unit gaze
            vector in the coordinate system of the eye's parent.
        """

        gaze_direction_inEye = torch.tensor([0.0, 0.0, 1.0])

        transform_toParent_fromSelf = self.get_transform_toParent_fromSelf()

        return \
            transform_toParent_fromSelf.rotation.transform(
                gaze_direction_inEye
            )

    def get_gaze_direction_inOther(self, other):
        """get_gaze_direction_inOther.

        Get the direction of the eye gaze in the coordinate system of other.

        Args:
            other (Node): node in which to represent the coordinates of the
            gaze direction.

        Returns:
            torch.Tensor: (3,) torch.Tensor with coordinates of a unit gaze
            vector in the coordinate system of other.
        """

        gaze_direction_inParent = self.get_gaze_direction_inParent()

        transform_toOther_fromParent = \
            self.parent.get_transform_toOther_fromSelf(other)

        return \
            transform_toOther_fromParent.rotation.transform(
                gaze_direction_inParent
            )

    def rotate_from_gaze_angles_inParent(
        self, angles_deg, order="direct", move_eyelids=True
    ):
        """
        Rotate by vertical and horizontal angles. First rotation to be applied
        is in the horizontal direction, followed by a rotation in the vertical
        direction, unless value of string parameter order is "reverse"

        Args:
            angles_deg (torch.Tensor or tuple): (2,) tensor or tuple
            representing, in this order, horizontal and vertical rotation
            angles in degrees.

            order (string, optional): if order == "reverse", horizontal
            rotation is applied first, followed by vertical rotation. Otherwise
            (e.g., "direct"), vertical rotation is applied first, followed by
            vertical rotation. Defaults to "direct".

            move_eyelids (bool, optional): if True, lower eyelid is kept fixed,
            i.e., not rotated together with the eye, and upper eyelid rotates
            faster than the eye, so that effect of downward eye rotation is
            that eyelids close a bit. Default is True.
        """

        if not torch.is_tensor(angles_deg):
            angles_deg = torch.tensor(angles_deg)

        # Rightmost transform is applied first. Note that a vertical rotation
        # is a rotation around x, and a horizontal rotation is a rotation
        # around y.
        first = core.rotation_around_y(angles_deg[0])   # horizontal
        second = core.rotation_around_x(angles_deg[1])  # vertical
        if order == "reverse":
            tmp = first
            first = second
            second = tmp.clone()

        rotation_matrix_toParent_fromParent = second @ first

        rotation_toParent_fromParent = \
            core.SO3(rotation_matrix_toParent_fromParent)

        self.rotate_inParent(rotation_toParent_fromParent)

        if move_eyelids:
            # Lower eyelid.
            self.eyelids.rotate_eyelid(-angles_deg[1], upper=False)

            # Upper eyelid. We undo the original eye rotation, then rotate the
            # eyelid even more. The overall effect is
            #
            # rotation_angle = -angle (undoing) + angle*(1 + alpha) (more) =
            #                angle * alpha
            upper_eyelid_rotation_angle = \
                angles_deg[1] * self.eyelids.upper_eyelid_to_eye_rotation_ratio

            self.eyelids.rotate_eyelid(upper_eyelid_rotation_angle, upper=True)

    def unrotate_from_gaze_angles_inParent(
        self, angles_deg, move_eyelids=True
    ):
        """
        Rotate by negative of horizontal and vertical angles. First rotation is
        in horizontal direction, followed by a rotation in the vertical
        direction.

        For an arbitrary EyeModel object eye, the transformations

        eye.rotate_from_gaze_angles_inParent(angles). \
            unrotate_from_gaze_angles_inParent(angles)

        or

        eye.unrotate_from_gaze_angles_inParent(angles). \
            eye.rotate_from_gaze_angles_inParent(angles)

        preserve the eye orientation, module floating-point errors.

        Args:
            angles_deg (torch.Tensor or tuple): (2,) tensor or tuple
            representing horizontal and vertical rotation angles in degrees.

            move_eyelids (bool, optional): if True, lower eyelid is kept fixed,
            i.e., not rotated together with the eye, and upper eyelid rotates
            faster than the eye, so that effect of upward eye rotation is that
            eyelids open a bit. Default is True.
        """

        if not torch.is_tensor(angles_deg):
            angles_deg = torch.tensor(angles_deg)

        self.rotate_from_gaze_angles_inParent(
            -angles_deg, order="reverse", move_eyelids=move_eyelids
        )

    def direct_at_point_inParent(self, point_inParent):
        """Direct the eye at a point in the eye's parent coordinate system.

        Args:
            point_inParent (torch.Tensor): (3,) point with coordinates in the
            eye's parent coordinate system, and towards which the eye will be
            directed.
        """
        # The center of rotation is the eye's parent, and therefore has
        # coordinates (0, 0, 0) in its own coordinate system.
        target_gaze_vector_inParent = core.normalize(point_inParent)

        # This is messy. The way pitch and yaw are computed assumes that the
        # vector is first transformed by a change in pitch, then by a change in
        # yaw. However, eye rotations are computed by a change in yaw then in
        # pitch.
        swap = torch.tensor([[0., 1, 0], [1, 0, 0], [0, 0, 1]])

        # Because of the swap, the output in this case is not yaw_pitch_deg, as
        # the name of the function would suggest.
        pitch_yaw_deg = \
            core.get_yaw_pitch_angles_deg(swap @ target_gaze_vector_inParent)
        target_yaw_pitch_deg = torch.flip(pitch_yaw_deg, [0, ])

        current_gaze_vector_inParent = self.get_gaze_direction_inParent()
        pitch_yaw_deg = \
            core.get_yaw_pitch_angles_deg(swap @ current_gaze_vector_inParent)
        current_yaw_pitch_deg = torch.flip(pitch_yaw_deg, [0, ])

        self.unrotate_from_gaze_angles_inParent(current_yaw_pitch_deg)
        self.rotate_from_gaze_angles_inParent(target_yaw_pitch_deg)

    def direct_at_point_inOther(self, other, point_inOther):
        """Direct the eye at a point in other's coordinate system.

        Args:
            other (Node): Node in whose coordinate system the point towards
            which the eye is to be trained is represented.
            point_inOther (_type_): coordinates of the point towards which the
            eye is to be trained represented in the coordinate system of node
            other.
        """

        point_inParent = \
            other.get_transform_toOther_fromSelf(
                self.parent
            ).inverse_transform(point_inOther)
        self.direct_at_point_inParent(point_inParent)

    def point_is_beyond_limbus_inOther(
        self,
        point_inOther,
        other_node
    ):
        """
        Test whether the input point point, represented in the coordinates of
        other_node, is beyond the limbus. Points beyond the limbus are often
        removed from further processing.

        Args:
            point_inOther (torch.Tensor): (3,) tensor representing point in
            coordinate system of other_node.

            other_node (Node): node in which the coordinates of the input point
            are represented.

        Returns:
            bool: False is point is at or beyond limbus, True otherwise.
        """

        transform_toOther_fromLimbus = \
            self.limbus.get_transform_toOther_fromSelf(other_node)
        point_inLimbus = \
            transform_toOther_fromLimbus.inverse_transform(point_inOther)

        if point_inLimbus[-1] <= 0:
            return True
        else:
            return False

    def apply_eyelid_occlusion_inOther(
        self,
        other_node,
        points_inOther
    ):
        """
        Given a list of points in the coordinate system of a node, generate a
        new list by replacing points in the original list with None if they are
        occluded by the eyelids.

        Args:
            other_node (Node): node in whose coordinate system the points in
            the list are represented.

            points_inOther (list torch.Tensor): list of (3,) torch.Tensors with
            corresponding to points with coordinates in other_node.

        Returns:
            list of torch.Tensor: a new version of points_inOther where the
            points under the eyelids are replaced with None.
        """

        transform_toEyelids_fromOther = \
            other_node.get_transform_toOther_fromSelf(self.eyelids)
        visible_points_inOther = []
        for single_point_inOther in points_inOther:
            if single_point_inOther is not None:
                point_inEyelids = \
                    transform_toEyelids_fromOther.transform(
                        single_point_inOther
                    )
                if not self.eyelids.is_between_inEyelids(point_inEyelids):
                    single_point_inOther = None

            visible_points_inOther = \
                visible_points_inOther + [single_point_inOther, ]

        return visible_points_inOther

    def remove_beyond_limbus_inOther(
        self,
        other_node,
        points_inOther
    ):
        """
        Given a list of points represented in the coordinate system of the node
        other_node, generate a new list by replacing points from the input list
        that are beyond the limbus with None.

        Args:
            other_node (Node): node in whose coordinate system the points in
            the input list are represented.

            points_inOther (list of torch.Tensor): list of (3,) torch Tensors
            corresponding to points in the coordinate system of other_node
            whose position with respect to the limbus is to be tested.

        Returns:
            list of torch.Tensor: a list generated from the input list by
            replacing the points in the input with None if they are beyond the
            limbus.
        """

        transform_toLimbus_fromOther = \
            other_node.get_transform_toOther_fromSelf(self.limbus)
        visible_point_inOther = []
        for single_point_inOther in points_inOther:
            if single_point_inOther is not None:
                point_inLimbus = \
                    transform_toLimbus_fromOther.transform(
                        single_point_inOther
                    )
                if point_inLimbus[-1] <= 0:
                    single_point_inOther = None

            visible_point_inOther = \
                visible_point_inOther + [single_point_inOther, ]

        return visible_point_inOther

    def generate_glint_inCornea(
        self,
        origin_inCornea,
        destination_inCornea,
        apply_eyelids_occlusion=True,
        remove_points_beyond_limbus=True,
    ):
        """generate_glint_inCornea.

        Generate 3D glint location in the coordinate system of the eye,
        corresponding to a reflection from origin to destination.

        The output is a list whose elements are either (3,) torch.Tensors if
        the glint is visible after all visibility criteria are applied, or None
        if the glint is not visible. This format allows for easy matching
        between the glints and their corresponding LEDs.

        Args:
            origin_inCornea (torch.Tensor): (3,) tensor representing origin of
            reflected ray in eye's coordinate system.

            destination_inCornea (torch.Tensor): (3,) tensor representing
            destination of reflected ray in eye's coordinate system.

            apply_eyelids_occlusion (bool, optional): if True, returns None if
            glint is occluded by eyelids. Defaults to True.

            remove_points_beyond_limbus (bool, optional): if True, returns None
            if glint is beyond the limbus. Defaults to True.

        Return:
            torch.Tensor or None: (3,) tensor corresponding to coordinates of
            glint in the coordinate system of the eye's cornea, or None if
            there is no glint or apply_eyelids_occlusion is True and glint is
            occluded by eyelids.
        """

        glint_inCornea = \
            self.cornea.compute_reflection_point_inEllipsoid(
                origin_inCornea,
                destination_inCornea
            )

        if glint_inCornea is not None and remove_points_beyond_limbus:
            if self.point_is_beyond_limbus_inOther(
                glint_inCornea, self.cornea
            ):
                glint_inCornea = None

        if glint_inCornea is not None and apply_eyelids_occlusion:
            transform_toEyelids_fromCornea = \
                self.cornea.get_transform_toOther_fromSelf(self.eyelids)
            glint_inEyelid = \
                transform_toEyelids_fromCornea.transform(glint_inCornea)
            if not self.eyelids.is_between_inEyelids(glint_inEyelid):
                glint_inCornea = None

        return glint_inCornea

    def generate_all_glints_inCornea(
        self,
        origin_inCornea,
        leds_inCornea,
        apply_eyelids_occlusion=True,
        remove_points_beyond_limbus=True,
        return_none=True
    ):
        """generate_all_glints_inCornea.

        Convenience method to generate all glints at once. Essentially a
        for-loop wrapping generate_glint_inCornea.

        Args:
            origin_inCornea (torch.Tensor): (3,) tensor representing origin of
            reflected ray in eye's coordinate system.

            leds_inCornea (list of torch.Tensor): list of (3,) tensors each
            representing the coordinate of one LED as the destination of
            reflected ray in eye's coordinate system.

            apply_eyelids_occlusion (bool, optional): if True, returns None if
            glint is occluded by eyelids. Defaults to True.

            remove_points_beyond_limbus (bool, optional): if True, returns None
            if glint is beyond the limbus. Defaults to True.

            return_none (bool, optional): if True, returns None as an element
            of the list if the corresponding glint is not visible. Otherwise,
            omits the glint altogether. Defaults to True.

        Return:
            list of (torch.Tensor or None): list of (3,) tensors or None
            corresponding to coordinates of glint in the coordinate system of
            the eye's cornea, or None if there is no glint or
            apply_eyelids_occlusion is True and glint is occluded by eyelids.
        """

        all_glints_inCornea = []
        for destination_inCornea in leds_inCornea.T:
            glint_inCornea = \
                self.generate_glint_inCornea(
                    origin_inCornea,
                    destination_inCornea,
                    apply_eyelids_occlusion=apply_eyelids_occlusion,
                    remove_points_beyond_limbus=remove_points_beyond_limbus
                )
            if glint_inCornea is None:
                if not return_none:
                    continue

            all_glints_inCornea = all_glints_inCornea + [glint_inCornea, ]

        return all_glints_inCornea

    def generate_refracted_pupil_inCornea(
        self,
        origin_inCornea,
        num_points=30,
        apply_eyelids_occlusion=True
    ):
        """
        Generate 3D pupil points in the coordinate system of cornea,
        corresponding to the intersection with the cornea of rays that refract
        on the cornea's surface from a given origin with destination on the
        pupil.

        The output is a list of either (3,) torch.Tensors, if the pupil point
        is visible after all visibility criteria are applied, or None if the
        pupil point is not visible. Note that the pupil may be a closed curve
        on the cornea surface, an open curve (if a segment of it is occluded by
        one eyelid), or two curves (if both eyelids occlude some part of it).
        This output format makes it easier to figure out which of these
        scenarios has occurred.

        Args:
            origin_inCornea (torch.Tensor): (3,) tensor corresponding to origin
            of rays that refract on the surface of the cornea and end at the
            pupil.

            num_points (int, optional): number of points on pupil to be
            refracted from origin.

            apply_eyelids_occlusion (bool, optional): if True, remove points
            that are occluded by the eyelids. Defaults to True.

        Returns:
            list of torch.Tensor: list with num_point elements, corresponding
            to either None, if there is no refraction or the refraction is
            occluded, or a (3,) tensor with the coordinates of the refraction
            in the coordinate system of the cornea.

            list of torch.Tensor: list with num_point elements corresponding to
            a scalar with the angle in radians the parameterizes the refracted
            point in pupil coordinates.
        """

        # Sample pupil points in the cornea's coordinate system.
        pupil_points_inCornea, angles_rad = \
            self.pupil.get_points_inOther(self.cornea, num_points=num_points)

        refractions_inCornea = []
        for destination_inCornea in pupil_points_inCornea.T:
            single_refraction_inCornea = \
                self.cornea.compute_refraction_point_inEllipsoid(
                    origin_inCornea,
                    destination_inCornea,
                    eta_at_destination=self.cornea.refractive_index
                )
            refractions_inCornea = \
                refractions_inCornea + [single_refraction_inCornea, ]

        if apply_eyelids_occlusion:
            refractions_inCornea = \
                self.apply_eyelid_occlusion_inOther(
                    self.cornea, refractions_inCornea
                )

        return refractions_inCornea, angles_rad

    def generate_occluding_contour_inCornea(
        self,
        point_inCornea,
        num_points=30,
        apply_eyelids_occlusion=True,
        remove_points_beyond_limbus=True,
    ):
        """
        Generate an ellipse node attached to the eye's cornea as seen from the
        point of view of the input point in the cornea's coordinate system.

        The output is a list of (3,) torch.Tensors. Note that this contrasts
        with the output format of glints and pupil, which are lists containing
        either torch.Tensors or None values. For occluding contours we do not
        need the more complicated format, as they cannot have discontinuities
        due to occlusion.

        Args:
            point_inCornea (torch.Tensor): (3,) tensor corresponding to
            coordinates of input point in cornea's coordinate system.

            num_points (int, optional): number of points to sample along the
            occluding contour. Defaults to 30.

            apply_eyelids_occlusion (bool, optional): if True, exclude points
            occluded by the eyelids. Defaults to True.

            remove_points_beyond_limbus (bool, optional): if True, sets points
            beyond the limbus to None. Defaults to True.

        Returns:
            list of torch.Tensor: list of (3,) tensors corresponding to visible
            occluding contours in coordinate system of the eye's cornea.
        """

        # This creates a node in the pose graph, we should delete it so as not
        # to bloat it.
        contour = \
            self.cornea.compute_occluding_contour_inEllipsoid(point_inCornea)

        if contour is None:
            return []

        contour_points_inParent, _ = \
            contour.get_points_inParent(num_points=num_points)
        contour_points_inCornea = list(contour_points_inParent.T)

        # Remove the occluding contour from the pose graph.
        self.cornea.remove_child(contour)

        # Remove the parts of the occluding contour beyond the limbus, if
        # required.
        if remove_points_beyond_limbus:
            contour_points_inCornea = \
                self.remove_beyond_limbus_inOther(
                    self.cornea, contour_points_inCornea
                )

        # Remove the parts of the occluding contour under the eyelids, if
        # required.
        if apply_eyelids_occlusion:
            contour_points_inCornea = \
                self.apply_eyelid_occlusion_inOther(
                    self.cornea, contour_points_inCornea
                )

        return contour_points_inCornea

    def generate_limbus_inOther(
        self,
        other_node=None,
        num_points=30,
        apply_eyelids_occlusion=True
    ):
        """
        Generate 3D limbus coordinates unoccluded by the device's occluder, the
        eyelids, and the corneal bulge, in the coordinate system of other_node.

        Args:
            other_node (Node, optional): Node in which to generate the 3D
            coordinates of the limbus. Defaults to None, in which case we use
            the root node of the scene's pose graph.

            num_points (int, optional): number of points to sample along
            limbus. Defaults to 30.

            apply_eyelid_occlusion (bool, optional): if True, sets points
            behind eyelids to None. Defaults to True

        Returns:
            list of torch.Tensor: list holding either None (if point is not
            visible) or (3,) torch.Tensors holding 3D coordinates of limbus
            points in coordinate system of other_node.

            list of torch.Tensor: list holding the angles parameterizing the
            locations of the limbus points along the limbus contour.
        """

        # Generate coordinates of limbus in coordinate system of other_node.
        if other_node is None:
            other_node = self.get_root()

        limbus_tensor_points_inOther, \
            angles_rad = \
            self.limbus.get_points_inOther(other_node, num_points=num_points)
        limbus_inOther = list(limbus_tensor_points_inOther.T)

        if apply_eyelids_occlusion:
            limbus_inOther = \
                self.apply_eyelid_occlusion_inOther(other_node, limbus_inOther)

        return limbus_inOther, angles_rad

    def generate_eyelids_inCornea(
        self,
        num_points=30,
        remove_points_beyond_limbus=True
    ):
        """
        Generate coordinates of points on eyelids in the coordinate system of
        the eye's cornea. The output is a list with two entries, one for each
        eyelid and each one a list itself. The inner lists hold the coordinates
        of sampled points for the corresponding eyelid. Note that when we
        generate occluding contours, we check for occlusion by eyelids, but
        when we generate eyelids we do not check for occlusion by an occluding
        contour, because in order to do so we'd need to know which point in 3D
        has generated the occluding contour, and knowing that is not in the
        scope of eyelids.

        Args:
            num_points (int, optional): number of points to sample along each
            eyelid. Defaults to 30.

            remove_points_beyond_limbus (bool, optional): if True, remove the
            points beyond the limbus. Defaults to True.

        Returns:
            list of lists of torch.Tensors: a (2,) list, one for each eyelid,
            whose elements are lists of torch.Tensors corresponding to the
            coordinates of eyelid points in the coordinate system of the
            cornea.
        """

        eyelids_inCornea = []
        for eyelid in (self.eyelids.upper_eyelid, self.eyelids.lower_eyelid):
            homogeneous_eyelid_plane_inCornea = \
                eyelid.get_homogeneous_coordinates_inOther(self.cornea)

            # Create a polar point to the eyelid plane.
            polar_point_to_eyelids_plane_inCornea = \
                self.cornea.compute_polar_point_to_plane_inEllipsoid(
                    homogeneous_eyelid_plane_inCornea
                )

            # Compute intersecting ellipse in cornea. The intersecting ellipse
            # is the occluding contour generated by the polar point. This
            # should not be mistaken for the occluding contour generated by the
            # corneal bulge.
            contour = \
                self.cornea.compute_occluding_contour_inEllipsoid(
                    polar_point_to_eyelids_plane_inCornea
                )  # This adds a node to the pose graph.

            # Ignore pylance complaint about contour being None. It cannot be,
            # unless we have a poorly formed eye.
            assert (contour is not None), "Poorly formed eye."
            contour_points_inCornea, _ = \
                contour.get_points_inParent(num_points=num_points)

            # Remove the contour node from the pose graph.
            self.cornea.remove_child(contour)

            single_eyelid_inCornea = []
            for single_point_inCornea in contour_points_inCornea.T:
                if remove_points_beyond_limbus and \
                        self.point_is_beyond_limbus_inOther(
                            single_point_inCornea, self.cornea
                        ):
                    continue

                single_eyelid_inCornea = \
                    single_eyelid_inCornea + [single_point_inCornea, ]

            eyelids_inCornea = eyelids_inCornea + [single_eyelid_inCornea, ]

        return eyelids_inCornea

    def generate_eyelids_inOther(
        self,
        other_node=None,
        num_points=30,
        remove_points_beyond_limbus=True
    ):
        """
        Generate the coordinates of the eyelids in the coordinate system of
        other_node.

        Args:
            other_node (Node, optional): Node in which to generate the
            coordinates of the eyelids. Defaults to None, in which case we use
            the root of the pose graph.

            num_points (int, optional): number of points to sample along each
            eyelid. Defaults to 30.

            remove_points_beyond_limbus (bool, optional): if True, points
            behind the limbs are omitted. Defaults to True.

        Returns:
            two lists of torch.Tensors: each list corresponds to one eyelid (0
            for upper, 1 for lower).
        """

        eyelids_inCornea = \
            self.generate_eyelids_inCornea(
                num_points=num_points,
                remove_points_beyond_limbus=remove_points_beyond_limbus
            )
        if other_node is None:
            other_node = self.get_root()

        transform_toOther_fromCornea = \
            self.cornea.get_transform_toOther_fromSelf(other_node)

        eyelids_inOther = []
        for single_eyelid_inCornea in eyelids_inCornea:
            single_eyelid_inOther = []
            for pt_inCornea in single_eyelid_inCornea:
                pt_inOther = \
                    transform_toOther_fromCornea.transform(pt_inCornea)
                single_eyelid_inOther = single_eyelid_inOther + [pt_inOther, ]

            eyelids_inOther = eyelids_inOther + [single_eyelid_inOther, ]

        return eyelids_inOther

    def sample_cornea_level_sets(self, num_level_sets=7):
        """sample_cornea_level_sets.

        Sample ellipses as level sets of the cornea along the optical axis.

        This creates num_level_sets nodes in the pose graph, one for each level
        set.

        Args:
            num_level_sets (int, optional): Number of level sets to sample.
            Defaults to 7.
        """

        # Min level is the level of the limbus in the coordinate system of the
        # cornea.
        transform_toCornea_fromLimbus = \
            self.limbus.get_transform_toOther_fromSelf(self.cornea)
        min_level_point_inCornea = \
            transform_toCornea_fromLimbus.transform(torch.zeros(3))
        min_level = torch.tensor([0.0, 0.0, 1.0]) @ min_level_point_inCornea

        ellipses = \
            self.cornea.sample_level_sets(
                min_level=min_level, num_level_sets=num_level_sets
            )

        return ellipses

    def intersect_ray_withPupilPlane_fromCornea(
        self, origin_inCornea, direction_inCornea
    ):
        """
        Intersect a ray defined by a direction and an origin in the cornea
        coordinate system with the pupil plane. If the ray intersects with the
        cornea, refraction through the cornea is appropriately taken into
        account.

        Args:
            origin_inCornea (torch.Tensor): (3,) tensor corresponding to the
            coordinates of the origin of the ray in the cornea coordinate
            system.

            direction_inCornea (torch.Tensor): (3,) tensor corresponding to the
            coordinates of the direction of the ray in the cornea coordinate
            system.

        Returns:
            torch.Tensor: (3,) tensor corresponding to intersection of ray in
            cornea coordinate system with pupil plane. Returned value is in the
            coordinate system of the pupil plane.
        """

        refraction_point_inCornea, \
            refraction_direction_inCornea, \
            _ = \
            self.cornea.refract_from_origin_and_direction_inEllipsoid(
                origin_inCornea,
                direction_inCornea,
                eta_at_destination=self.cornea.refractive_index
            )
        # The method above returns the original direction and origin points if
        # there is no intersection with the cornea. This suits us, because we
        # want to return the intersection of the ray with pupil plane with or
        # without refraction through the cornea.

        T_toPupilPlane_fromCornea = \
            self.cornea.get_transform_toOther_fromSelf(self.pupil_plane)
        refraction_point_inPupilPlane = \
            T_toPupilPlane_fromCornea.transform(refraction_point_inCornea)
        refraction_direction_inPupilPlane = \
            T_toPupilPlane_fromCornea.rotation.transform(
                refraction_direction_inCornea
            )

        return \
            self.pupil_plane.intersect_from_origin_and_direction_inPlane(
                refraction_point_inPupilPlane,
                refraction_direction_inPupilPlane
            )

    def intersect_ray_withLimbusPlane_fromPlane(
        self, origin_inPlane, direction_inPlane
    ):
        """intersect_ray_withLimbusPlane_fromPlane.

        Intersect a ray defined by a direction and an origin in the limbus
        plane coordinate system with the limbus plane.

        Args:
            origin_inPlane (torch.Tensor): (3,) tensor corresponding to the
            coordinates of the origin of the ray in the limbus-plane coordinate
            system.

            direction_inPlane (torch.Tensor): (3,) tensor corresponding to the
            coordinates of the direction of the ray in the limbus-plane
            coordinate system.

        Returns:
            torch.Tensor: (3,) tensor corresponding to intersection of ray in
            limbus-plane coordinate system with limbus plane. Returned value is
            in the coordinate system of the limbus plane.
        """

        return \
            self.limbus_plane.intersect_from_origin_and_direction_inPlane(
                origin_inPlane, direction_inPlane
            )
