"""Classes setting parameters with respect to which we compute derivatives.

Some parameters are trivial, e.g., camera focal length: they are simple 'there
already.' However, parameters such as the camera pose are expressed through a
transformation matrix, which is not a parsimonious representation of what we
would want in this case, which would simply be rotation in SO(3) and
translation in R(3). In this case, we apply an 'identity perturbation'
to the camera, indeed represented parsimoniously, and compute derivatives with
respect to that representation.

Specifically we apply an identity perturbation to eye-, camera-, and LED-pose
parameters, and directly access eye-shape and intrinsic camera parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import torch


# Some small helper classes.
class LEDsParameters():
    """Wrapper for LED location parameters.

    The main elements of this class are:

    self.all_coordinates_inSubsystem: a list of (3,) torch tensors
    corresponding to the coordinates of each LED in the coordinate system of
    the subsystem to which they are associated.

    self.all_translations: a list of (3,) torch tensor corresponding to an
    identity (i.e., zero) translation applied to each LED in the coordinate
    system of the main camera of the same subsystem to which the LEDs are
    associated.

    self.joint_axis_angle: a (3,) torch tensor corresponding to the axis-angle
    of an identity (i.e., zero) rotation applied to all LEDs in the coordinate
    system of the their subsystem.

    self.joint_translation: a (3,) torch tensor corresponding to an identity
    translation (i.e., zero) applied to all LEDs.
    """

    def __init__(self, subsystem):
        """Set LED location parameters.

        We create identity transformations for each LED. In addition to
        individual transformations, we also provide a helper parameters that
        transforms all LEDs at once. All transformations are in the coordinate
        system of the reference camera of the subsystem with which the LEDs are
        associated.

        Args:
            subsystem (device.SubsystemModel): eye-tracking subsystem with LEDs
            with respect to whose position we compute derivatives. We pass the
            whole subsystem as an argument because the derivatives are computed
            with respect to a transformation in the coordinate system of the
            subsystem's camera.
        """

        self.subsystem = subsystem
        self.camera = self.subsystem.cameras[0]
        self.leds = self.subsystem.led_set

        # First, we breakdown the LED set into individual LEDs
        all_coordinates_inSubsystem = \
            torch.split(self.leds.coordinates, 1, dim=1)

        # Then, we transform each LED with its own identity transformation.
        T_toCamera_fromLEDParent = \
            self.leds.parent.get_transform_toOther_fromSelf(self.camera)
        T_toLEDParent_fromCamera = T_toCamera_fromLEDParent.create_inverse()

        self.all_translations = []
        self.all_coordinates_inSubsystem = []
        for led in all_coordinates_inSubsystem:
            translation = torch.zeros(3, requires_grad=True)
            self.all_translations = self.all_translations + [translation, ]

            # We want to transform the LED coordinates with respect to the
            # camera coordinate system, but we need to achieve that through the
            # base-class method of the update_transform_toParent_fromSelf, as
            # those are applied to the LED coordinates prior to their addition
            # to the computational graph of glint measurements. Therefore, we
            # seek the transformation T_toParent_fromParent such that
            #
            # T_toLEDParent_fromLEDParent = \
            #   T_toLEDParent_fromCamera * \
            #       T_toCamera_fromCamera * T_toCamera_fromLEDParent

            translation_toCamera_fromCamera = core.SE3(translation)

            # This is the identity transformation, but we seek to compute
            # derivative with respect to changes around the LEDs current
            # position.
            T_toParent_fromParent = \
                core.SE3.compose_transforms(
                    core.SE3.compose_transforms(
                        T_toLEDParent_fromCamera,
                        translation_toCamera_fromCamera
                    ),
                    T_toCamera_fromLEDParent
                )

            new_led = T_toParent_fromParent.transform(led).flatten()
            self.all_coordinates_inSubsystem = \
                self.all_coordinates_inSubsystem + [new_led, ]

        # Reconstruct the LED set a single 3 x N tensor, as assumed elsewhere.
        # This are exactly the same values as before, but now they have a
        # dependence on axis-angles and translation tensors stored in
        # self.all_axis_angles and self.all_translations.
        self.leds.coordinates = \
            torch.stack(self.all_coordinates_inSubsystem, dim=1)

        # Finally, we also create axis-angle and translation tensors that apply
        # jointly to all LEDs.
        self.joint_axis_angle = torch.zeros(3, requires_grad=True)
        self.joint_translation = torch.zeros(3, requires_grad=True)

        # We want to transform the LED coordinates with respect to the
        # camera coordinate system, but we need to achieve that through the
        # base-class method of the update_transform_toParent_fromSelf, as
        # those are applied to the LED coordinates prior to their addition
        # to the computational graph of glint measurements. Therefore, we
        # seek the transformation T_toParent_fromParent such that
        #
        # T_toLEDParent_fromLEDParent = \
        #   T_toLEDParent_fromCamera * \
        #       T_toCamera_fromCamera * T_toCamera_fromLEDParent

        rotation_toCamera_fromCamera = core.SO3(self.joint_axis_angle)
        translation_toCamera_fromCamera = core.SE3(self.joint_translation)

        # This is the identity transformation, but we seek to compute
        # derivative with respect to changes around the LEDs current
        # position.
        T_toCamera_fromCamera = \
            core.SE3.compose_transforms(
                rotation_toCamera_fromCamera,
                translation_toCamera_fromCamera
            )

        T_toParent_fromParent = \
            core.SE3.compose_transforms(
                core.SE3.compose_transforms(
                    T_toLEDParent_fromCamera, T_toCamera_fromCamera
                ),
                T_toCamera_fromLEDParent
            )

        self.leds.update_transform_toParent_fromSelf(T_toParent_fromParent)


class CameraExtrinsicParameters():
    """Wrapper for angle-axis and translation parameters for camera pose.
    """

    def __init__(self, camera):
        """Set extrinsics with respect to which we compute derivatives.

        We create identity rotation and translation perturbations, with respect
        to which we will later compute derivatives.

        Args:
            camera (device.Polynomial3KCamera): camera with respect to whose
            extrinsics we compute derivatives.
        """

        self.camera = camera

        # We wish to compute derivatives of our measurements with respect to
        # the rotation and translation below.
        self.axis_angle = torch.zeros(3, requires_grad=True)
        self.translation = torch.zeros(3, requires_grad=True)

        # We have to effect this transformation using the method
        # update_transform_toParent_fromSelf, which takes as input an update
        # transformation to parent from parent. Therefore, we seek a
        # transformation T_toParent_fromParent such that
        #
        # T_toParent_fromParent = \
        # T_toParent_fromCamera * T_toCamera_fromCamera * T_toCamera_fromParent
        T_toParent_fromCamera = self.camera.transform_toParent_fromSelf
        T_toCamera_fromParent = T_toParent_fromCamera.create_inverse()

        rotation_toCamera_fromCamera = core.SO3(self.axis_angle)
        translation_toCamera_fromCamera = core.SE3(self.translation)

        T_toCamera_fromCamera = core.SE3.compose_transforms(
            rotation_toCamera_fromCamera, translation_toCamera_fromCamera
        )

        T_toParent_fromParent = \
            core.SE3.compose_transforms(
                core.SE3.compose_transforms(
                    T_toParent_fromCamera, T_toCamera_fromCamera
                ),
                T_toCamera_fromParent
            )

        # We change the camera through its reference.
        self.camera.update_transform_toParent_fromSelf(T_toParent_fromParent)


class CameraIntrinsicParameters():
    """Wrapper for camera intrinsic parameters.
    """

    def __init__(self, camera):
        """Set intrinsics with respect to which we compute derivatives.

        We directly access the camera intrinsic parameters.

        Args:
            camera (device.Polynomial3KCamera): camera with respect to whose
            intrinsics we compute derivatives.
        """

        self.camera = camera

        self.focal_lengths = self.camera.focal_lengths
        self.principal_point = self.camera.principal_point
        self.distortion_center = self.camera.distortion_center
        self.distortion_coefficients = self.camera.distortion_coefficients


class EyePoseParameters():
    """Wrapper for eye-pose parameters.
    """

    def __init__(self, eye):
        """Set eye-pose parameters w.r.t. which we compute derivatives.

        We create identity rotation and translation with respect to which we
        will later compute derivatives.

        Args:
            eye (EyeModel): eye model with respect to whose shape
            parameters we compute derivatives.
        """

        self.eye = eye
        self.cornea = self.eye.cornea

        self.set_differentiable_parameters()

    def set_differentiable_parameters(self):
        """Set rotation angles and translation w.r.t. which to differentiate.
        """

        self.angles_deg = torch.zeros(2, requires_grad=True)
        self.translation = torch.zeros(3, requires_grad=True)

        self.eye.rotate_from_gaze_angles_inParent(self.angles_deg)
        self.eye.translate_inParent(self.translation)

    def rotate_eye(self, angles_deg):
        """Rotate the eye and reset the derivative parameters.

        Once the eye is rotated, we reset the parameters with respect to which
        we compute derivatives.

        Args:
            angles_deg (torch.Tensor): (2,) torch tensor corresponding to the
            horizontal and vertical angles in degrees by which the gaze is
            rotated.
        """

        self.eye.rotate_from_gaze_angles_inParent(angles_deg)
        self.set_differentiable_parameters()

    def unrotate_eye(self, angles_deg):
        """Reverse the effect of eye rotation.

        Args:
            angles_deg (torch.Tensor): (2,) torch tensor corresponding to the
            horizontal and vertical angles in degrees by which the gaze is
            unrotated.
        """

        self.eye.unrotate_from_gaze_angles_inParent(angles_deg)
        self.set_differentiable_parameters()


class EyeShapeParameters():
    """Wrapper for eye-shape parameters.
    """

    def __init__(self, eye):
        """Set eye-shape parameters w.r.t. which we compute derivatives.

        We directly access eye-shape parameters, plus the refractive index of
        the cornea.

        Args:
            eye (user.EyeModel): eye model with respect to whose shape
            parameters we compute derivatives.
        """

        self.eye = eye
        self.cornea = self.eye.cornea

        self.distance_from_rotation_center_to_cornea_center = \
            self.eye.distance_from_rotation_center_to_cornea_center
        self.distance_from_rotation_center_to_pupil_plane = \
            self.eye.distance_from_rotation_center_to_pupil_plane
        self.distance_from_rotation_center_to_limbus_plane = \
            self.eye.distance_from_rotation_center_to_limbus_plane
        self.shape_parameters = self.cornea.shape_parameters
        self.refractive_index = self.cornea.refractive_index
        self.pupil_radius = self.eye.pupil.radius
        self.limbus_radius = self.eye.limbus.radius
