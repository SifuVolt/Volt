use super::Vector;
use glam::{EulerRot, Mat3};

/// Car orientation basis extracted from the same ZYX Euler matrix used by
/// RLBot's physics rotation:
///
/// Mat3::from_euler(EulerRot::ZYX, yaw, pitch, roll)
///
/// Columns are deliberately interpreted as forward, right, and up. This
/// matches the local-coordinate convention used by Python Volt/VirxERLU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3 {
    pub forward: Vector,
    pub right: Vector,
    pub up: Vector,
}

impl Matrix3 {
    pub fn new(pitch: f32, yaw: f32, roll: f32) -> Self {
        let rotation = Mat3::from_euler(EulerRot::ZYX, yaw, pitch, roll);
        let forward = rotation.x_axis;
        let right = rotation.y_axis;
        let up = rotation.z_axis;

        Self {
            forward: Vector::new(forward.x, forward.y, forward.z),
            right: Vector::new(right.x, right.y, right.z),
            up: Vector::new(up.x, up.y, up.z),
        }
    }

    /// Convert a world-space vector to local forward/right/up coordinates.
    pub fn dot(self, vector: Vector) -> Vector {
        Vector::new(
            self.forward.dot(vector),
            self.right.dot(vector),
            self.up.dot(vector),
        )
    }

    pub fn determinant(self) -> f32 {
        self.forward.dot(self.right.cross(self.up))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn near(a: Vector, b: Vector) {
        assert!((a.x - b.x).abs() < EPS, "x: {:?} != {:?}", a, b);
        assert!((a.y - b.y).abs() < EPS, "y: {:?} != {:?}", a, b);
        assert!((a.z - b.z).abs() < EPS, "z: {:?} != {:?}", a, b);
    }

    #[test]
    fn zero_rotation_maps_columns_to_world_forward_right_up() {
        let matrix = Matrix3::new(0.0, 0.0, 0.0);
        near(matrix.forward, Vector::new(1.0, 0.0, 0.0));
        near(matrix.right, Vector::new(0.0, 1.0, 0.0));
        near(matrix.up, Vector::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn yaw_rotates_forward_and_right_around_world_up() {
        let matrix = Matrix3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0);
        near(matrix.forward, Vector::new(0.0, 1.0, 0.0));
        near(matrix.right, Vector::new(-1.0, 0.0, 0.0));
        near(matrix.up, Vector::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn positive_rlbot_pitch_rotates_forward_toward_negative_world_z() {
        let matrix = Matrix3::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        near(matrix.forward, Vector::new(0.0, 0.0, -1.0));
        near(matrix.right, Vector::new(0.0, 1.0, 0.0));
        near(matrix.up, Vector::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn euler_basis_is_orthonormal_and_right_handed() {
        let matrix = Matrix3::new(0.4, 1.2, -0.7);
        assert!((matrix.forward.magnitude() - 1.0).abs() < EPS);
        assert!((matrix.right.magnitude() - 1.0).abs() < EPS);
        assert!((matrix.up.magnitude() - 1.0).abs() < EPS);
        assert!(matrix.forward.dot(matrix.right).abs() < EPS);
        assert!(matrix.forward.dot(matrix.up).abs() < EPS);
        assert!(matrix.right.dot(matrix.up).abs() < EPS);
        assert!((matrix.determinant() - 1.0).abs() < EPS);
    }

    #[test]
    fn local_coordinates_are_dot_products_with_matrix_columns() {
        let matrix = Matrix3::new(0.0, 0.0, 0.0);
        let world = Vector::new(10.0, 20.0, 30.0);
        assert_eq!(matrix.dot(world), world);
    }
}
