
use super::Vector;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3 {
    pub forward: Vector,
    pub right: Vector,
    pub up: Vector,
}

impl Matrix3 {
    pub fn new(pitch: f32, yaw: f32, roll: f32) -> Self {
        let (sp, cp) = pitch.sin_cos();
        let (sy, cy) = yaw.sin_cos();
        let (sr, cr) = roll.sin_cos();

        Self {
            forward: Vector::new(cp * cy, cp * sy, sp),

            right: Vector::new(
                cy * sp * sr - cr * sy,
                sy * sp * sr + cr * cy,
                -cp * sr,
            ),

            up: Vector::new(
                -cr * cy * sp - sr * sy,
                -cr * sy * sp + sr * cy,
                cp * cr,
            ),
        }
    }

    // Converts a world-space vector into local car coordinates.
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

    #[test]
    fn zero_rotation_matches_world_axes() {
        let matrix = Matrix3::new(0.0, 0.0, 0.0);

        assert_eq!(matrix.forward, Vector::new(1.0, 0.0, 0.0));
        assert_eq!(matrix.right, Vector::new(0.0, 1.0, 0.0));
        assert_eq!(matrix.up, Vector::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn local_coordinates_match_basis() {
        let matrix = Matrix3::new(0.0, 0.0, 0.0);
        let world = Vector::new(10.0, 20.0, 30.0);

        assert_eq!(matrix.dot(world), world);
    }

    #[test]
    fn orientation_is_orthonormal() {
        let matrix = Matrix3::new(0.4, 1.2, -0.7);

        assert!((matrix.forward.magnitude() - 1.0).abs() < 0.0001);
        assert!((matrix.right.magnitude() - 1.0).abs() < 0.0001);
        assert!((matrix.up.magnitude() - 1.0).abs() < 0.0001);

        assert!(matrix.forward.dot(matrix.right).abs() < 0.0001);
        assert!(matrix.forward.dot(matrix.up).abs() < 0.0001);
        assert!(matrix.right.dot(matrix.up).abs() < 0.0001);

        assert!((matrix.determinant() - 1.0).abs() < 0.0001);
    }
}
