
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn magnitude(self) -> f32 {
        self.x.mul_add(
            self.x,
            self.y.mul_add(self.y, self.z * self.z),
        ).sqrt()
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x.mul_add(
            other.x,
            self.y.mul_add(other.y, self.z * other.z),
        )
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn normalize(self) -> Self {
        let magnitude = self.magnitude();

        if magnitude == 0.0 {
            Self::ZERO
        } else {
            self / magnitude
        }
    }

    pub fn normalize_with_magnitude(self) -> (Self, f32) {
        let magnitude = self.magnitude();

        if magnitude == 0.0 {
            (Self::ZERO, 0.0)
        } else {
            (self / magnitude, magnitude)
        }
    }

    pub fn flatten(self) -> Self {
        Self::new(self.x, self.y, 0.0)
    }

    pub fn distance(self, other: Self) -> f32 {
        (self - other).magnitude()
    }

    pub fn flat_distance(self, other: Self) -> f32 {
        self.flatten().distance(other.flatten())
    }

    pub fn angle(self, other: Self) -> f32 {
        self.normalize()
            .dot(other.normalize())
            .clamp(-1.0, 1.0)
            .acos()
    }

    pub fn angle_2d(self, other: Self) -> f32 {
        self.flatten().angle(other.flatten())
    }

    pub fn rotate(self, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();

        Self::new(
            cos * self.x - sin * self.y,
            sin * self.x + cos * self.y,
            self.z,
        )
    }

    pub fn cap(self, low: f32, high: f32) -> Self {
        Self::new(
            self.x.clamp(low, high),
            self.y.clamp(low, high),
            self.z.clamp(low, high),
        )
    }

    // Matches Volt's clamp2D: constrains a direction to the
    // angular region bounded by start and end.
    pub fn clamp_2d(self, start: Self, end: Self) -> Self {
        let s = self.normalize();
        let down = Self::new(0.0, 0.0, -1.0);

        let right = s.dot(end.cross(down)) < 0.0;
        let left = s.dot(start.cross(down)) > 0.0;

        let inside = if end.dot(start.cross(down)) > 0.0 {
            right && left
        } else {
            right || left
        };

        if inside {
            return self;
        }

        if start.dot(s) < end.dot(s) {
            end
        } else {
            start
        }
    }

    // Matches Volt's 3D clamp: first clamps the horizontal
    // direction, then clamps height between the two bounds.
    pub fn clamp_between(self, start: Self, end: Self) -> Self {
        let mut result = self.clamp_2d(start, end);

        let low_z = start.z.min(end.z);
        let high_z = start.z.max(end.z);

        result.z = result.z.clamp(low_z, high_z);
        result
    }

    pub fn copy(self) -> Self {
        self
    }

    /// Component-wise midpoint, matching VirxERLU's Vector.midpoint().
    pub fn midpoint(self, other: Self) -> Self {
        (self + other) * 0.5
    }

    /// Compatibility names for the original Python Vector API.
    pub fn dist(self, other: Self) -> f32 {
        self.distance(other)
    }

    pub fn flat_dist(self, other: Self) -> f32 {
        self.flat_distance(other)
    }

    pub fn angle2d(self, other: Self) -> f32 {
        self.angle_2d(other)
    }

    pub fn clamp(self, start: Self, end: Self) -> Self {
        self.clamp_between(start, end)
    }

    pub fn round_to(self, decimals: i32) -> Self {
        let factor = 10.0_f32.powi(decimals);
        Self::new(
            (self.x * factor).round() / factor,
            (self.y * factor).round() / factor,
            (self.z * factor).round() / factor,
        )
    }

    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vector {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vector> for f32 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Vector {
        rhs * self
    }
}

impl Mul<Vector> for Vector {
    type Output = Self;

    /// Element-wise multiplication, matching Python Vector * Vector.
    fn mul(self, rhs: Vector) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Div<f32> for Vector {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Div<Vector> for Vector {
    type Output = Self;

    /// Element-wise division, matching Python Vector / Vector.
    fn div(self, rhs: Vector) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Neg for Vector {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::Vector;

    #[test]
    fn vector_math_works() {
        let a = Vector::new(3.0, 4.0, 0.0);
        let b = Vector::new(0.0, 0.0, 2.0);

        assert_eq!(a.magnitude(), 5.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), Vector::new(8.0, -6.0, 0.0));
        assert_eq!(a.flatten(), Vector::new(3.0, 4.0, 0.0));
        assert_eq!(a.distance(b), (29.0_f32).sqrt());
    }

    #[test]
    fn zero_vector_normalizes_safely() {
        assert_eq!(Vector::ZERO.normalize(), Vector::ZERO);
    }

    #[test]
    fn python_vector_helpers_and_componentwise_ops_are_preserved() {
        let a = Vector::new(2.0, 4.0, 6.0);
        let b = Vector::new(4.0, 8.0, 10.0);
        assert_eq!(a.midpoint(b), Vector::new(3.0, 6.0, 8.0));
        assert_eq!(a * b, Vector::new(8.0, 32.0, 60.0));
        assert_eq!(b / a, Vector::new(2.0, 2.0, 10.0 / 6.0));
        assert_eq!(a.dist(b), a.distance(b));
        assert_eq!(a.flat_dist(b), a.flat_distance(b));
        assert_eq!(Vector::new(1.234, 2.345, 3.456).round_to(2), Vector::new(1.23, 2.35, 3.46));
    }

    #[test]
    fn cap_is_component_wise_like_virxerlu_vector() {
        let value = Vector::new(-300.0, 1000.0, 3000.0);
        assert_eq!(value.cap(0.0, 2300.0), Vector::new(0.0, 1000.0, 2300.0));
    }

    #[test]
    fn rotation_preserves_height() {
        let v = Vector::new(1.0, 0.0, 5.0);
        let rotated = v.rotate(std::f32::consts::FRAC_PI_2);

        assert!(rotated.x.abs() < 0.0001);
        assert!((rotated.y - 1.0).abs() < 0.0001);
        assert_eq!(rotated.z, 5.0);
    }
}
