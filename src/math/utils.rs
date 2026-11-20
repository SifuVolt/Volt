use crate::{math::Vector, state::CarState};

/// Clamp a scalar to an inclusive interval, matching Volt's Python cap().
pub fn cap(value: f32, low: f32, high: f32) -> f32 {
    value.max(low).min(high)
}

/// Keep a target inside the playable field using the car's hitbox length.
///
/// This is a direct port of Volt's utils.cap_in_field: the x boundary changes
/// between the side-wall and goal-mouth regions, while y is only clamped near
/// the side-wall/goal-mouth transition.
pub fn cap_in_field(car: &CarState, mut target: Vector) -> Vector {
    let half_length = car.hitbox.length;
    if target.x.abs() > 893.0 - half_length {
        target.y = cap(target.y, -5120.0 + half_length, 5120.0 - half_length);
    }
    target.x = if car.location.y.abs() > 5120.0 - half_length / 2.0 {
        cap(target.x, -893.0 + half_length, 893.0 - half_length)
    } else {
        cap(target.x, -4093.0 + half_length, 4093.0 - half_length)
    };
    target
}

/// True when a point lies inside Volt's standard field boundary.
pub fn in_field(point: Vector, radius: f32) -> bool {
    let x = point.x.abs();
    let y = point.y.abs();
    let z = point.z.abs();
    let _ = z;
    !(x > 4080.0 - radius
        || y > 5900.0 - radius
        || (x > 880.0 - radius && y > 5105.0 - radius)
        || (x > 2650.0 && y > -x + 8025.0 - radius))
}

/// Signed lateral position relative to a shot axis (the shot vector is +Y).
pub fn find_slope(shot_vector: Vector, car_to_target: Vector) -> f32 {
    let forward = shot_vector.dot(car_to_target);
    let lateral = shot_vector
        .cross(Vector::new(0.0, 0.0, 1.0))
        .dot(car_to_target)
        .abs();
    if lateral <= f32::EPSILON {
        return 10.0 * sign(forward);
    }
    cap(forward / lateral, -3.0, 3.0)
}

/// Return real roots of a quadratic, or an empty vector for no real roots.
pub fn quadratic(a: f32, b: f32, c: f32) -> Vec<f32> {
    if a.abs() <= f32::EPSILON {
        if b.abs() <= f32::EPSILON {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Vec::new();
    }
    let root = discriminant.sqrt();
    if root == 0.0 {
        vec![-b / (2.0 * a)]
    } else {
        vec![(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)]
    }
}

/// Team side convention used by Volt: blue=-1, orange=+1.
pub fn side(team: u32) -> f32 {
    if team == 0 { -1.0 } else { 1.0 }
}

pub fn sign(value: f32) -> f32 {
    if value < 0.0 { -1.0 } else if value > 0.0 { 1.0 } else { 0.0 }
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a) * t + a
}

pub fn invlerp(a: f32, b: f32, value: f32) -> f32 {
    (value - a) / (b - a)
}

pub fn almost_equals(x: f32, y: f32, threshold: f32) -> bool {
    x - threshold < y && y < x + threshold
}

fn triangle_area(a: Vector, b: Vector, c: Vector) -> f32 {
    ((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2.0).abs()
}

/// Point-in-quadrilateral test using the same area decomposition as Volt.
pub fn point_inside_quadrilateral_2d(point: Vector, quad: [Vector; 4]) -> bool {
    let actual = triangle_area(quad[0], quad[1], point)
        + triangle_area(quad[2], quad[1], point)
        + triangle_area(quad[2], quad[3], point)
        + triangle_area(quad[0], quad[3], point);
    let total = triangle_area(quad[0], quad[2], quad[1])
        + triangle_area(quad[0], quad[2], quad[3]);
    almost_equals(actual, total, 0.001)
}

/// Ramanujan-style ellipse perimeter approximation used by Volt.
pub fn perimeter_of_ellipse(a: f32, b: f32) -> f32 {
    std::f32::consts::PI
        * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn car(location: Vector, hitbox_length: f32) -> CarState {
        CarState {
            index: 0,
            team: 0,
            location,
            velocity: Vector::ZERO,
            orientation: Matrix3::new(0.0, 0.0, 0.0),
            angular_velocity: Vector::ZERO,
            demolished: false,
            airborne: false,
            supersonic: false,
            jumped: false,
            doublejumped: false,
            air_state: AirState::OnGround,
            has_jumped: false,
            has_double_jumped: false,
            has_dodged: false,
            dodge_timeout: 0.0,
            dodge_elapsed: 0.0,
            dodge_dir: Vector::ZERO,
            flip_reset_available: false,
            boost: 0.0,
            hitbox: Hitbox { length: hitbox_length, ..Hitbox::default() },
        }
    }

    #[test]
    fn cap_in_field_respects_side_and_goal_mouth_bounds() {
        let normal = car(Vector::ZERO, 118.0);
        let target = cap_in_field(&normal, Vector::new(4200.0, 5200.0, 0.0));
        assert!((target.x - 3975.0).abs() < 0.001);
        assert!((target.y - 5002.0).abs() < 0.001);

        let inside_goal = car(Vector::new(0.0, 5100.0, 0.0), 118.0);
        let target = cap_in_field(&inside_goal, Vector::new(1200.0, 0.0, 0.0));
        assert!((target.x - 775.0).abs() < 0.001);
    }

    #[test]
    fn field_geometry_excludes_walls_and_corner_cutouts() {
        assert!(in_field(Vector::ZERO, 0.0));
        assert!(!in_field(Vector::new(4100.0, 0.0, 0.0), 0.0));
        assert!(!in_field(Vector::new(1000.0, 5200.0, 0.0), 0.0));
    }

    #[test]
    fn slope_and_quadratic_match_python_helpers() {
        assert_eq!(find_slope(Vector::new(0.0, 1.0, 0.0), Vector::new(0.0, 2.0, 0.0)), 10.0);
        assert_eq!(find_slope(Vector::new(0.0, 1.0, 0.0), Vector::new(2.0, 2.0, 0.0)), 1.0);
        assert_eq!(quadratic(1.0, -3.0, 2.0), vec![2.0, 1.0]);
        assert_eq!(quadratic(1.0, 0.0, 1.0), Vec::<f32>::new());
    }

    #[test]
    fn interpolation_and_shape_helpers_are_consistent() {
        assert_eq!(side(0), -1.0);
        assert_eq!(side(1), 1.0);
        assert_eq!(lerp(10.0, 20.0, 0.25), 12.5);
        assert_eq!(invlerp(10.0, 20.0, 15.0), 0.5);
        assert!(almost_equals(1.0, 1.0005, 0.001));
        assert!(point_inside_quadrilateral_2d(
            Vector::new(1.0, 1.0, 0.0),
            [
                Vector::new(0.0, 0.0, 0.0),
                Vector::new(2.0, 0.0, 0.0),
                Vector::new(2.0, 2.0, 0.0),
                Vector::new(0.0, 2.0, 0.0),
            ]
        ));
        assert!((perimeter_of_ellipse(1.0, 1.0) - 2.0 * std::f32::consts::PI).abs() < 1e-5);
    }
}
