use crate::math::Vector;

/// Exact throttle-only acceleration curve used by Volt's Python
/// `throttle_acceleration` utility. Input is local forward velocity.
pub fn throttle_acceleration(car_velocity_x: f32) -> f32 {
    let speed = car_velocity_x.abs();

    if speed >= 1410.0 {
        return 0.0;
    }

    if speed < 1400.0 {
        return (-36.0 / 35.0) * speed + 1600.0;
    }

    let speed = speed - 1400.0;
    -16.0 * speed + 160.0
}

/// Volt's piecewise turn-curvature lookup, copied from the Python utility.
pub fn curvature(speed: f32) -> f32 {
    if (0.0..500.0).contains(&speed) {
        0.0069 - 5.84e-6 * speed
    } else if (500.0..1000.0).contains(&speed) {
        0.00561 - 3.26e-6 * speed
    } else if (1000.0..1500.0).contains(&speed) {
        0.0043 - 1.95e-6 * speed
    } else if (1500.0..1750.0).contains(&speed) {
        0.003025 - 1.1e-7 * speed
    } else if (1750.0..2500.0).contains(&speed) {
        0.0018 - 0.4e-7 * speed
    } else {
        0.0
    }
}

/// Turn radius in uu for a forward speed in uu/s.
///
/// Volt calls this with an absolute speed. At speeds outside the curvature
/// table, infinity is returned rather than allowing a division-by-zero.
pub fn turn_radius(speed: f32) -> f32 {
    if speed == 0.0 {
        return 0.0;
    }

    let curve = curvature(speed);
    if curve <= 0.0 {
        f32::INFINITY
    } else {
        1.0 / curve
    }
}

/// Matches Volt's circle-based check for whether a target lies inside the
/// car's current turn radius. The target is interpreted in local x/y space.
pub fn is_inside_turn_radius(
    turn_rad: f32,
    local_target: Vector,
    steer_direction: f32,
) -> bool {
    let target = local_target.flatten();
    let circle_y = -steer_direction * turn_rad;
    let dx = target.x;
    let dy = target.y - circle_y;

    dx * dx + dy * dy < turn_rad * turn_rad
}

/// Volt's cubic proportional-derivative steering response.
pub fn steer_pd(angle: f32, rate: f32) -> f32 {
    (((35.0 * (angle + rate)).powi(3)) / 10.0).clamp(-1.0, 1.0)
}

/// Dodge impulse from Volt's Python utility.
///
/// `max_speed` is supplied from the active physics profile; the Python
/// implementation uses the standard 2300 uu/s limit.
pub fn dodge_impulse(car_velocity: Vector, max_speed: f32) -> f32 {
    let car_speed = car_velocity.magnitude();
    let mut impulse = 500.0 * (1.0 + 0.9 * (car_speed / 2300.0));
    let excess = car_speed + impulse - max_speed;

    if excess > 0.0 {
        impulse -= excess;
    }

    impulse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_acceleration_matches_python_curve() {
        assert!((throttle_acceleration(0.0) - 1600.0).abs() < 0.001);
        assert!((throttle_acceleration(700.0) - (1600.0 - 36.0 / 35.0 * 700.0)).abs() < 0.001);
        assert!((throttle_acceleration(1400.0) - 160.0).abs() < 0.001);
        assert_eq!(throttle_acceleration(1410.0), 0.0);
        assert_eq!(throttle_acceleration(-1410.0), 0.0);
        assert_eq!(throttle_acceleration(-700.0), throttle_acceleration(700.0));
    }

    #[test]
    fn curvature_uses_python_speed_bands() {
        assert!((curvature(0.0) - 0.0069).abs() < 0.000001);
        assert!((curvature(500.0) - (0.00561 - 3.26e-6 * 500.0)).abs() < 0.000001);
        assert!((curvature(1000.0) - (0.0043 - 1.95e-6 * 1000.0)).abs() < 0.000001);
        assert!((curvature(1500.0) - (0.003025 - 1.1e-7 * 1500.0)).abs() < 0.000001);
        assert!((curvature(1750.0) - (0.0018 - 0.4e-7 * 1750.0)).abs() < 0.000001);
        assert_eq!(curvature(2500.0), 0.0);
    }

    #[test]
    fn turn_radius_is_inverse_curvature() {
        assert_eq!(turn_radius(0.0), 0.0);
        assert!((turn_radius(1000.0) - 1.0 / curvature(1000.0)).abs() < 0.001);
        assert!(turn_radius(2500.0).is_infinite());
    }

    #[test]
    fn inside_turn_radius_matches_offset_circle_geometry() {
        let radius = 100.0;
        assert!(is_inside_turn_radius(
            radius,
            Vector::new(0.0, -50.0, 0.0),
            1.0
        ));
        assert!(!is_inside_turn_radius(
            radius,
            Vector::new(0.0, 200.0, 0.0),
            1.0
        ));
    }

    #[test]
    fn steer_pd_is_cubic_and_clamped() {
        assert_eq!(steer_pd(0.0, 0.0), 0.0);
        assert_eq!(steer_pd(1.0, 0.0), 1.0);
        assert_eq!(steer_pd(-1.0, 0.0), -1.0);
        assert!((steer_pd(0.01, 0.0) - (0.35_f32.powi(3) / 10.0)).abs() < 0.000001);
    }

    #[test]
    fn dodge_impulse_respects_speed_cap() {
        let impulse = dodge_impulse(Vector::new(1000.0, 0.0, 0.0), 2300.0);
        assert!((impulse - (500.0 * (1.0 + 0.9 * 1000.0 / 2300.0))).abs() < 0.001);

        let capped = dodge_impulse(Vector::new(2200.0, 0.0, 0.0), 2300.0);
        assert!((2200.0 + capped - 2300.0).abs() < 0.001);
    }
}
