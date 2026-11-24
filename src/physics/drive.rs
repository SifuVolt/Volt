use crate::math::Vector;
use crate::state::CarState;

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


/// Party Cannon's speed-dependent turn-radius table, used by its Drive ETA
/// model. Unlike a fixed-radius Dubins path, this follows Rocket League's
/// measured cornering response as speed changes.
pub fn party_cannon_turn_radius(speed: f32) -> f32 {
    let speed = speed.abs().clamp(0.01, 2300.0);
    if speed <= 500.0 {
        145.0 + (speed / 500.0) * (251.0 - 145.0)
    } else if speed <= 1000.0 {
        251.0 + ((speed - 500.0) / 500.0) * (425.0 - 251.0)
    } else if speed <= 1500.0 {
        425.0 + ((speed - 1000.0) / 500.0) * (727.0 - 425.0)
    } else if speed <= 1750.0 {
        727.0 + ((speed - 1500.0) / 250.0) * (909.0 - 727.0)
    } else {
        909.0 + ((speed - 1750.0) / 550.0) * (1136.0 - 909.0)
    }
}

pub fn party_cannon_speed_from_turn_radius(radius: f32) -> f32 {
    let radius = radius.clamp(145.0, 1136.0);
    if radius <= 251.0 {
        ((radius - 145.0) / 106.0) * 500.0
    } else if radius <= 425.0 {
        500.0 + ((radius - 251.0) / 174.0) * 500.0
    } else if radius <= 727.0 {
        1000.0 + ((radius - 425.0) / 302.0) * 500.0
    } else if radius <= 909.0 {
        1500.0 + ((radius - 727.0) / 182.0) * 250.0
    } else {
        1750.0 + ((radius - 909.0) / 227.0) * 550.0
    }
}

pub fn party_cannon_speed_after_turn(current_speed: f32, angle: f32, modifier: f32) -> f32 {
    let turn = angle.max(0.0) * 0.49 * modifier;
    let exponential = turn.exp();
    let response = exponential * turn * if current_speed > 1234.0 { 0.2 } else { 1.0 };
    ((1234.0 * response + current_speed) / (response + 1.0)).clamp(0.0, 2300.0)
}

fn party_cannon_turn_geometry(
    car: &CarState,
    target: Vector,
    backwards: bool,
) -> (f32, f32, f32, f32) {
    let normal = car.up().normalize();
    let forward = (car.forward() - normal * car.forward().dot(normal)).normalize();
    let right = forward.cross(-normal).normalize();
    let target_delta = target - car.location;
    let target_delta = target_delta - normal * target_delta.dot(normal);
    let direct_distance = target_delta.magnitude();
    if direct_distance < 1.0 || forward.magnitude() < 0.001 || right.magnitude() < 0.001 {
        return (direct_distance, 0.0, party_cannon_turn_radius(car.velocity.magnitude()), 0.0);
    }

    let target_direction = target_delta / direct_distance;
    let heading = if backwards { -forward } else { forward };
    let angle = heading.dot(target_direction).clamp(-1.0, 1.0).acos();
    let current_speed = car.velocity.dot(forward);
    let turn_speed = if backwards {
        party_cannon_speed_after_turn(-current_speed, angle, 0.4)
    } else {
        party_cannon_speed_after_turn(current_speed, angle, 0.5)
    };
    let mut radius = party_cannon_turn_radius(turn_speed);
    let side = right.dot(target_delta).signum();

    if angle < 1.0e-4 || side.abs() < 1.0e-4 {
        if angle < 1.0e-4 {
            return (direct_distance, 0.0, radius, angle);
        }
        let turn_distance = radius * angle;
        return (turn_distance + (direct_distance - turn_distance).max(0.0), turn_distance, radius, angle);
    }

    let mut center = car.location + right * (side * radius);
    let mut center_to_target = target - center;
    center_to_target = center_to_target - normal * center_to_target.dot(normal);
    let mut center_distance = center_to_target.magnitude();

    if center_distance < radius {
        let lateral_component = right.dot(target_direction).abs().max(1.0e-4);
        radius = direct_distance / (2.0 * lateral_component);
        center = car.location + right * (side * radius);
        center_to_target = target - center;
        center_to_target = center_to_target - normal * center_to_target.dot(normal);
        center_distance = center_to_target.magnitude();
    }

    if !radius.is_finite() || !center_distance.is_finite() || center_distance < 1.0 {
        return (direct_distance, 0.0, party_cannon_turn_radius(turn_speed), angle);
    }

    let radial_start = (car.location - center).normalize();
    let radial_end = center_to_target.normalize();
    let flat_angle = radial_start.dot(radial_end).clamp(-1.0, 1.0).acos();
    let swept_angle = if target_delta.dot(heading) < 0.0 {
        (flat_angle - 2.0 * std::f32::consts::PI).abs()
    } else {
        flat_angle
    };
    let tangent_angle = (radius / center_distance).clamp(0.0, 1.0).acos();
    let arc_angle = (swept_angle - tangent_angle).clamp(0.0, 2.0 * std::f32::consts::PI);
    let tangent_distance = (center_distance * center_distance - radius * radius).max(0.0).sqrt();
    let turn_distance = radius * arc_angle;
    (tangent_distance + turn_distance, turn_distance, radius, angle)
}

/// Party Cannon's Drive.GetEta model for a grounded car on its current
/// drivable surface. Returns None for airborne cars so callers can use their
/// ballistic/landing-aware ETA instead.
pub fn party_cannon_ground_eta(
    car: CarState,
    target: Vector,
    boost_accel: f32,
    allow_dodges: bool,
) -> Option<f32> {
    if car.airborne {
        return None;
    }

    let eta_for_direction = |backwards: bool, allow_dodges: bool| {
        let (path_distance, turn_distance, radius, angle) =
            party_cannon_turn_geometry(&car, target, backwards);
        let distance = (path_distance - turn_distance).max(0.0);
        let turn_time = turn_distance
            / party_cannon_speed_from_turn_radius(radius).max(400.0);
        let current_speed = car.velocity.dot(car.forward().normalize());

        if backwards {
            let speed = party_cannon_speed_after_turn(-current_speed, angle, 0.8).max(1400.0);
            return turn_time + distance / speed;
        }

        let min_speed = party_cannon_speed_after_turn(current_speed, angle, 1.0).max(1400.0);
        let accel = boost_accel.max(0.0);
        let boost_consumption = 33.333_333;
        let fin_speed = (min_speed + accel * car.boost.max(0.0) / boost_consumption)
            .clamp(1400.0, 2300.0);
        if accel <= 0.0 {
            return turn_time + distance / min_speed;
        }

        let distance_while_boosting =
            ((fin_speed * fin_speed - min_speed * min_speed) / (2.0 * accel)).max(0.0);
        if distance < distance_while_boosting {
            let final_speed = (min_speed * min_speed + 2.0 * accel * distance)
                .max(0.0).sqrt().clamp(1400.0, 2300.0);
            return turn_time + distance / ((min_speed + final_speed) * 0.5).max(1.0);
        }

        let acceleration_distance = distance_while_boosting;
        let acceleration_time = acceleration_distance
            / ((min_speed + fin_speed) * 0.5).max(1.0);
        let remaining = (distance - acceleration_distance).max(0.0);
        let cruise_speed = if allow_dodges && distance / fin_speed.max(1.0) > 1.25 {
            fin_speed + 500.0
        } else {
            fin_speed
        };
        turn_time + acceleration_time + remaining / cruise_speed.max(1.0)
    };

    let forward_eta = eta_for_direction(false, false);
    let backward_eta = eta_for_direction(true, false);
    let local_direction = (target - car.location).flatten().normalize();
    let forward = car.forward().flatten().normalize();
    let backwards = backward_eta + 0.5 < forward_eta
        && car.forward().dot(car.velocity) < 500.0
        && forward.dot(local_direction).clamp(-1.0, 1.0).acos()
            > std::f32::consts::PI * 0.6;

    Some(eta_for_direction(backwards, allow_dodges))
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
