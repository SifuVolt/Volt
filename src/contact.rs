//! Hitbox-centre targets and labelled ball-side approach geometry shared by
//! Volt's shot routines. Ground/jump contact labels are selected by the same
//! five-surface scoring used in Python tools.select_ball_contact.

use crate::math::Vector;
use crate::state::CarState;

pub const BALL_RADIUS: f32 = 92.75;

/// Convert a ball contact into a car-centre target using actual hitbox length
/// and local hitbox offset. contact_drop is zero for front-centre hits and
/// half the ball radius for Party Cannon aerial nose contact.
/// Build the approach point for the five legacy ball-side labels used by
/// the Python ground/jump routines. Seven-point hitbox contacts use
/// select_front_contact directly; this helper remains for older side labels.
pub fn approach_from_face(
    ball: Vector,
    shot_direction: Vector,
    face: &str,
    standoff: f32,
) -> Option<Vector> {
    let shot = shot_direction.flatten().normalize();
    if shot.magnitude() < 0.001 {
        return None;
    }
    let lateral = Vector::new(-shot.y, shot.x, 0.0).normalize();
    let normal = match face {
        "centre" => -shot,
        "left" => (-shot + lateral * 0.55).normalize(),
        "right" => (-shot - lateral * 0.55).normalize(),
        "high" => (-shot + Vector::new(0.0, 0.0, 0.30)).normalize(),
        "low" => (-shot - Vector::new(0.0, 0.0, 0.30)).normalize(),
        // Labels emitted by the shared front-hitbox search. A top-face
        // contact places the car centre below the ball; bottom places it above.
        "top_left" => (-shot + lateral * 0.55 - Vector::new(0.0, 0.0, 0.30)).normalize(),
        "top_right" => (-shot - lateral * 0.55 - Vector::new(0.0, 0.0, 0.30)).normalize(),
        "top" => (-shot - Vector::new(0.0, 0.0, 0.30)).normalize(),
        "bottom" => (-shot + Vector::new(0.0, 0.0, 0.30)).normalize(),
        _ => return None,
    };
    Some(ball + normal * standoff.max(120.0))
}

/// Temporary flank route shared by ground and jump approaches. It fades out
/// before contact so the final hit still follows the chosen contact normal.
pub fn route_around_ball(
    car: CarState,
    ball: Vector,
    final_target: Vector,
    shot_direction: Vector,
    time_left: f32,
) -> Vector {
    let final_target = final_target.flatten();
    if time_left <= 0.25 {
        return final_target;
    }
    let shot_direction = shot_direction.flatten().normalize();
    if shot_direction.magnitude() < 1.0 {
        return final_target;
    }
    let car_to_ball = (ball - car.location).flatten().normalize();
    if car_to_ball.magnitude() < 1.0 {
        return final_target;
    }
    let alignment = car_to_ball.dot(shot_direction).clamp(-1.0, 1.0);
    if alignment >= 0.88 {
        return final_target;
    }
    let lateral = shot_direction.cross(Vector::new(0.0, 0.0, 1.0)).flatten().normalize();
    if lateral.magnitude() < 1.0 {
        return final_target;
    }
    let side_value = (car.location - ball).dot(lateral);
    let car_side = if side_value < 0.0 { -1.0 } else { 1.0 };
    let time_blend = ((time_left - 0.25) / 0.9).clamp(0.0, 1.0);
    let alignment_blend = ((0.88 - alignment) / 1.35).clamp(0.0, 1.0);
    let radius = 700.0 * time_blend * alignment_blend;
    if radius < 80.0 {
        final_target
    } else {
        final_target + lateral * (car_side * radius)
    }
}

pub fn hitbox_center_target(
    car: CarState,
    ball: Vector,
    shot_direction: Vector,
    standoff: f32,
    contact_drop: f32,
) -> Vector {
    let forward = car.forward().normalize();
    let right = car.right().normalize();
    let up = car.up().normalize();
    let hitbox_offset =
        forward * car.hitbox.offset.x
        + right * car.hitbox.offset.y
        + up * car.hitbox.offset.z;
    let mut target = ball - shot_direction.normalize() * standoff - hitbox_offset;
    target.z -= contact_drop;
    target
}


/// Select the jump-shot front-face contact exactly as Volt's Python routine.
///
/// The hitbox is evaluated at a short predicted impact orientation. Each
/// candidate is scored by outgoing shot alignment, front-face alignment, and
/// how far the required car centre is from the predicted car centre.
pub fn select_jump_front_contact(
    car: CarState,
    ball: Vector,
    shot_direction: Vector,
    time_remaining: f32,
    jumping: bool,
    previous_label: Option<&'static str>,
    gravity: Vector,
) -> Option<(Vector, &'static str)> {
    fn rotate_axis_angle(vector: Vector, axis: Vector, angle: f32) -> Vector {
        let axis = axis.normalize();
        if axis.magnitude() < 0.001 || angle.abs() < 1.0e-6 {
            return vector;
        }
        let (sine, cosine) = angle.sin_cos();
        vector * cosine
            + axis.cross(vector) * sine
            + axis * (axis.dot(vector) * (1.0 - cosine))
    }

    let shot_forward = shot_direction.flatten().normalize();
    let car_forward = car.forward().normalize();
    let car_up = car.up().normalize();
    if shot_forward.magnitude() < 0.001
        || car_forward.magnitude() < 0.001
        || car_up.magnitude() < 0.001
    {
        return None;
    }

    let car_right = car_forward.cross(car_up).normalize();
    if car_right.magnitude() < 0.001 {
        return None;
    }
    let car_up = car_right.cross(car_forward).normalize();

    let orientation_horizon = if jumping {
        time_remaining.max(0.0).min(0.18)
    } else {
        0.0
    };
    let angular_speed = car.angular_velocity.magnitude();
    let mut impact_forward = car_forward;
    let mut impact_up = car_up;
    if angular_speed > 0.05 && orientation_horizon > 0.0 {
        let axis = car.angular_velocity / angular_speed;
        let angle = (angular_speed * orientation_horizon).min(0.65);
        impact_forward = rotate_axis_angle(car_forward, axis, angle).normalize();
        impact_up = rotate_axis_angle(car_up, axis, angle).normalize();
    }

    let impact_right = impact_up.cross(impact_forward).normalize();
    let impact_up = impact_forward.cross(impact_right).normalize();
    let half_length = car.hitbox.length * 0.5;
    let half_width = car.hitbox.width * 0.5;
    let half_height = car.hitbox.height * 0.5;
    let world_hitbox_offset =
        impact_forward * car.hitbox.offset.x
        + impact_right * car.hitbox.offset.y
        + impact_up * car.hitbox.offset.z;

    let candidates: [(&'static str, f32, f32); 7] = [
        ("centre", 0.0, 0.0),
        ("top_left", -half_width * 0.82, half_height * 0.82),
        ("top_right", half_width * 0.82, half_height * 0.82),
        ("left", -half_width * 0.88, 0.0),
        ("right", half_width * 0.88, 0.0),
        ("top", 0.0, half_height * 0.88),
        ("bottom", 0.0, -half_height * 0.70),
    ];

    let impact_horizon = time_remaining.max(0.0).min(0.30);
    let predicted_car = car.location
        + car.velocity * impact_horizon
        + gravity * (0.5 * impact_horizon * impact_horizon);

    let mut scored: Vec<(f32, &'static str, Vector)> = Vec::with_capacity(7);
    for (label, lateral_offset, vertical_offset) in candidates {
        let front_point_offset = impact_forward * half_length
            + impact_right * lateral_offset
            + impact_up * vertical_offset
            + world_hitbox_offset;
        let predicted_contact_point = predicted_car + front_point_offset;
        let incoming_normal = (ball - predicted_contact_point).normalize();
        if incoming_normal.magnitude() < 0.001 {
            continue;
        }

        let shot_alignment = incoming_normal.dot(shot_forward);
        let front_alignment = incoming_normal.dot(impact_forward);
        let required_car_centre =
            ball - shot_forward * BALL_RADIUS - front_point_offset;
        let reach_cost = required_car_centre.distance(predicted_car);
        let score = shot_alignment
            + front_alignment.clamp(-1.0, 1.0) * 0.12
            - reach_cost * 0.00020;
        scored.push((score, label, required_car_centre));
    }

    let best = scored.iter().max_by(|a, b| a.0.total_cmp(&b.0))?;
    if let Some(previous) = previous_label {
        if let Some(old) = scored.iter().find(|candidate| candidate.1 == previous) {
            if old.0 >= best.0 - 0.06 {
                return Some((old.2, old.1));
            }
        }
    }
    Some((best.2, best.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn car() -> CarState {
        CarState {
            index: 0,
            team: 0,
            location: Vector::ZERO,
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
            hitbox: Hitbox {
                length: 118.0,
                width: 84.0,
                height: 36.16,
                offset: Vector::new(13.0, 2.0, 4.0),
            },
        }
    }

    #[test]
    fn shared_hitbox_target_accounts_for_offset_and_aerial_drop() {
        let me = car();
        let target = hitbox_center_target(
            me, Vector::new(1000.0, 0.0, 300.0),
            Vector::new(1.0, 0.0, 0.0), 151.75, 46.375,
        );
        assert!((target.x - (1000.0 - 151.75 - 13.0)).abs() < 1.0e-4);
        assert!((target.y + 2.0).abs() < 1.0e-4);
        assert!((target.z - (300.0 - 4.0 - 46.375)).abs() < 1.0e-4);
    }
}
