//! Shared hitbox-aware ball-contact geometry for ground, jump and aerial shots.
//!
//! The contact search is the common seven-point front-face model used by
//! Party Cannon/Volt. Callers choose the prediction horizon and scoring
//! weights for their motion phase; the hitbox and contact-point math stays
//! identical across routines.

use crate::math::Vector;
use crate::state::CarState;

pub const BALL_RADIUS: f32 = 92.75;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontContact {
    pub approach: Vector,
    pub normal: Vector,
    pub quality: f32,
    pub contact_point: Vector,
    pub face: &'static str,
}

fn rotate_axis(vector: Vector, axis: Vector, angle: f32) -> Vector {
    let axis = axis.normalize();
    let cosine = angle.cos();
    let sine = angle.sin();
    vector * cosine
        + axis.cross(vector) * sine
        + axis * (axis.dot(vector) * (1.0 - cosine))
}

/// Shared seven-point front-hitbox search. Callers choose the prediction
/// horizons, closing-speed weight and flat-versus-3D reach cost for their phase.
pub fn select_front_contact(
    car: CarState,
    ball: Vector,
    ball_velocity: Vector,
    shot_direction: Vector,
    gravity: Vector,
    orientation_horizon: f32,
    impact_horizon: f32,
    closing_weight: f32,
    flat_reach_cost: bool,
    previous_face: Option<&'static str>,
) -> Option<FrontContact> {
    let shot_forward = shot_direction.flatten().normalize();
    let car_forward = car.forward().normalize();
    let car_up = car.up().normalize();
    if shot_forward.magnitude() <= 0.001
        || car_forward.magnitude() <= 0.001
        || car_up.magnitude() <= 0.001
    {
        return None;
    }

    let omega = car.world_angular_velocity();
    let angular_speed = omega.magnitude();
    let mut impact_forward = car_forward;
    let mut impact_up = car_up;
    if angular_speed > 0.05 && orientation_horizon > 0.0 {
        let angle = (angular_speed * orientation_horizon).min(0.65);
        impact_forward = rotate_axis(impact_forward, omega, angle).normalize();
        impact_up = rotate_axis(impact_up, omega, angle).normalize();
    }

    let impact_right = impact_up.cross(impact_forward).normalize();
    impact_up = impact_forward.cross(impact_right).normalize();
    if impact_forward.magnitude() <= 0.001 || impact_right.magnitude() <= 0.001 {
        return None;
    }

    let half_length = car.hitbox.length.max(80.0) * 0.5;
    let half_width = car.hitbox.width.max(60.0) * 0.5;
    let half_height = car.hitbox.height.max(24.0) * 0.5;
    let hitbox_offset = impact_forward * car.hitbox.offset.x
        + impact_right * car.hitbox.offset.y
        + impact_up * car.hitbox.offset.z;
    let faces: [(&str, f32, f32); 7] = [
        ("centre", 0.0, 0.0),
        ("top_left", -half_width * 0.82, half_height * 0.82),
        ("top_right", half_width * 0.82, half_height * 0.82),
        ("left", -half_width * 0.88, 0.0),
        ("right", half_width * 0.88, 0.0),
        ("top", 0.0, half_height * 0.88),
        ("bottom", 0.0, -half_height * 0.70),
    ];

    let predicted_car = car.location
        + car.velocity * impact_horizon
        + gravity * (0.5 * impact_horizon * impact_horizon);
    let relative_velocity = (car.velocity - ball_velocity).flatten().normalize();
    let mut scored: Vec<(f32, &'static str, Vector, Vector)> =
        Vec::with_capacity(faces.len());

    for (label, lateral, vertical) in faces {
        let front_offset = impact_forward * half_length
            + impact_right * lateral
            + impact_up * vertical
            + hitbox_offset;
        let predicted_contact = predicted_car + front_offset;
        let incoming_normal = (ball - predicted_contact).normalize();
        if incoming_normal.magnitude() <= 0.001 {
            continue;
        }

        let shot_alignment = incoming_normal.dot(shot_forward);
        let front_alignment = incoming_normal.dot(impact_forward);
        let required_car_centre = ball - shot_forward * BALL_RADIUS - front_offset;
        let reach_cost = if flat_reach_cost {
            required_car_centre.flat_distance(predicted_car)
        } else {
            required_car_centre.distance(predicted_car)
        };
        let closing_alignment = if relative_velocity.magnitude() > 0.001 {
            relative_velocity.dot(incoming_normal).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let score = shot_alignment
            + front_alignment.clamp(-1.0, 1.0) * 0.12
            + closing_alignment * closing_weight
            - reach_cost * 0.00020;
        scored.push((score, label, required_car_centre, incoming_normal));
    }

    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut selected = scored[0];
    if let Some(previous) = previous_face {
        if let Some(candidate) = scored.iter().copied().find(|entry| entry.1 == previous) {
            if candidate.0 >= selected.0 - 0.06 {
                selected = candidate;
            }
        }
    }

    Some(FrontContact {
        approach: selected.2,
        normal: selected.3,
        quality: ((selected.0 + 0.25) / 1.25).clamp(0.0, 1.0),
        contact_point: ball - shot_forward * BALL_RADIUS,
        face: selected.1,
    })
}

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
    fn shared_front_contact_targets_actual_hitbox_front() {
        let me = car();
        let ball = Vector::new(500.0, 0.0, 93.0);
        let contact = select_front_contact(
            me, ball, Vector::ZERO, Vector::new(1.0, 0.0, 0.0),
            Vector::ZERO, 0.0, 0.0, 0.0, false, None,
        ).unwrap();
        assert_eq!(contact.contact_point, Vector::new(407.25, 0.0, 93.0));
        assert!(contact.approach.x < ball.x - BALL_RADIUS);
        assert!(contact.normal.magnitude() > 0.99);
        assert!((0.0..=1.0).contains(&contact.quality));
        assert!(["centre", "top_left", "top_right", "left", "right", "top", "bottom"]
            .contains(&contact.face));
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
