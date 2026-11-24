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
