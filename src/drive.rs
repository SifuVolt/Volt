//! Ground and airborne vehicle control.
//!
//! Ground steering/throttle use the tested Volt defaultPD/defaultThrottle
//! ports. This layer only chooses a speed profile and a safe, field-clamped
//! waypoint; it does not replace those lower-level calculations.

use crate::{
    control::{default_drive, default_pd, DriveInput},
    math::{utils::cap_in_field, Vector},
    physics::drive::dodge_impulse,
    state::CarState,
};
use rlbot::flat::ControllerState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveRequest {
    pub target: Vector,
    pub target_speed: f32,
    pub boost_accel: f32,
    pub allow_handbrake: bool,
    /// Keep forward momentum for a shot approach instead of braking at the
    /// waypoint. The shot executor uses this for its final contact run.
    pub preserve_momentum: bool,
}

impl DriveRequest {
    pub fn toward(target: Vector, boost_accel: f32) -> Self {
        Self {
            target,
            target_speed: 2300.0,
            boost_accel,
            allow_handbrake: true,
            preserve_momentum: false,
        }
    }
}

fn speed_for_heading(angle: f32) -> f32 {
    let angle = angle.abs();
    if angle > 2.6 {
        650.0
    } else if angle > 2.0 {
        850.0
    } else if angle > 1.35 {
        1150.0
    } else if angle > 0.85 {
        1550.0
    } else {
        2300.0
    }
}

/// Volt-style speed-aware ground drive. A shot approach can request
/// preserve_momentum so the car accelerates through the contact window.
pub fn drive_to(car: &CarState, request: DriveRequest) -> ControllerState {
    let target = cap_in_field(car, request.target);
    let local_target = car.local_location(target);
    let distance = local_target.flatten().magnitude();
    let heading = local_target.y.atan2(local_target.x);

    let mut target_speed = request.target_speed.clamp(-2300.0, 2300.0);
    target_speed = target_speed.min(speed_for_heading(heading));

    if !request.preserve_momentum && distance > 160.0 {
        // Braking envelope begins before the waypoint so navigation does not
        // oscillate around small support/defensive targets.
        let braking_distance = (distance - 150.0).max(0.0);
        let braking_speed = (2.0 * 2800.0 * braking_distance).sqrt();
        target_speed = target_speed.min(braking_speed.max(450.0));
    } else if request.preserve_momentum {
        target_speed = target_speed.max(1550.0);
    }

    let speed = car.velocity.magnitude();
    let sharp_turn = heading.abs() > 1.72;
    let handbrake = request.allow_handbrake
        && sharp_turn
        && distance > 280.0
        && speed > 650.0
        && speed < 1750.0;

    let output = default_drive(
        car,
        DriveInput {
            target_speed,
            local_target,
            upside_down: false,
            up: None,
            handbrake,
            boost_accel: request.boost_accel,
        },
    );

    let mut controller = ControllerState::default();
    controller.throttle = output.throttle.unwrap_or(1.0);
    controller.steer = output.controls.steer;
    controller.pitch = output.controls.pitch;
    controller.yaw = output.controls.yaw;
    controller.roll = output.controls.roll;
    controller.handbrake = output.handbrake.unwrap_or(false);
    controller.boost = output.boost.unwrap_or(false);
    controller.jump = false;

    if request.preserve_momentum && distance < 420.0 {
        // Do not let target arrival turn the final contact frames into a
        // brake pulse. The contact routine owns the final jump/dodge timing.
        if car.local_velocity().x < 2050.0 && !controller.handbrake {
            controller.throttle = 1.0;
            if request.boost_accel > 0.0
                && car.boost > 0.0
                && heading.abs() < 0.38
                && car.velocity.magnitude() < 2200.0
            {
                controller.boost = true;
            }
        }
    }

    controller
}

/// Basic air recovery: align the nose with the travel direction and the roof
/// with world-up. Boost is used only when the nose is already reasonably
/// aligned, avoiding the old habit of boosting while pointing away.
pub fn recover_in_air(car: &CarState, boost_accel: f32, target: Option<Vector>) -> ControllerState {
    let velocity_target = if car.velocity.magnitude() > 300.0 {
        car.velocity
    } else {
        target.unwrap_or(car.forward() * 100.0)
    };
    let local_forward = car.local(velocity_target);
    let local_up = car.local(Vector::new(0.0, 0.0, 1.0));
    let (_, controls) = default_pd(car, local_forward, false, Some(local_up));

    let mut controller = ControllerState::default();
    controller.pitch = controls.pitch;
    controller.yaw = controls.yaw;
    controller.roll = controls.roll;
    controller.steer = controls.steer;
    controller.throttle = 0.0;
    controller.boost = car.boost > 0.0
        && boost_accel > 0.0
        && local_forward.x > 0.0
        && local_forward.angle(Vector::new(1.0, 0.0, 0.0)) < 0.42;
    controller
}

/// A short-horizon speed estimate including the dodge impulse. Used for
/// contact timing and kickoff eligibility, not as a replacement for the
/// parser's 60 Hz reachability simulation.
pub fn effective_forward_speed(car: &CarState) -> f32 {
    car.local_velocity().x.max(0.0) + dodge_impulse(car.velocity, 2300.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn car(location: Vector, velocity: Vector) -> CarState {
        CarState {
            index: 0,
            team: 0,
            location,
            velocity,
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
            boost: 40.0,
            hitbox: Hitbox { length: 118.0, width: 84.0, height: 36.0, offset: Vector::ZERO },
        }
    }

    #[test]
    fn shot_approach_never_requests_reverse_or_stops_at_contact() {
        let me = car(Vector::ZERO, Vector::new(1700.0, 0.0, 0.0));
        let controller = drive_to(
            &me,
            DriveRequest {
                target: Vector::new(200.0, 0.0, 0.0),
                target_speed: 2300.0,
                boost_accel: 991.6667,
                allow_handbrake: false,
                preserve_momentum: true,
            },
        );
        assert!(controller.throttle >= 0.0);
        assert!(!controller.handbrake);
    }

    #[test]
    fn ordinary_navigation_uses_a_finite_controller() {
        let me = car(Vector::ZERO, Vector::ZERO);
        let controller = drive_to(
            &me,
            DriveRequest::toward(Vector::new(1500.0, 900.0, 0.0), 991.6667),
        );
        assert!(controller.throttle.is_finite());
        assert!(controller.steer.is_finite());
        assert!((-1.0..=1.0).contains(&controller.steer));
    }
}
