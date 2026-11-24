use crate::math::Vector;
use crate::state::CarState;

use super::pd::{default_pd, OrientationAngles, OrientationControls};
use super::throttle::{default_throttle, ThrottleInput};

/// Inputs to Volt's Python defaultDrive routine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveInput {
    pub target_speed: f32,
    pub local_target: Vector,
    pub upside_down: bool,
    pub up: Option<Vector>,
    pub handbrake: bool,
    pub boost_accel: f32,
}

/// Combined outputs from defaultPD followed by defaultThrottle.
///
/// The orientation controls are always populated by defaultPD. Ground-drive
/// controls are optional because Python defaultThrottle leaves them untouched
/// while airborne.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DriveOutput {
    pub target_angles: OrientationAngles,
    pub car_speed: f32,
    pub target_speed: f32,
    pub controls: OrientationControls,
    pub throttle: Option<f32>,
    pub handbrake: Option<bool>,
    pub boost: Option<bool>,
}

/// Pure Rust composition matching Volt's Python defaultDrive:
/// defaultPD(agent, local_target), then
/// defaultThrottle(agent, target_speed, target_angles, local_target).
///
/// This function does not mutate CarState or an RLBot controller.
pub fn default_drive(car: &CarState, input: DriveInput) -> DriveOutput {
    let (target_angles, mut controls) =
        default_pd(car, input.local_target, input.upside_down, input.up);

    let throttle = default_throttle(ThrottleInput {
        airborne: car.airborne,
        car_speed: car.local_velocity().x,
        world_speed: car.velocity.magnitude(),
        angular_velocity: car.angular_velocity,
        target_speed: input.target_speed,
        target_angles,
        local_target: Some(input.local_target),
        steer: controls.steer,
        yaw: controls.yaw,
        handbrake: input.handbrake,
        boost_accel: input.boost_accel,
    });

    if let Some(steer) = throttle.steer {
        controls.steer = steer;
    }

    DriveOutput {
        target_angles,
        car_speed: throttle.car_speed,
        target_speed: throttle.target_speed,
        controls,
        throttle: throttle.throttle,
        handbrake: throttle.handbrake,
        boost: throttle.boost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn test_car(airborne: bool, velocity: Vector) -> CarState {
        CarState {
            index: 0,
            team: 0,
            location: Vector::ZERO,
            velocity,
            orientation: Matrix3::new(0.0, 0.0, 0.0),
            angular_velocity: Vector::ZERO,
            demolished: false,
            airborne,
            supersonic: false,
            jumped: airborne,
            doublejumped: false,
            air_state: if airborne {
                AirState::InAir
            } else {
                AirState::OnGround
            },
            has_jumped: airborne,
            has_double_jumped: false,
            has_dodged: false,
            dodge_timeout: 0.0,
            dodge_elapsed: 0.0,
            dodge_dir: Vector::ZERO,
            flip_reset_available: false,
            boost: 50.0,
            hitbox: Hitbox::default(),
        }
    }

    fn drive_input() -> DriveInput {
        DriveInput {
            target_speed: 1200.0,
            local_target: Vector::new(1500.0, 200.0, 0.0),
            upside_down: false,
            up: None,
            handbrake: false,
            boost_accel: 991.6667,
        }
    }

    #[test]
    fn default_drive_runs_pd_then_throttle() {
        let car = test_car(false, Vector::new(300.0, 0.0, 0.0));
        let input = drive_input();
        let output = default_drive(&car, input);
        let (expected_angles, expected_controls) =
            default_pd(&car, input.local_target, input.upside_down, input.up);

        assert_eq!(output.target_angles, expected_angles);
        assert_eq!(output.controls.pitch, expected_controls.pitch);
        assert_eq!(output.controls.yaw, expected_controls.yaw);
        assert_eq!(output.controls.roll, expected_controls.roll);
        assert_eq!(output.car_speed, 300.0);
        assert!(output.throttle.is_some());
        assert!(output.handbrake.is_some());
        assert!(output.boost.is_some());
    }

    #[test]
    fn default_drive_keeps_pd_controls_when_airborne() {
        let car = test_car(true, Vector::new(300.0, 0.0, 0.0));
        let input = drive_input();
        let output = default_drive(&car, input);
        let (_, expected_controls) =
            default_pd(&car, input.local_target, input.upside_down, input.up);

        assert_eq!(output.controls, expected_controls);
        assert_eq!(output.throttle, None);
        assert_eq!(output.handbrake, None);
        assert_eq!(output.boost, None);
    }

    #[test]
    fn default_drive_reports_local_forward_speed() {
        let car = test_car(false, Vector::new(400.0, 250.0, 0.0));
        let output = default_drive(&car, drive_input());

        assert!((output.car_speed - 400.0).abs() < 1e-6);
    }
}
