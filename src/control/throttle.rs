use crate::math::Vector;
use crate::physics::drive::{is_inside_turn_radius, throttle_acceleration, turn_radius};

use super::pd::OrientationAngles;

const BRAKE_ACC: f32 = 3500.0;
const COAST_ACC: f32 = 525.0;
const REACTION_TIME: f32 = 0.04;
const BRAKE_COAST_TRANSITION: f32 = -(0.45 * BRAKE_ACC + 0.55 * COAST_ACC);
const COASTING_THROTTLE_TRANSITION: f32 = -0.5 * COAST_ACC;
const MIN_WALL_SPEED: f32 = -0.5 * BRAKE_ACC;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrottleInput {
    pub airborne: bool,
    pub car_speed: f32,
    pub world_speed: f32,
    pub up_z: f32,
    pub target_speed: f32,
    pub target_angles: OrientationAngles,
    pub local_target: Option<Vector>,
    pub steer: f32,
    pub yaw: f32,
    pub handbrake: bool,
    pub boost_accel: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ThrottleOutput {
    pub car_speed: f32,
    pub target_speed: f32,
    pub throttle: Option<f32>,
    pub steer: Option<f32>,
    pub handbrake: Option<bool>,
    pub boost: Option<bool>,
}

fn sign(value: f32) -> f32 {
    if value < 0.0 { -1.0 } else if value > 0.0 { 1.0 } else { 0.0 }
}

/// Direct port of VirxERLU's defaultThrottle used by Volt's Python
/// defaultDrive. It deliberately preserves the source's brake/coast bands:
/// coasting leaves throttle untouched instead of forcing a speed correction.
pub fn default_throttle(input: ThrottleInput) -> ThrottleOutput {
    let mut output = ThrottleOutput {
        car_speed: input.car_speed,
        target_speed: input.target_speed,
        ..ThrottleOutput::default()
    };
    if input.airborne {
        return output;
    }

    let mut handbrake = input.handbrake;
    let mut steer = input.steer;
    if let Some(local_target) = input.local_target {
        let turn_rad = turn_radius(input.car_speed.abs());
        let inside_turn = if local_target.y.abs() < turn_rad || input.car_speed > 1410.0 {
            is_inside_turn_radius(turn_rad, local_target, sign(input.steer))
        } else {
            local_target.x.abs() < turn_rad
        };
        handbrake = input.world_speed > 600.0 && inside_turn;
    }

    let mut angle_to_target = input.target_angles.yaw.abs();
    if input.target_speed < 0.0 {
        angle_to_target = std::f32::consts::PI - angle_to_target;
    }

    if handbrake {
        if angle_to_target > 2.6 {
            steer = sign(input.steer);
            handbrake = false;
        } else {
            steer = input.yaw;
        }
    }

    let speed_error = input.target_speed - input.car_speed;
    let mut acceleration = speed_error / REACTION_TIME;
    if input.car_speed < 0.0 {
        acceleration *= -1.0;
    }

    let mut brake_transition = BRAKE_COAST_TRANSITION;
    let mut coast_transition = COASTING_THROTTLE_TRANSITION;
    if input.up_z < 0.7 {
        brake_transition = MIN_WALL_SPEED;
        coast_transition = MIN_WALL_SPEED;
    }

    let throttle_accel = throttle_acceleration(input.car_speed);
    let throttle_boost_transition = throttle_accel + 0.5 * input.boost_accel;

    if acceleration <= brake_transition {
        output.throttle = Some(-1.0);
    } else if acceleration < coast_transition {
        // Intentional coast band: Python leaves the controller throttle
        // unchanged (normally zero) rather than applying a weak brake.
    } else if acceleration <= throttle_boost_transition {
        output.throttle = Some(if throttle_accel == 0.0 {
            1.0
        } else {
            (acceleration / throttle_accel).clamp(0.02, 1.0)
        });
    } else {
        output.throttle = Some(1.0);
        if !handbrake && speed_error > 0.0 && angle_to_target < 1.0 {
            output.boost = Some(true);
        }
    }

    if input.car_speed < 0.0 {
        if let Some(throttle) = output.throttle.as_mut() {
            *throttle *= -1.0;
        }
    }

    output.target_speed = input.target_speed;
    output.handbrake = Some(handbrake);
    if steer != input.steer || (handbrake && input.steer != input.yaw) {
        output.steer = Some(steer);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> ThrottleInput {
        ThrottleInput {
            airborne: false,
            car_speed: 0.0,
            world_speed: 0.0,
            up_z: 1.0,
            target_speed: 1000.0,
            target_angles: OrientationAngles::default(),
            local_target: Some(Vector::new(2000.0, 0.0, 0.0)),
            steer: 0.0,
            yaw: 0.0,
            handbrake: false,
            boost_accel: 991.6667,
        }
    }

    #[test]
    fn airborne_leaves_controller_fields_untouched() {
        let mut args = input();
        args.airborne = true;
        let result = default_throttle(args);
        assert_eq!(result.throttle, None);
        assert_eq!(result.handbrake, None);
        assert_eq!(result.boost, None);
    }

    #[test]
    fn acceleration_uses_python_reaction_time_and_throttle_curve() {
        let result = default_throttle(input());
        let expected = ((1000.0 / REACTION_TIME) / throttle_acceleration(0.0)).clamp(0.02, 1.0);
        assert_eq!(result.throttle, Some(expected));
    }

    #[test]
    fn python_coast_band_leaves_throttle_untouched() {
        let mut args = input();
        args.car_speed = 1000.0;
        args.target_speed = 990.0;
        let result = default_throttle(args);
        assert_eq!(result.throttle, None);
    }

    #[test]
    fn strong_braking_uses_full_reverse_throttle() {
        let mut args = input();
        args.car_speed = 1000.0;
        args.target_speed = 800.0;
        let result = default_throttle(args);
        assert_eq!(result.throttle, Some(-1.0));
    }

    #[test]
    fn aligned_large_acceleration_uses_python_boost_branch() {
        let mut args = input();
        args.target_speed = 2200.0;
        let result = default_throttle(args);
        assert_eq!(result.throttle, Some(1.0));
        assert_eq!(result.boost, Some(true));
    }

    #[test]
    fn boost_is_withheld_when_not_facing_target() {
        let mut args = input();
        args.target_speed = 2200.0;
        args.target_angles.yaw = 1.1;
        let result = default_throttle(args);
        assert_eq!(result.boost, None);
        assert_eq!(result.throttle, Some(1.0));
    }

    #[test]
    fn reverse_motion_flips_throttle_sign() {
        let mut args = input();
        args.car_speed = -500.0;
        args.target_speed = -300.0;
        let result = default_throttle(args);
        assert!(result.throttle.unwrap() < 0.0);
    }

    #[test]
    fn handbrake_uses_python_world_speed_gate() {
        let mut args = input();
        args.car_speed = 650.0;
        args.world_speed = 650.0;
        args.local_target = Some(Vector::new(100.0, -100.0, 0.0));
        args.steer = 1.0;
        args.yaw = 0.4;
        args.target_angles.yaw = 0.2;
        let result = default_throttle(args);
        assert_eq!(result.handbrake, Some(true));
        assert_eq!(result.steer, Some(0.4));
    }

    #[test]
    fn wall_drive_uses_wall_brake_threshold() {
        let mut args = input();
        args.up_z = 0.0;
        args.car_speed = 1000.0;
        args.target_speed = 900.0;
        let result = default_throttle(args);
        assert_eq!(result.throttle, Some(0.02));
    }
}
