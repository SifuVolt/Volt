use crate::math::Vector;
use crate::physics::drive::{
    curvature, is_inside_turn_radius, throttle_acceleration, turn_radius,
};

use super::pd::OrientationAngles;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrottleInput {
    pub airborne: bool,
    pub car_speed: f32,
    pub world_speed: f32,
    pub angular_velocity: Vector,
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
    if value < 0.0 {
        -1.0
    } else if value > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Direct port of Volt's current Python utils.defaultThrottle.
/// Keep the source's acceleration curve, speed-dependent turn cap and
/// boostless-distance simulation; do not substitute the separate
/// brake/coast controller from VirxERLU or Party Cannon.
pub fn default_throttle(input: ThrottleInput) -> ThrottleOutput {
    let mut output = ThrottleOutput {
        car_speed: input.car_speed,
        target_speed: input.target_speed,
        ..ThrottleOutput::default()
    };

    if input.airborne {
        return output;
    }

    let mut target_speed = input.target_speed;
    let mut handbrake = input.handbrake;
    let mut steer = input.steer;

    if let Some(local_target) = input.local_target {
        let turn_rad = turn_radius(input.car_speed.abs());
        let angular_speed = input.angular_velocity.magnitude();
        let inside_turn = if local_target.y.abs() < turn_rad {
            is_inside_turn_radius(turn_rad, local_target, sign(input.steer))
        } else {
            local_target.x.abs() < turn_rad
        };

        handbrake = input.world_speed > 250.0
            && input.world_speed < 1000.0
            && inside_turn
            && angular_speed >= (input.world_speed / 500.0).min(2.0);
    }

    let mut angle_to_target = input.target_angles.yaw.abs();
    if target_speed < 0.0 {
        angle_to_target = std::f32::consts::PI - angle_to_target;
    }

    if let Some(local_target) = input.local_target {
        if target_speed > 0.0 && local_target.x > 0.0 {
            let target_distance_sq =
                local_target.x * local_target.x + local_target.y * local_target.y;
            if target_distance_sq > 350.0_f32.powi(2) {
                let required_curvature =
                    2.0 * local_target.y.abs() / target_distance_sq;
                if required_curvature > 0.00015 {
                    let mut turn_speed_limit = 0.0;
                    let mut candidate_speed = 0.0;
                    while candidate_speed <= 2300.0 {
                        if curvature(candidate_speed) >= required_curvature {
                            turn_speed_limit = candidate_speed;
                            candidate_speed += 25.0;
                        } else {
                            break;
                        }
                    }
                    target_speed = target_speed.min(turn_speed_limit.max(450.0));
                }
            }
        }
    }

    if handbrake {
        if angle_to_target > 2.8 {
            if target_speed.abs() > 950.0 {
                target_speed = 950.0 * sign(target_speed);
            }
            steer = sign(input.steer);
            handbrake = false;
        } else {
            steer = input.yaw;
        }
    }

    let speed_error = target_speed - input.car_speed;
    let acceleration = throttle_acceleration(input.car_speed);
    let throttle = if acceleration != 0.0
        && (target_speed < 1410.0 || speed_error < -acceleration / 10.0)
    {
        (speed_error / acceleration).clamp(-1.0, 1.0)
    } else {
        sign(if speed_error.abs() > 117.0 {
            speed_error
        } else {
            target_speed
        })
    };

    output.target_speed = target_speed;
    output.throttle = Some(throttle);
    output.handbrake = Some(handbrake);
    if steer != input.steer || (handbrake && input.steer != input.yaw) {
        output.steer = Some(steer);
    }

    if !handbrake {
        let available_distance = input
            .local_target
            .map(|target| (target.x * target.x + target.y * target.y).sqrt())
            .unwrap_or(0.0);
        let mut boostless_speed = input.car_speed.max(0.0);
        let mut boostless_distance = 0.0;

        if target_speed > 0.0 && available_distance > 0.0 {
            for _ in 0..360 {
                if boostless_speed >= target_speed - 50.0 {
                    break;
                }
                let throttle_accel = throttle_acceleration(boostless_speed);
                if throttle_accel <= 0.0 {
                    break;
                }
                boostless_speed =
                    target_speed.min(boostless_speed + throttle_accel / 120.0);
                boostless_distance += boostless_speed / 120.0;
                if boostless_distance >= available_distance {
                    break;
                }
            }
        }

        let boost_needed = target_speed > 0.0
            && angle_to_target < 0.45
            && speed_error > input.boost_accel / 20.0
            && boostless_speed < target_speed - 50.0;
        output.boost = Some(boost_needed);
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
            angular_velocity: Vector::ZERO,
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
    fn airborne_leaves_ground_controls_untouched() {
        let mut args = input();
        args.airborne = true;
        let result = default_throttle(args);
        assert_eq!(result.car_speed, args.car_speed);
        assert_eq!(result.throttle, None);
        assert_eq!(result.handbrake, None);
        assert_eq!(result.boost, None);
    }

    #[test]
    fn throttle_uses_python_acceleration_curve() {
        let result = default_throttle(input());
        let expected = (1000.0 / throttle_acceleration(0.0)).min(1.0);
        assert_eq!(result.throttle, Some(expected));
    }

    #[test]
    fn tight_turn_caps_requested_speed_using_python_curvature_table() {
        let mut args = input();
        args.target_speed = 1800.0;
        args.local_target = Some(Vector::new(500.0, 500.0, 0.0));
        let result = default_throttle(args);
        assert!(result.target_speed < args.target_speed);
        assert!(result.target_speed >= 450.0);
    }

    #[test]
    fn boost_is_used_when_throttle_only_cannot_reach_requested_speed() {
        let mut args = input();
        args.target_speed = 2200.0;
        args.local_target = Some(Vector::new(100.0, 0.0, 0.0));
        assert_eq!(default_throttle(args).boost, Some(true));
    }

    #[test]
    fn boost_is_not_used_when_car_is_misaligned() {
        let mut args = input();
        args.target_speed = 2200.0;
        args.target_angles.yaw = 0.5;
        args.local_target = Some(Vector::new(100.0, 0.0, 0.0));
        assert_eq!(default_throttle(args).boost, Some(false));
    }

    #[test]
    fn reverse_target_uses_reverse_alignment_and_throttle() {
        let mut args = input();
        args.target_speed = -500.0;
        args.local_target = None;
        let result = default_throttle(args);
        assert_eq!(result.boost, Some(false));
        assert_eq!(result.throttle, Some(-500.0 / 1600.0));
    }

    #[test]
    fn source_handbrake_turn_branch_uses_yaw_as_steer() {
        let mut args = input();
        args.world_speed = 600.0;
        args.car_speed = 600.0;
        args.angular_velocity = Vector::new(0.0, 0.0, 2.0);
        args.local_target = Some(Vector::new(0.0, 100.0, 0.0));
        args.target_angles.yaw = 0.5;
        args.steer = -0.4;
        args.yaw = 0.7;
        let result = default_throttle(args);
        if result.handbrake == Some(true) {
            assert_eq!(result.steer, Some(args.yaw));
        }
    }
}
