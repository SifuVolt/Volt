use crate::math::Vector;

/// Rocket League movement constants used by Volt's Python routines.
///
/// These are the standard-mutator values. Gravity is supplied from the live
/// match packet so low/high/super-high gravity mutators are respected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsConfig {
    pub gravity: Vector,
    pub max_speed: f32,
    pub throttle_accel: f32,
    pub boost_accel: f32,
    pub boost_consumption: f32,
    pub jump_impulse: f32,
    pub jump_accel: f32,
    pub jump_hold_duration: f32,
    pub timestep: f32,
    pub ball_radius: f32,
}

impl PhysicsConfig {
    pub const STANDARD_BOOST_ACCEL: f32 = 991.0 + (2.0 / 3.0);

    pub fn from_world_gravity(gravity_z: f32) -> Self {
        Self {
            gravity: Vector::new(0.0, 0.0, gravity_z),
            max_speed: 2300.0,
            throttle_accel: 66.0 + (2.0 / 3.0),
            boost_accel: Self::STANDARD_BOOST_ACCEL,
            boost_consumption: 33.0 + (1.0 / 3.0),
            jump_impulse: 291.0 + (2.0 / 3.0),
            jump_accel: 1458.0 + (1.0 / 3.0),
            jump_hold_duration: 0.2,
            timestep: 1.0 / 120.0,
            ball_radius: 91.25,
        }
    }

    /// Construct the same physics profile with a match boost-strength
    /// multiplier (1.0, 1.5, 2.0, 5.0, or 10.0 in Volt's Python agent).
    pub fn with_boost_multiplier(mut self, multiplier: f32) -> Self {
        self.boost_accel = Self::STANDARD_BOOST_ACCEL * multiplier.max(0.0);
        self
    }
}

/// Jump inputs for the simple first/second-jump ballistic model.
///
/// This deliberately does not model dodge impulses, air-control torque,
/// boost thrust, or collisions. Those belong in later vehicle simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JumpPlan {
    pub hold_duration: f32,
    pub second_jump_at: Option<f32>,
}

impl Default for JumpPlan {
    fn default() -> Self {
        Self {
            hold_duration: 0.2,
            second_jump_at: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MotionState {
    pub position: Vector,
    pub velocity: Vector,
}

/// Predict a car's ballistic position after a jump.
///
/// The first jump impulse is applied at t=0. Jump acceleration is applied
/// along the car's launch-up axis while jump is held, up to the game's
/// 0.2-second hold limit. An optional second jump impulse can be scheduled.
/// The fixed-step integration uses Volt's 120 Hz simulation timestep.
pub fn predict_jump(
    position: Vector,
    velocity: Vector,
    launch_up: Vector,
    config: PhysicsConfig,
    plan: JumpPlan,
    duration: f32,
) -> MotionState {
    let mut state = MotionState { position, velocity };
    if duration <= 0.0 {
        return state;
    }

    let up = launch_up.normalize();
    if up == Vector::ZERO {
        return state;
    }

    state.velocity = state.velocity + up * config.jump_impulse;

    let hold_duration = plan
        .hold_duration
        .clamp(0.0, config.jump_hold_duration);
    let second_jump_at = plan
        .second_jump_at
        .filter(|time| *time >= 0.0 && *time <= duration);

    let mut elapsed = 0.0;
    let mut second_jump_applied = false;

    while elapsed < duration {
        if !second_jump_applied {
            if let Some(jump_time) = second_jump_at {
                if elapsed >= jump_time {
                    state.velocity = state.velocity + up * config.jump_impulse;
                    second_jump_applied = true;
                }
            }
        }

        let mut step = config.timestep.min(duration - elapsed);

        if elapsed < hold_duration {
            step = step.min(hold_duration - elapsed);
        }

        if !second_jump_applied {
            if let Some(jump_time) = second_jump_at {
                if jump_time > elapsed {
                    step = step.min(jump_time - elapsed);
                }
            }
        }

        if step <= f32::EPSILON {
            // Process an event at the current instant on the next iteration.
            if !second_jump_applied
                && second_jump_at.is_some_and(|jump_time| elapsed >= jump_time)
            {
                continue;
            }
            break;
        }

        let jump_acceleration = if elapsed < hold_duration {
            up * config.jump_accel
        } else {
            Vector::ZERO
        };
        let acceleration = config.gravity + jump_acceleration;

        state.position = state.position
            + state.velocity * step
            + acceleration * (0.5 * step * step);
        state.velocity = state.velocity + acceleration * step;
        elapsed += step;
    }

    // A second jump exactly at the requested endpoint changes velocity but
    // cannot change the position at that same instant.
    if !second_jump_applied
        && second_jump_at.is_some_and(|jump_time| (jump_time - duration).abs() <= 1.0e-6)
    {
        state.velocity = state.velocity + up * config.jump_impulse;
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_profile_matches_volt_python_constants() {
        let physics = PhysicsConfig::from_world_gravity(-650.0);

        assert_eq!(physics.max_speed, 2300.0);
        assert!((physics.throttle_accel - 66.666_67).abs() < 0.001);
        assert!((physics.boost_accel - 991.666_7).abs() < 0.001);
        assert!((physics.boost_consumption - 33.333_33).abs() < 0.001);
        assert!((physics.jump_impulse - 291.666_66).abs() < 0.001);
        assert!((physics.jump_accel - 1458.333_4).abs() < 0.001);
        assert_eq!(physics.gravity, Vector::new(0.0, 0.0, -650.0));
    }

    #[test]
    fn jump_prediction_respects_gravity_and_hold_duration() {
        let physics = PhysicsConfig::from_world_gravity(-650.0);
        let result = predict_jump(
            Vector::ZERO,
            Vector::ZERO,
            Vector::new(0.0, 0.0, 1.0),
            physics,
            JumpPlan::default(),
            0.2,
        );

        let expected_height = physics.jump_impulse * 0.2
            + 0.5 * (physics.jump_accel - 650.0) * 0.2_f32.powi(2);

        assert!((result.position.z - expected_height).abs() < 0.01);
        assert!((result.velocity.z
            - (physics.jump_impulse + (physics.jump_accel - 650.0) * 0.2))
            .abs()
            < 0.01);
    }

    #[test]
    fn second_jump_adds_impulse_at_its_scheduled_time() {
        let physics = PhysicsConfig::from_world_gravity(-650.0);
        let initial = Vector::ZERO;
        let up = Vector::new(0.0, 0.0, 1.0);
        let one_jump = predict_jump(
            initial,
            Vector::ZERO,
            up,
            physics,
            JumpPlan {
                hold_duration: 0.2,
                second_jump_at: None,
            },
            0.3,
        );
        let two_jumps = predict_jump(
            initial,
            Vector::ZERO,
            up,
            physics,
            JumpPlan {
                hold_duration: 0.2,
                second_jump_at: Some(0.2),
            },
            0.3,
        );

        assert!((two_jumps.velocity.z - one_jump.velocity.z - physics.jump_impulse).abs() < 0.01);
        assert!(
            (two_jumps.position.z - one_jump.position.z - physics.jump_impulse * 0.1).abs()
                < 0.01
        );
    }

    #[test]
    fn zero_duration_does_not_apply_a_jump() {
        let physics = PhysicsConfig::from_world_gravity(-650.0);
        let initial_position = Vector::new(10.0, 20.0, 30.0);
        let initial_velocity = Vector::new(100.0, 200.0, 300.0);
        let result = predict_jump(
            initial_position,
            initial_velocity,
            Vector::new(0.0, 0.0, 1.0),
            physics,
            JumpPlan::default(),
            0.0,
        );

        assert_eq!(result.position, initial_position);
        assert_eq!(result.velocity, initial_velocity);
    }
}
