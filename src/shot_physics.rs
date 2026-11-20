//! Source-faithful shot reachability physics port from VirxERLU-CLib.
//!
//! This module ports the low-level jump and aerial validation routines used by
//! Volt's Python shot parser. Values are evaluated in f64 to match the C
//! implementation's double precision; packet state is converted at the edge.
//! Candidate generation, goal-post correction and travel-path planning live in
//! the higher-level shot planner and are intentionally separate.

use crate::math::Vector;
use crate::state::CarState;

const SIMULATION_DT: f64 = 1.0 / 60.0;
const PHYSICS_DT: f64 = 1.0 / 120.0;
const JUMP_MAX_DURATION: f64 = 0.2;
const JUMP_SPEED: f64 = 291.0 + 2.0 / 3.0;
const JUMP_ACCEL: f64 = 1458.0 + 1.0 / 3.0;
const DODGE_OFFSET: f64 = 0.12;
const AERIAL_THROTTLE_ACCEL: f64 = 100.0 * 2.0 / 3.0;
const BOOST_CONSUMPTION: f64 = 100.0 / 3.0;
const SAFE_MAX_SPEED: f64 = 2100.0;
const MAX_SPEED: f64 = 2300.0;
const MAX_BOOST: f64 = 100.0;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct JumpTime {
    pub press: f64,
    pub time: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotType {
    Ground,
    Jump,
    DoubleJump,
    Aerial,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AerialValidation {
    pub boost_estimate: f64,
    pub required_acceleration_ratio: f64,
    pub estimated_final_speed: f64,
    pub turn_time: f64,
    pub boost_start_time: f64,
    pub boost_end_time: f64,
    pub valid: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AerialShotResult {
    pub valid: bool,
    pub fast: bool,
    pub validation: AerialValidation,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct DVec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl DVec3 {
    fn from(v: Vector) -> Self {
        Self { x: v.x as f64, y: v.y as f64, z: v.z as f64 }
    }

    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }

    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }

    fn mul(self, scale: f64) -> Self {
        Self { x: self.x * scale, y: self.y * scale, z: self.z * scale }
    }

    fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    fn magnitude(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalize(self) -> Self {
        let m = self.magnitude();
        if m > f64::EPSILON { self.mul(1.0 / m) } else { Self::default() }
    }

    fn angle(self, rhs: Self) -> f64 {
        let a = self.normalize();
        let b = rhs.normalize();
        a.dot(b).clamp(-1.0, 1.0).acos()
    }
}

/// Port of VirxERLU-CLib get_jump_time.
pub fn get_jump_time(car_to_target_z: f32, car_z_velocity: f32, gravity_z: f32) -> JumpTime {
    let mut result = JumpTime::default();
    let mut height = 0.0_f64;
    let mut velocity = car_z_velocity as f64;
    let target = car_to_target_z as f64;
    let gravity = gravity_z as f64;

    let g = gravity * SIMULATION_DT;
    let g2 = 2.0 * gravity;
    let jump_acceleration = JUMP_ACCEL * SIMULATION_DT;

    velocity += JUMP_SPEED;
    velocity += gravity * PHYSICS_DT;
    height += velocity * PHYSICS_DT;
    result.time += PHYSICS_DT;

    while target - height > 0.0 {
        // This intentionally mirrors the C source's actual statement
        // semantics: only press accumulation is conditional; v += ja is
        // unconditional within the loop (the original C has no braces).
        if result.time <= JUMP_MAX_DURATION
            && target - height > (velocity * velocity) / g2
        {
            result.press += SIMULATION_DT;
        }
        velocity += jump_acceleration;
        velocity += g;

        if velocity <= 0.0 {
            return result;
        }

        height += velocity * SIMULATION_DT;
        result.time += SIMULATION_DT;

        if result.time > 10.0 {
            break;
        }
    }

    result
}

/// Port of VirxERLU-CLib get_double_jump_time.
pub fn get_double_jump_time(
    car_to_target_z: f32,
    car_z_velocity: f32,
    gravity_z: f32,
) -> JumpTime {
    let mut result = JumpTime::default();
    let mut height = 0.0_f64;
    let mut velocity = car_z_velocity as f64;
    let target = car_to_target_z as f64;
    let gravity = gravity_z as f64;

    let g = gravity * SIMULATION_DT;
    let g2 = 2.0 * gravity;
    let jump_acceleration = JUMP_ACCEL * SIMULATION_DT;
    let mut jump_count = 0_u8;
    let mut jump_done = false;

    velocity += JUMP_SPEED;
    velocity += gravity * PHYSICS_DT;
    height += velocity * PHYSICS_DT;
    result.time += PHYSICS_DT;

    while target - height > 0.0 {
        if jump_count == 1 {
            velocity += JUMP_SPEED;
        }

        if !jump_done {
            if result.time > JUMP_MAX_DURATION {
                jump_done = true;
            } else {
                let velocity_incremented = velocity + JUMP_SPEED;
                if target - height > (velocity_incremented * velocity_incremented) / g2 {
                    result.press += SIMULATION_DT;
                    velocity += jump_acceleration;
                } else {
                    jump_done = true;
                }
            }
        } else if jump_count < 2 {
            jump_count += 1;
        }

        velocity += g;
        if velocity <= 0.0 {
            return result;
        }

        height += velocity * SIMULATION_DT;
        result.time += SIMULATION_DT;

        if result.time > 10.0 {
            break;
        }
    }

    result
}

pub fn ground_shot_is_viable(
    hitbox_height: f32,
    airborne: bool,
    offset_target_z: f32,
) -> bool {
    (offset_target_z as f64) < (hitbox_height as f64) * 2.0 && !airborne
}

pub fn jump_shot_is_viable(
    time_available: f32,
    gravity_z: f32,
    velocity_z: f32,
    hitbox_height: f32,
    airborne: bool,
    offset_target_z: f32,
    safe: bool,
) -> bool {
    let target_z = offset_target_z as f64;
    let height = hitbox_height as f64;

    if safe && (airborne || height * 1.5 > target_z || target_z > 250.0 + height / 2.0) {
        return false;
    }
    if !safe {
        return time_available >= 0.2;
    }

    time_available as f64
        >= get_jump_time(offset_target_z, velocity_z, gravity_z).time + DODGE_OFFSET + 0.05
}

pub fn double_jump_shot_is_viable(
    time_available: f32,
    gravity_z: f32,
    velocity_z: f32,
    half_hitbox_height: f32,
    airborne: bool,
    offset_target_z: f32,
    safe: bool,
) -> bool {
    let target_z = offset_target_z as f64;
    let half_height = half_hitbox_height as f64;

    if safe
        && (airborne
            || 250.0 + half_height > target_z
            || target_z > 500.0 + half_height)
    {
        return false;
    }
    if !safe {
        return time_available >= 0.2;
    }

    time_available as f64
        >= get_double_jump_time(offset_target_z, velocity_z, gravity_z).time
            + DODGE_OFFSET
            + 0.05
}

fn basic_aerial_validation(
    target: DVec3,
    predicted_position: DVec3,
    predicted_velocity: DVec3,
    car_forward: DVec3,
    car_boost: f64,
    time_available: f64,
    boost_acceleration: f64,
    safe: bool,
) -> AerialValidation {
    let delta = target.sub(predicted_position);
    let direction = delta.normalize();
    let phi = direction.angle(car_forward);
    let mut turn_time = 2.0 * (phi / 9.0).sqrt();
    if !safe {
        turn_time *= 0.7;
    }

    let tau1 = turn_time * (1.0 - 0.3 / phi).clamp(0.0, 1.0);
    let remaining = time_available - tau1;
    if !remaining.is_finite() || remaining <= 0.0 || boost_acceleration <= 0.0 {
        return AerialValidation { turn_time, ..AerialValidation::default() };
    }

    let required_acceleration = 2.0 * delta.magnitude() / (remaining * remaining);
    let ratio = required_acceleration / boost_acceleration;
    let tau2 = time_available - remaining * (1.0 - ratio.clamp(0.0, 1.0)).sqrt();
    let boost_estimate = ((tau2 - tau1) * BOOST_CONSUMPTION).floor();
    let velocity_estimate = predicted_velocity
        .add(direction.mul(boost_acceleration * (tau2 - tau1)))
        .magnitude();

    let valid = if safe {
        (car_boost > MAX_BOOST || boost_estimate < 0.9 * car_boost)
            && ratio.abs() < 0.9
            && velocity_estimate < SAFE_MAX_SPEED
    } else {
        (car_boost > MAX_BOOST || boost_estimate < car_boost)
            && ratio.abs() < 1.0
            && velocity_estimate < MAX_SPEED
    };

    AerialValidation {
        boost_estimate,
        required_acceleration_ratio: ratio,
        estimated_final_speed: velocity_estimate,
        turn_time,
        boost_start_time: tau1,
        boost_end_time: tau2,
        valid,
    }
}

/// Port of VirxERLU-CLib aerial_shot_is_viable, including its fast-aerial
/// first attempt and regular-jump fallback.
pub fn aerial_shot_is_viable(
    car: &CarState,
    target: Vector,
    time_available: f32,
    gravity: Vector,
    boost_acceleration: f32,
    safe: bool,
) -> AerialShotResult {
    let time = time_available as f64;
    if !time.is_finite() || time <= 0.0 || !boost_acceleration.is_finite() {
        return AerialShotResult::default();
    }

    let gravity = DVec3::from(gravity);
    let location = DVec3::from(car.location);
    let velocity = DVec3::from(car.velocity);
    let up = DVec3::from(car.up());
    let forward = DVec3::from(car.forward());
    let target = DVec3::from(target);
    let boost_acceleration = boost_acceleration as f64 + AERIAL_THROTTLE_ACCEL;

    let velocity_base = velocity.add(gravity.mul(time));
    let position_base = location
        .add(velocity.mul(time))
        .add(gravity.mul(0.5 * time * time));
    let total_jump_acceleration = JUMP_SPEED + JUMP_ACCEL * JUMP_MAX_DURATION;
    let partial_jump_location =
        JUMP_ACCEL * (time * JUMP_MAX_DURATION - 0.5 * JUMP_MAX_DURATION * JUMP_MAX_DURATION);

    let mut fast = false;
    let (mut predicted_velocity, mut predicted_position) = if !car.airborne {
        (
            velocity_base.add(up.mul(JUMP_SPEED + total_jump_acceleration)),
            position_base.add(up.mul(
                JUMP_SPEED * (2.0 * time - JUMP_MAX_DURATION) + partial_jump_location,
            )),
        )
    } else {
        (velocity_base, position_base)
    };

    let mut validation = basic_aerial_validation(
        target,
        predicted_position,
        predicted_velocity,
        forward,
        car.boost as f64,
        time,
        boost_acceleration,
        safe,
    );
    let mut found = validation.valid;

    if found && !car.airborne {
        fast = true;
    }

    if !car.airborne && (time < 1.45 || !found) {
        predicted_velocity = velocity_base.add(up.mul(total_jump_acceleration));
        predicted_position = position_base.add(up.mul(
            JUMP_SPEED * time + partial_jump_location,
        ));
        let regular = basic_aerial_validation(
            target,
            predicted_position,
            predicted_velocity,
            forward,
            car.boost as f64,
            time,
            boost_acceleration,
            safe,
        );
        if regular.valid {
            found = true;
            fast = false;
            validation = regular;
        }
    }

    AerialShotResult { valid: found, fast, validation }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn test_car(airborne: bool, boost: f32) -> CarState {
        CarState {
            index: 0,
            team: 0,
            location: Vector::ZERO,
            velocity: Vector::ZERO,
            orientation: Matrix3::new(0.0, 0.0, 0.0),
            angular_velocity: Vector::ZERO,
            demolished: false,
            airborne,
            supersonic: false,
            jumped: airborne,
            doublejumped: false,
            air_state: if airborne { AirState::InAir } else { AirState::OnGround },
            has_jumped: airborne,
            has_double_jumped: false,
            has_dodged: false,
            dodge_timeout: 0.0,
            dodge_elapsed: 0.0,
            dodge_dir: Vector::ZERO,
            flip_reset_available: false,
            boost,
            hitbox: Hitbox { length: 118.0, width: 84.0, height: 36.0, offset: Vector::ZERO },
        }
    }

    #[test]
    fn jump_time_returns_finite_press_and_contact_time() {
        let result = get_jump_time(100.0, 0.0, -650.0);
        assert!(result.time.is_finite() && result.time > 0.0);
        assert!(result.press.is_finite() && result.press >= 0.0);
    }

    #[test]
    fn double_jump_reaches_higher_targets_than_single_jump() {
        let single = get_jump_time(350.0, 0.0, -650.0);
        let double = get_double_jump_time(350.0, 0.0, -650.0);
        assert!(double.time < single.time);
        assert!(double.time > 0.0);
    }

    #[test]
    fn safe_jump_shots_reject_airborne_cars_and_out_of_range_heights() {
        assert!(!jump_shot_is_viable(2.0, -650.0, 0.0, 36.0, true, 100.0, true));
        assert!(!jump_shot_is_viable(2.0, -650.0, 0.0, 36.0, false, 400.0, true));
        assert!(jump_shot_is_viable(0.3, -650.0, 0.0, 36.0, false, 60.0, false));
    }

    #[test]
    fn safe_double_jump_uses_double_jump_height_window() {
        assert!(!double_jump_shot_is_viable(3.0, -650.0, 0.0, 18.0, false, 200.0, true));
        assert!(double_jump_shot_is_viable(3.0, -650.0, 0.0, 18.0, false, 300.0, true));
    }

    #[test]
    fn ground_shots_require_grounded_car_and_reachable_ball_height() {
        assert!(ground_shot_is_viable(36.0, false, 60.0));
        assert!(!ground_shot_is_viable(36.0, true, 60.0));
        assert!(!ground_shot_is_viable(36.0, false, 80.0));
    }

    #[test]
    fn aerial_validation_rejects_impossible_boost_and_speed_budget() {
        let car = test_car(false, 0.0);
        let result = aerial_shot_is_viable(
            &car,
            Vector::new(2500.0, 0.0, 1000.0),
            0.5,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            true,
        );
        assert!(!result.valid);
    }

    #[test]
    fn reachable_fast_aerial_reports_launch_mode() {
        let car = test_car(false, 100.0);
        let result = aerial_shot_is_viable(
            &car,
            Vector::new(0.0, 0.0, 900.0),
            1.8,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            true,
        );
        assert!(result.valid);
        assert!(result.fast);
    }
}
