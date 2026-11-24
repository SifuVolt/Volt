//! Stateful strike and kickoff execution.
//!
//! Candidate geometry/reachability comes from the Python-derived
//! shot_parser/shot_physics/shot_planner modules. This module owns only the
//! per-tick controller state: approach, jump timing, aerial alignment,
//! contact dodge, and the source-sequenced kickoff speedflip.

use crate::{
    control::default_pd,
    drive::{drive_to, DriveRequest},
    math::Vector,
    shot_parser::ShotTargets,
    shot_physics::{get_jump_time, ShotType},
    shot_planner::ShotCandidate,
    state::{BallState, CarState},
};
use rlbot::flat::ControllerState;

const BALL_RADIUS: f32 = 92.75;
const MAX_SPEED: f32 = 2300.0;

fn sign(value: f32) -> f32 {
    if value < 0.0 { -1.0 } else if value > 0.0 { 1.0 } else { 0.0 }
}

fn clamp_magnitude(vector: Vector, max: f32) -> Vector {
    let magnitude = vector.magnitude();
    if magnitude > max && magnitude > 0.0 {
        vector * (max / magnitude)
    } else {
        vector
    }
}

fn target_center(targets: ShotTargets) -> Vector {
    (targets.left + targets.right) * 0.5
}

fn party_cannon_power_modifier(speed: f32) -> f32 {
    let speed = speed.clamp(0.0, 4600.0);
    if speed <= 500.0 {
        0.65
    } else if speed <= MAX_SPEED {
        let blend = (speed - 500.0) / 1800.0;
        0.65 + (0.55 - 0.65) * blend
    } else {
        let blend = (speed - MAX_SPEED) / 2300.0;
        0.55 + (0.30 - 0.55) * blend
    }
}

/// Party Cannon's predicted post-contact direction calculation. It models
/// the velocity exchange and the ball's subsequent flight to the target
/// instead of aiming only at the goal centre from the current ball position.
pub fn party_cannon_shot_direction(
    car: CarState,
    ball: BallState,
    time_remaining: f32,
    targets: ShotTargets,
    gravity: Vector,
) -> Vector {
    let goal = target_center(targets);
    let safe_time = time_remaining.max(1.0 / 120.0);
    let car_final_velocity = clamp_magnitude((ball.location - car.location) / safe_time, MAX_SPEED);
    let velocity_for_power = (car_final_velocity * 6.0 + ball.velocity) / 7.0;
    let direction_to_score = (goal - ball.location).flatten().normalize();
    let velocity_difference = (car_final_velocity - ball.velocity).magnitude();

    let score_speed = velocity_difference * party_cannon_power_modifier(velocity_difference)
        + velocity_for_power.dot(direction_to_score);
    let time_to_score = ball.location.flatten().distance(goal.flatten())
        / score_speed.clamp(500.0, 6000.0);

    let predicted_after_hit_velocity = (car_final_velocity + ball.velocity * 2.0) / 3.0;
    let predicted_ball = ball.location
        + predicted_after_hit_velocity * time_to_score
        + gravity * (0.5 * time_to_score * time_to_score);

    let direction = (goal - predicted_ball).normalize();
    if direction.magnitude() > 0.001 {
        direction
    } else {
        direction_to_score
    }
}

/// Car-centre location which places the front hitbox face at the intended
/// ball contact. The RLBot hitbox offset is transformed using the same
/// forward/right/up basis as the rest of the bot.
pub fn hitbox_contact_target(car: CarState, ball: Vector, shot_direction: Vector) -> Vector {
    let half_length = car.hitbox.length.max(1.0) * 0.5;
    let world_offset = car.forward() * car.hitbox.offset.x
        + car.right() * car.hitbox.offset.y
        + car.up() * car.hitbox.offset.z;
    ball - shot_direction.normalize() * (BALL_RADIUS + half_length) - world_offset
}

fn align_to_direction(car: &CarState, direction: Vector) -> ControllerState {
    let local = car.local(direction);
    let (_, output) = default_pd(car, local, false, None);
    let mut controller = ControllerState::default();
    controller.steer = output.steer;
    controller.pitch = output.pitch;
    controller.yaw = output.yaw;
    controller.roll = output.roll;
    controller.throttle = 1.0;
    controller
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrikePhase {
    Approach,
    JumpHold,
    JumpRelease,
    SecondJump,
    Flight,
    Dodge,
}

#[derive(Debug, Clone)]
pub struct StrikeExecutor {
    candidate: Option<ShotCandidate>,
    phase: StrikePhase,
    phase_started: f32,
    last_dodge_at: f32,
    jump_started: bool,
    jump_hold_duration: f32,
    last_tick: Option<f32>,
}

impl Default for StrikeExecutor {
    fn default() -> Self {
        Self {
            candidate: None,
            phase: StrikePhase::Approach,
            phase_started: 0.0,
            last_dodge_at: -999.0,
            jump_started: false,
            jump_hold_duration: 0.06,
            last_tick: None,
        }
    }
}

impl StrikeExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn active(&self) -> bool {
        self.phase != StrikePhase::Approach
    }

    pub fn active_candidate(&self) -> Option<ShotCandidate> {
        self.candidate
    }

    pub fn reset(&mut self) {
        self.candidate = None;
        self.phase = StrikePhase::Approach;
        self.phase_started = 0.0;
        self.jump_started = false;
        self.jump_hold_duration = 0.06;
        self.last_tick = None;
    }

    fn set_phase(&mut self, phase: StrikePhase, now: f32) {
        self.phase = phase;
        self.phase_started = now;
    }

    fn current_target(
        &self,
        car: CarState,
        targets: ShotTargets,
        gravity: Vector,
        time_remaining: f32,
    ) -> (Vector, Vector) {
        let shot = self.candidate.expect("strike candidate is set");
        let predicted_ball = BallState {
            location: shot.ball,
            velocity: shot.ball_velocity,
        };
        let predicted_direction = party_cannon_shot_direction(
            car, predicted_ball, time_remaining, targets, gravity,
        );
        let shot_direction = if shot.shot_direction.magnitude() > 0.001
            && shot.shot_type == ShotType::Ground
        {
            shot.shot_direction.flatten().normalize()
        } else {
            predicted_direction
        };

        let target = hitbox_contact_target(car, shot.ball, shot_direction);
        (target, shot_direction)
    }

    /// Update the active shot with the newest selected candidate and emit
    /// a fresh, fully initialized controller every tick.
    pub fn tick(
        &mut self,
        car: CarState,
        ball: BallState,
        candidate: Option<ShotCandidate>,
        targets: ShotTargets,
        gravity: Vector,
        boost_accel: f32,
        now: f32,
    ) -> ControllerState {
        let elapsed_since_tick = self.last_tick
            .map(|last| (now - last).max(0.0))
            .unwrap_or(0.0);
        self.last_tick = Some(now);

        if candidate.is_none() {
            if let Some(mut current) = self.candidate {
                current.time -= elapsed_since_tick;
                current.ball = current.ball
                    + current.ball_velocity * elapsed_since_tick
                    + gravity * (0.5 * elapsed_since_tick * elapsed_since_tick);
                current.ball_velocity = current.ball_velocity + gravity * elapsed_since_tick;
                self.candidate = Some(current);
            }
        }

        if let Some(next) = candidate {
            if self.phase == StrikePhase::Approach
                || self.candidate.is_none_or(|current| current.shot_type == next.shot_type)
            {
                self.candidate = Some(next);
            }
        }

        let Some(shot) = self.candidate else {
            if self.phase == StrikePhase::Approach {
                return ControllerState::default();
            }
            // Once airborne, keep the last reachable contact rather than
            // abandoning the maneuver because a fresh ground-only scan
            // correctly returned no candidate.
            if !car.airborne {
                self.reset();
                return ControllerState::default();
            }
            return crate::drive::recover_in_air(&car, boost_accel, Some(ball.location));
        };

        let time_remaining = shot.time.max(0.0);
        let (contact_target, shot_direction) = self.current_target(
            car, targets, gravity, time_remaining,
        );
        let distance_to_contact = car.location.distance(contact_target);

        if self.phase == StrikePhase::Approach {
            if shot.shot_type == ShotType::Ground {
                let mut target = shot.approach;
                let mut preserve = false;
                if car.location.flat_distance(target) < 380.0 {
                    // Follow through beyond the ball rather than treating
                    // the parser's approach point as a place to stop.
                    target = ball.location + shot_direction * 720.0;
                    preserve = true;
                }
                return drive_to(
                    &car,
                    DriveRequest {
                        target,
                        target_speed: MAX_SPEED,
                        boost_accel,
                        allow_handbrake: true,
                        preserve_momentum: preserve,
                    },
                );
            }

            let vertical_gap = (shot.ball.z - BALL_RADIUS - car.location.z).max(0.0);
            let jump_prediction = get_jump_time(vertical_gap, car.velocity.z, gravity.z);
            let jump_time = jump_prediction.time as f32;
            let approach_distance = car.location.flat_distance(shot.approach);
            let launch_due = time_remaining <= jump_time.max(0.06) + 0.10
                || approach_distance < 230.0
                || time_remaining < 0.18;

            if launch_due && !car.airborne {
                self.jump_started = true;
                self.jump_hold_duration = (jump_prediction.press as f32).clamp(0.06, 0.20);
                self.set_phase(StrikePhase::JumpHold, now);
            } else {
                return drive_to(
                    &car,
                    DriveRequest {
                        target: shot.approach,
                        target_speed: MAX_SPEED,
                        boost_accel,
                        allow_handbrake: true,
                        preserve_momentum: false,
                    },
                );
            }
        }

        match self.phase {
            StrikePhase::Approach => ControllerState::default(),

            StrikePhase::JumpHold => {
                let mut controller = align_to_direction(&car, shot_direction);
                controller.jump = true;
                controller.throttle = 1.0;
                controller.boost = car.boost > 0.0
                    && shot.shot_type == ShotType::Aerial
                    && car.local(shot_direction).x > 0.6;
                if now - self.phase_started >= self.jump_hold_duration {
                    self.set_phase(StrikePhase::JumpRelease, now);
                    controller.jump = false;
                }
                controller
            }

            StrikePhase::JumpRelease => {
                let mut controller = align_to_direction(&car, shot_direction);
                controller.jump = false;
                controller.boost = car.boost > 0.0
                    && shot.shot_type == ShotType::Aerial
                    && car.local(shot_direction).x > 0.6;
                if now - self.phase_started >= 0.05 {
                    if matches!(shot.shot_type, ShotType::DoubleJump | ShotType::Aerial) {
                        self.set_phase(StrikePhase::SecondJump, now);
                    } else {
                        self.set_phase(StrikePhase::Flight, now);
                    }
                }
                controller
            }

            StrikePhase::SecondJump => {
                let mut controller = align_to_direction(&car, shot_direction);
                controller.jump = now - self.phase_started < 0.06;
                controller.boost = car.boost > 0.0 && car.local(shot_direction).x > 0.65;
                controller.throttle = 1.0;
                if now - self.phase_started >= 0.06 {
                    self.set_phase(StrikePhase::Flight, now);
                    controller.jump = false;
                }
                controller
            }

            StrikePhase::Flight => {
                if !car.airborne && self.jump_started {
                    self.reset();
                    return ControllerState::default();
                }
                if shot.time < -0.15 && distance_to_contact > 180.0 {
                    self.reset();
                    return crate::drive::recover_in_air(&car, boost_accel, Some(ball.location));
                }

                let target_delta = contact_target - car.location;
                let local_target = car.local(target_delta);
                let local_shot = car.local(shot_direction);
                let aim_vector = if target_delta.magnitude() > 65.0 {
                    target_delta
                } else {
                    shot_direction
                };
                let mut controller = align_to_direction(&car, aim_vector);
                controller.throttle = 1.0;

                // Python Volt's characteristic S-tilt on a non-quick
                // single jump: nose up while the ball is elevated, preserve
                // goal-directed yaw, and counter-roll slightly against yaw.
                if shot.shot_type == ShotType::Jump
                    && ball.location.z > 130.0
                    && ball.location.z > car.location.z + 65.0
                    && shot.time > 0.45
                {
                    controller.pitch = 1.0;
                    if controller.yaw.abs() > 0.08 {
                        controller.roll = -0.20 * sign(controller.yaw);
                    }
                }

                let angle = local_shot.angle(Vector::new(1.0, 0.0, 0.0));
                let projected_forward = car.velocity.dot(shot_direction);
                let need_acceleration = projected_forward < 2180.0
                    || local_target.x > 150.0 && shot.time > 0.08;
                controller.boost = car.boost > 0.0
                    && boost_accel > 0.0
                    && local_shot.x > 0.0
                    && angle < 0.48
                    && need_acceleration
                    && distance_to_contact > 70.0;

                let can_directional_dodge = shot.shot_type == ShotType::Jump
                    && !car.has_dodged
                    && now - self.last_dodge_at > 0.45
                    && distance_to_contact < (BALL_RADIUS + car.hitbox.length) * 1.02
                    && local_target.x > 0.0
                    && shot.time < 0.30;
                if can_directional_dodge {
                    self.last_dodge_at = now;
                    self.set_phase(StrikePhase::Dodge, now);
                    let local_ball = car.local(ball.location - car.location).flatten().normalize();
                    controller.pitch = -local_ball.x;
                    controller.yaw = local_ball.y;
                    controller.roll = 0.0;
                    controller.jump = true;
                    controller.boost = car.boost > 0.0;
                }
                controller
            }

            StrikePhase::Dodge => {
                let local_ball = car.local(ball.location - car.location).flatten().normalize();
                let mut controller = ControllerState::default();
                controller.throttle = 1.0;
                controller.pitch = -local_ball.x;
                controller.yaw = local_ball.y;
                controller.jump = now - self.phase_started < 0.10;
                controller.boost = car.boost > 0.0;
                if now - self.phase_started >= 0.10 {
                    self.reset();
                    controller.jump = false;
                    controller.boost = false;
                }
                controller
            }
        }
    }
}

/// Exact timed speedflip input sequence used by Python Volt's speed_flip.
/// Each tick begins with clean control values, preventing stale jump/yaw/roll.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedFlip {
    direction: f32,
    use_boost: bool,
    step: usize,
    step_started: Option<f32>,
    finished: bool,
}

impl SpeedFlip {
    pub fn new(right_handed: bool, use_boost: bool) -> Self {
        Self {
            direction: if right_handed { 1.0 } else { -1.0 },
            use_boost,
            step: 0,
            step_started: None,
            finished: false,
        }
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    pub fn tick(&mut self, car: CarState, now: f32) -> ControllerState {
        if self.finished {
            return ControllerState::default();
        }
        let start = *self.step_started.get_or_insert(now);

        let (duration, steer, yaw, pitch, roll, jump, boost) = match self.step {
            0 => (5.0 / 60.0, -self.direction, 0.0, 0.0, 0.0, false, true),
            1 => (1.0 / 60.0, 0.0, 0.0, 0.0, 0.0, true, true),
            2 => (0.05, 0.0, 0.0, 0.0, 0.0, false, true),
            3 => (1.0 / 60.0, 0.0, self.direction, -1.0, 0.0, true, true),
            4 => (0.60, 0.0, 0.0, 1.0, 0.0, false, true),
            5 => (0.20, 0.0, self.direction, 0.0, self.direction, false, true),
            _ => {
                self.finished = true;
                return ControllerState::default();
            }
        };

        let mut controller = ControllerState::default();
        controller.throttle = 1.0;
        controller.steer = steer;
        controller.yaw = yaw;
        controller.pitch = pitch;
        controller.roll = roll;
        controller.jump = jump;
        controller.boost = boost && self.use_boost && car.boost > 0.0;

        if now - start > duration {
            self.step += 1;
            self.step_started = Some(now);
            if self.step >= 6 {
                self.finished = true;
            }
        }

        controller
    }
}

#[derive(Debug, Clone, Default)]
pub struct KickoffExecutor {
    speedflip: Option<SpeedFlip>,
    speedflip_used: bool,
}

impl KickoffExecutor {
    pub fn reset(&mut self) {
        self.speedflip = None;
        self.speedflip_used = false;
    }

    pub fn tick_kicker(
        &mut self,
        car: CarState,
        ball: BallState,
        team: u32,
        boost_accel: f32,
        now: f32,
    ) -> ControllerState {
        if let Some(speedflip) = self.speedflip.as_mut() {
            let output = speedflip.tick(car, now);
            if speedflip.finished() {
                self.speedflip = None;
                self.speedflip_used = true;
            }
            return output;
        }

        let local_target = car.local_location(ball.location);
        let distance = car.location.flat_distance(ball.location);
        let angle = local_target.y.atan2(local_target.x).abs();
        let forward_speed = car.local_velocity().x;
        let right_handed = car.location.x * (if team == 0 { 1.0 } else { -1.0 }) > 0.0;

        let eligible = !self.speedflip_used
            && !car.airborne
            && local_target.x > 0.0
            && angle < 0.14
            && distance > 3000.0
            && (600.0..1600.0).contains(&forward_speed)
            && car.boost >= 20.0;

        if eligible {
            self.speedflip = Some(SpeedFlip::new(right_handed, true));
            return self.speedflip.as_mut().unwrap().tick(car, now);
        }

        drive_to(
            &car,
            DriveRequest {
                target: ball.location,
                target_speed: MAX_SPEED,
                boost_accel,
                allow_handbrake: false,
                preserve_momentum: true,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math::Matrix3, state::Hitbox};
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
            boost: 70.0,
            hitbox: Hitbox {
                length: 118.0,
                width: 84.0,
                height: 36.0,
                offset: Vector::ZERO,
            },
        }
    }

    #[test]
    fn party_cannon_hitbox_target_is_behind_the_ball() {
        let me = car(Vector::ZERO, Vector::ZERO);
        let ball = Vector::new(0.0, 1000.0, 300.0);
        let target = hitbox_contact_target(me, ball, Vector::new(0.0, 1.0, 0.0));
        assert!(target.y < ball.y);
        assert!((ball.y - target.y - (BALL_RADIUS + 59.0)).abs() < 0.01);
    }

    #[test]
    fn party_cannon_direction_is_finite_for_stationary_ball() {
        let me = car(Vector::new(0.0, -1000.0, 17.0), Vector::ZERO);
        let ball = BallState {
            location: Vector::new(0.0, 0.0, 120.0),
            velocity: Vector::ZERO,
        };
        let targets = ShotTargets {
            left: Vector::new(-820.0, 5120.0, 350.0),
            right: Vector::new(820.0, 5120.0, 350.0),
        };
        let direction = party_cannon_shot_direction(
            me, ball, 1.0, targets, Vector::new(0.0, 0.0, -650.0),
        );
        assert!(direction.x.is_finite() && direction.y.is_finite() && direction.z.is_finite());
        assert!(direction.magnitude() > 0.9);
    }

    #[test]
    fn speedflip_uses_clean_controller_defaults_between_steps() {
        let me = car(Vector::ZERO, Vector::ZERO);
        let mut flip = SpeedFlip::new(true, true);
        let output = flip.tick(me, 0.0);
        assert_eq!(output.throttle, 1.0);
        assert_eq!(output.steer, -1.0);
        assert!(!output.jump);
    }
}
