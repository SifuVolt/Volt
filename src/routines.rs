//! Stateful strike and kickoff execution.
//!
//! Candidate geometry/reachability comes from the Python-derived
//! shot_parser/shot_physics/shot_planner modules. This module owns only the
//! per-tick controller state: approach, jump timing, aerial alignment,
//! contact dodge, and the source-sequenced kickoff speedflip.

use crate::{
    aerial::wall_shot_adjust_jump,
    control::{default_drive, default_pd, DriveInput},
    drive::{drive_to, DriveRequest},
    math::{utils::{cap, cap_in_field}, Vector},
    physics::drive::dodge_impulse,
    prediction::{BallSlice, BallTrajectory},
    shot_parser::{runtime_ground_reachability, ShotTargets},
    shot_physics::{double_jump_shot_is_viable, ground_shot_is_viable, get_jump_time, jump_shot_is_viable, ShotType},
    shot_planner::{grid_jump_time, ShotCandidate},
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

/// Python _relative_goal_aim_point: blue is side -1, orange is side +1.
/// The returned z is retained for diagnostics/tests even though the current
/// ground-shot vector is flattened before contact-direction calculation.
fn ground_aim_point(car: &CarState, live_ball: Vector, targets: ShotTargets) -> Vector {
    let mut aim = target_center(targets);
    aim.z = 300.0;
    let python_team_side = if car.team == 0 { -1.0 } else { 1.0 };
    if live_ball.y * python_team_side <= 0.0 {
        aim.z = 470.0;
    } else if car.airborne {
        aim.z -= 90.0;
    } else {
        aim.z += 10.0;
    }
    aim
}

fn ground_shot_direction(
    car: &CarState,
    ball: Vector,
    live_ball: Vector,
    targets: ShotTargets,
    fallback: Vector,
) -> Vector {
    let aim = ground_aim_point(car, live_ball, targets);
    let direction = (aim - ball).flatten().normalize();
    if direction.magnitude() > 0.001 {
        direction
    } else {
        fallback.flatten().normalize()
    }
}

/// Exact Python _shot_approach_target flank: only move to the side when
/// poorly aligned, keep the car on its existing side, and fade the offset
/// away during the final 0.25 seconds.
fn ground_flank_target(
    car: &CarState,
    ball: Vector,
    contact_target: Vector,
    shot_direction: Vector,
    time_left: f32,
) -> Vector {
    let final_target = contact_target.flatten();
    if time_left <= 0.25 {
        return final_target;
    }
    let shot = shot_direction.flatten().normalize();
    if shot.magnitude() < 1.0 {
        return final_target;
    }
    let car_to_ball = (ball - car.location).flatten().normalize();
    if car_to_ball.magnitude() < 1.0 {
        return final_target;
    }
    let alignment = cap(car_to_ball.dot(shot), -1.0, 1.0);
    if alignment >= 0.88 {
        return final_target;
    }
    // shot x world-up, matching Python's Vector.cross convention.
    let lateral = Vector::new(shot.y, -shot.x, 0.0).normalize();
    if lateral.magnitude() < 1.0 {
        return final_target;
    }
    let mut car_side = sign((car.location - ball).dot(lateral));
    if car_side == 0.0 {
        car_side = 1.0;
    }
    let time_blend = cap((time_left - 0.25) / 0.9, 0.0, 1.0);
    let alignment_blend = cap((0.88 - alignment) / 1.35, 0.0, 1.0);
    let radius = 700.0 * time_blend * alignment_blend;
    if radius < 80.0 {
        return final_target;
    }
    final_target + lateral * (car_side * radius)
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
    /// Python ground_shot's timed jump-release-directional-flip sequence.
    GroundFlip,
    /// Python jump_shot's jump, orientation, counter, and directional-dodge sequence.
    JumpFlight,
    /// Double-jump path retained separately from the single-jump dodge.
    JumpHold,
    JumpRelease,
    SecondJump,
    Flight,
}

const JUMP_SPEED: f32 = 291.6667;
const JUMP_ACCEL: f32 = 1458.3333;
const JUMP_MAX_DURATION: f32 = 0.20;
const MIN_AIR_BOOST_TIME: f32 = 1.0 / 30.0;
const AIR_THROTTLE_ACCEL: f32 = 66.666667;
const MAX_JUMP_SHOT_AGE: f32 = 0.80;

/// Port of routines.flip: 75 ms initial jump, three release frames, then a
/// directional dodge held for 0.4 s (cancel) or 0.9 s (full flip).
#[derive(Debug, Clone, Copy)]
struct GroundFlip {
    direction: Vector,
    cancel: bool,
    started_at: Option<f32>,
    flip_started_at: Option<f32>,
    release_frames: u8,
}

impl GroundFlip {
    fn new(local_direction: Vector, cancel: bool) -> Self {
        Self {
            direction: local_direction.flatten().normalize(),
            cancel,
            started_at: None,
            flip_started_at: None,
            release_frames: 0,
        }
    }

    fn tick(&mut self, now: f32) -> (ControllerState, bool, bool) {
        let start = *self.started_at.get_or_insert(now);
        let elapsed = (now - start).max(0.0);
        let mut controller = ControllerState::default();

        if elapsed < 0.075 {
            controller.jump = true;
        } else if self.release_frames < 3 {
            controller.jump = false;
            self.release_frames += 1;
        } else if elapsed < 0.4 || (!self.cancel && elapsed < 0.9) {
            self.flip_started_at.get_or_insert(now);
            controller.jump = true;
            controller.pitch = -self.direction.x;
            controller.yaw = self.direction.y;
        } else {
            let recover = self.flip_started_at
                .is_some_and(|flip_started| now - flip_started >= 0.5);
            return (controller, true, recover);
        }

        (controller, false, false)
    }
}

#[derive(Debug, Clone)]
pub struct StrikeExecutor {
    candidate: Option<ShotCandidate>,
    phase: StrikePhase,
    phase_started: f32,
    jump_started_at: Option<f32>,
    jump_counter: u8,
    quick_shot: bool,
    dodge_started_at: Option<f32>,
    last_tick: Option<f32>,
    last_debug_at: f32,
    current_jump_target: Vector,
    current_shot_direction: Vector,
    ground_flip: Option<GroundFlip>,
    double_jump_prev_jump: bool,
    double_jump_sent: bool,
}

impl Default for StrikeExecutor {
    fn default() -> Self {
        Self {
            candidate: None,
            phase: StrikePhase::Approach,
            phase_started: 0.0,
            jump_started_at: None,
            jump_counter: 0,
            quick_shot: false,
            dodge_started_at: None,
            last_tick: None,
            last_debug_at: -999.0,
            current_jump_target: Vector::ZERO,
            current_shot_direction: Vector::new(1.0, 0.0, 0.0),
            ground_flip: None,
            double_jump_prev_jump: false,
            double_jump_sent: false,
        }
    }
}

fn controller_from_drive_output(
    output: crate::control::DriveOutput,
    car: &CarState,
) -> ControllerState {
    let mut controller = ControllerState::default();
    controller.steer = output.controls.steer;
    controller.pitch = output.controls.pitch;
    controller.yaw = output.controls.yaw;
    controller.roll = output.controls.roll;
    if let Some(throttle) = output.throttle {
        controller.throttle = throttle;
    }
    if let Some(handbrake) = output.handbrake {
        controller.handbrake = handbrake;
    }
    if let Some(boost) = output.boost {
        controller.boost = boost && car.boost > 0.0;
    }
    controller
}

fn nearest_prediction_slice(
    prediction: &BallTrajectory,
    time: f32,
) -> Option<BallSlice> {
    prediction.slices.iter().copied().min_by(|a, b| {
        (a.time - time).abs().total_cmp(&(b.time - time).abs())
    })
}

impl StrikeExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    /// A strike owns the car from candidate acquisition, not merely after the
    /// first jump press. This prevents a one-frame parser miss from erasing
    /// the committed approach before launch.
    pub fn active(&self) -> bool {
        self.candidate.is_some()
    }

    pub fn active_candidate(&self) -> Option<ShotCandidate> {
        self.candidate
    }

    pub fn is_approaching(&self) -> bool {
        self.candidate.is_some() && self.phase == StrikePhase::Approach
    }

    /// Cancel only an unlaunched approach when teamplay transfers the
    /// challenge. Once jump flight begins, finish the committed contact.
    pub fn cancel_approach(&mut self) {
        if self.is_approaching() {
            self.reset();
        }
    }

    /// Transfer a wall approach directly to its launch routine.
    pub fn start_from_wall(
        &mut self,
        candidate: ShotCandidate,
        now: f32,
        _jump_hold_duration: f32,
    ) {
        self.candidate = Some(candidate);
        self.phase_started = now;
        self.last_tick = Some(now);
        self.jump_counter = 0;
        self.jump_started_at = None;
        self.dodge_started_at = None;
        self.current_jump_target = candidate.approach;
        self.current_shot_direction = candidate.shot_direction.flatten().normalize();
        self.ground_flip = None;
        self.double_jump_prev_jump = false;
        self.double_jump_sent = false;
        self.phase = if candidate.shot_type == ShotType::Jump {
            StrikePhase::JumpFlight
        } else {
            StrikePhase::JumpHold
        };
    }

    pub fn reset(&mut self) {
        self.candidate = None;
        self.phase = StrikePhase::Approach;
        self.phase_started = 0.0;
        self.jump_started_at = None;
        self.jump_counter = 0;
        self.quick_shot = false;
        self.dodge_started_at = None;
        self.last_tick = None;
        self.last_debug_at = -999.0;
        self.current_jump_target = Vector::ZERO;
        self.current_shot_direction = Vector::new(1.0, 0.0, 0.0);
        self.ground_flip = None;
        self.double_jump_prev_jump = false;
        self.double_jump_sent = false;
    }

    fn set_phase(&mut self, phase: StrikePhase, now: f32) {
        self.phase = phase;
        self.phase_started = now;
    }

    fn refresh_committed_candidate(
        &mut self,
        next: Option<ShotCandidate>,
        prediction: &BallTrajectory,
        now: f32,
        gravity: Vector,
    ) {
        let elapsed = self.last_tick
            .map(|last| (now - last).max(0.0))
            .unwrap_or(0.0);
        self.last_tick = Some(now);

        match (self.candidate, next) {
            (None, Some(candidate)) => self.candidate = Some(candidate),
            (Some(current), Some(candidate)) if self.phase == StrikePhase::Approach => {
                // Keep the chosen shot family stable through approach. A
                // different family may replace it only when its contact is
                // materially earlier, not because two adjacent scans scored
                // nearly equal candidates differently.
                if current.shot_type == candidate.shot_type
                    || candidate.time + 0.20 < current.time
                {
                    self.candidate = Some(candidate);
                } else {
                    let mut updated = current;
                    updated.time = (updated.time - elapsed).max(-1.0);
                    if updated.time >= 0.0 {
                        if let Some(slice) = nearest_prediction_slice(prediction, updated.time) {
                            updated.ball = slice.location;
                            updated.ball_velocity = slice.velocity;
                        }
                    }
                    self.candidate = Some(updated);
                }
            }
            (Some(mut current), Some(_)) => {
                // Once launch has begun, never swap the contact underneath
                // the airborne routine. Advance the committed intercept clock
                // even when the planner continues to return a fresh candidate.
                current.time -= elapsed;
                self.candidate = Some(current);
            }
            (Some(mut current), None) => {
                current.time -= elapsed;
                if self.phase == StrikePhase::Approach && current.time >= 0.0 {
                    if let Some(slice) = nearest_prediction_slice(
                        prediction,
                        current.time,
                    ) {
                        current.ball = slice.location;
                        current.ball_velocity = slice.velocity;
                        // Keep the finder clock synchronized to the actual
                        // prediction slice rather than accumulating timer drift.
                        current.time = slice.time;
                    } else {
                        // No prediction is a transient sensor gap, not a
                        // reason to erase a committed intercept.
                        current.ball = current.ball
                            + current.ball_velocity * elapsed
                            + gravity * (0.5 * elapsed * elapsed);
                        current.ball_velocity = current.ball_velocity + gravity * elapsed;
                    }
                }
                self.candidate = Some(current);
            }
            (None, None) => {}
        }
    }

    fn refreshed_jump_geometry(
        car: &CarState,
        shot: ShotCandidate,
        live_ball: BallState,
        targets: ShotTargets,
        gravity: Vector,
        time_remaining: f32,
    ) -> (Vector, Vector, Option<Vector>) {
        if shot.shot_type == ShotType::DoubleJump {
            // Python double_jump uses relative goal-centre aim, not the
            // Party Cannon predicted outgoing vector used by single jumps.
            let direction = ground_shot_direction(
                car,
                shot.ball,
                live_ball.location,
                targets,
                shot.shot_direction,
            );
            let target = hitbox_contact_target(*car, shot.ball, direction);
            return (target, direction, None);
        }

        let goal = target_center(targets);
        let predicted_direction = party_cannon_shot_direction(
            *car,
            BallState {
                location: shot.ball,
                velocity: shot.ball_velocity,
            },
            time_remaining.max(1.0 / 120.0),
            targets,
            gravity,
        );
        let goal_direction = (goal - shot.ball).flatten().normalize();
        let predicted_direction = predicted_direction.flatten().normalize();
        let blended = if goal_direction.magnitude() > 0.001
            && predicted_direction.magnitude() > 0.001
        {
            (goal_direction * 0.65 + predicted_direction * 0.35).normalize()
        } else if goal_direction.magnitude() > 0.001 {
            goal_direction
        } else {
            predicted_direction
        };

        let (shot_direction, wall_normal, _) =
            wall_shot_adjust_jump(shot.ball, blended);
        let hitbox_target = hitbox_contact_target(*car, shot.ball, shot_direction);
        let mut approach = hitbox_target;

        if let Some(normal) = wall_normal {
            let tangent = shot_direction - normal * shot_direction.dot(normal);
            if tangent.magnitude() > 0.001 {
                approach = approach - tangent.normalize() * 25.0;
            }
            if normal.z.abs() < 0.95 {
                let jump_time = get_jump_time(
                    shot.ball.z - car.location.z,
                    car.velocity.z,
                    gravity.z,
                ).time as f32;
                approach.z = shot.ball.z
                    + (approach.z - shot.ball.z).clamp(-120.0, 120.0);
                approach = approach - gravity * (0.5 * jump_time * jump_time);
            }
        }

        (approach, shot_direction, wall_normal)
    }

    fn tick_ground_or_jump_approach(
        &mut self,
        car: CarState,
        live_ball: BallState,
        shot: ShotCandidate,
        targets: ShotTargets,
        gravity: Vector,
        boost_accel: f32,
        now: f32,
    ) -> ControllerState {
        if shot.shot_type == ShotType::Ground {
            let raw_time = shot.time;
            let time_remaining = raw_time.clamp(0.000001, 6.0);
            let shot_direction = ground_shot_direction(
                &car,
                shot.ball,
                live_ball.location,
                targets,
                shot.shot_direction,
            );
            let contact_target = hitbox_contact_target(car, shot.ball, shot_direction);
            let live_ball_local = car.local_location(live_ball.location);
            let ball_inside_hitbox = live_ball_local.y.abs()
                < BALL_RADIUS + car.hitbox.width * 0.5
                && live_ball_local.z.abs()
                    < BALL_RADIUS + car.hitbox.height * 0.5;

            // When the live ball is already inside the car's front-face
            // envelope, stop chasing a future parser point and drive through
            // the current ball lane at full speed.
            let final_target = if ball_inside_hitbox {
                hitbox_contact_target(car, live_ball.location, shot_direction)
            } else {
                ground_flank_target(
                    &car,
                    shot.ball,
                    contact_target,
                    shot_direction,
                    time_remaining,
                )
            };
            let final_target = cap_in_field(&car, final_target);
            let distance_remaining =
                car.location.flat_distance(final_target) - car.hitbox.length / 3.0;
            let mut speed_required = if ball_inside_hitbox {
                MAX_SPEED
            } else {
                distance_remaining / time_remaining
            };
            let ball_speed_toward_shot = live_ball.velocity.dot(shot_direction);
            if speed_required < 1900.0
                && car.boost > 50.0
                && time_remaining > 2.0
                && distance_remaining > 2560.0
                && ball_speed_toward_shot > 1000.0
            {
                speed_required *= 0.75;
            }

            let local_final_target = car.local_location(final_target);
            let angle_to_target = local_final_target.y.atan2(local_final_target.x).abs();
            let drive = default_drive(
                &car,
                DriveInput {
                    target_speed: speed_required,
                    local_target: local_final_target,
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel,
                },
            );
            let velocity = if drive.car_speed.abs() < 0.001 {
                1.0
            } else {
                drive.car_speed
            };
            let local_offset_target = car.local_location(contact_target.flatten());
            let predicted_velocity = car.velocity + gravity * time_remaining;
            let local_predicted_velocity = car.local(predicted_velocity.flatten());
            let arrival_speed = (velocity.abs() + dodge_impulse(car.velocity, MAX_SPEED))
                .min(MAX_SPEED);
            let speed_limit = if car.boost > 12.0 {
                speed_required.abs()
            } else {
                1900.0
            };
            let contact_geometry = (velocity.abs() < 100.0
                && distance_remaining < car.hitbox.length * 0.5)
                || (local_offset_target.y.abs() < 92.0
                    && local_predicted_velocity.x
                        >= local_offset_target.x - car.hitbox.length / 3.0
                    && local_offset_target.x > 0.0);

            // This is the actual Python ground_shot flip gate. A ground
            // candidate now performs a real jump/release/dodge rather than
            // stopping at a non-jumping drive controller.
            if time_remaining < 0.35
                && time_remaining > 0.10
                && arrival_speed <= speed_limit
                && contact_geometry
                && !car.airborne
            {
                let local_direction = local_offset_target.normalize();
                self.ground_flip = Some(GroundFlip::new(
                    local_direction,
                    angle_to_target > 2.0,
                ));
                self.set_phase(StrikePhase::GroundFlip, now);
                println!(
                    "[VOLT GROUND FLIP] T={:.3} distance={:.1} speed={:.1} required={:.1} angle={:.2} direction=({:.2},{:.2})",
                    time_remaining,
                    distance_remaining,
                    velocity,
                    speed_required,
                    angle_to_target,
                    local_direction.x,
                    local_direction.y,
                );
                return controller_from_drive_output(drive, &car);
            }

            if car.airborne {
                return crate::drive::recover_in_air(
                    &car,
                    boost_accel,
                    Some(live_ball.location),
                );
            }

            let runtime_reachable = runtime_ground_reachability(
                raw_time,
                boost_accel,
                &car,
                (final_target - car.location).normalize(),
                distance_remaining.max(0.0),
            )
            .is_some_and(|_| {
                ground_shot_is_viable(
                    car.hitbox.height,
                    car.airborne,
                    contact_target.z,
                )
            });
            if raw_time <= 0.0
                || (raw_time > 0.35 && distance_remaining > 50.0 && !runtime_reachable)
            {
                if now - self.last_debug_at >= 0.25 {
                    println!(
                        "[VOLT GROUND ABORT] T={:.3} distance={:.1} speed={:.1} reachable={}",
                        raw_time, distance_remaining, velocity, runtime_reachable
                    );
                    self.last_debug_at = now;
                }
                self.reset();
                return ControllerState::default();
            }

            return controller_from_drive_output(drive, &car);
        }

        let time_remaining = shot.time.clamp(0.000001, 6.0);
        let (approach, shot_direction, _) =
            Self::refreshed_jump_geometry(&car, shot, live_ball, targets, gravity, time_remaining);

        let jump_height = if shot.shot_type == ShotType::DoubleJump {
            (approach.z - car.location.z).max(0.0)
        } else {
            shot.ball.z - car.location.z
        };
        let needed_jump_time = if shot.shot_type == ShotType::DoubleJump {
            let Some(grid_time) = grid_jump_time(jump_height, true) else {
                if now - self.last_debug_at >= 0.25 {
                    println!(
                        "[VOLT DOUBLE JUMP ABORT] unreachable contact height={:.1}",
                        jump_height,
                    );
                    self.last_debug_at = now;
                }
                self.reset();
                return ControllerState::default();
            };
            grid_time
        } else {
            get_jump_time(jump_height, car.velocity.z, gravity.z).time as f32
        };
        let time_to_jump = time_remaining - needed_jump_time * 1.15;
        let mut final_target = approach.flatten();
        if shot.shot_type == ShotType::DoubleJump && time_to_jump > 0.0 {
            final_target = ground_flank_target(
                &car,
                shot.ball,
                final_target,
                shot_direction,
                time_to_jump,
            );
        }
        let final_target = cap_in_field(&car, final_target);
        let mut distance_remaining = car.location.flat_distance(final_target);
        if car.airborne {
            distance_remaining = car.local_location(approach.flatten()).x;
        }
        distance_remaining -= car.hitbox.length / 3.0;
        let local_final_target = car.local_location(final_target);
        let mut speed_required = distance_remaining / time_remaining;
        let ball_speed_toward_shot = live_ball.velocity.dot(shot_direction);
        if shot.shot_type == ShotType::Jump
            && speed_required < 1900.0
            && car.boost > 50.0
            && time_remaining > 2.0
            && distance_remaining > 2560.0
            && ball_speed_toward_shot > 1000.0
        {
            speed_required *= 0.75;
        }

        let current_forward_speed = car.local_velocity().x;
        let power_dodge_window = shot.shot_type == ShotType::Jump
            && time_remaining < 0.35
            && time_remaining > 0.10
            && car.local_location(live_ball.location).x > 0.0
            && current_forward_speed > 700.0;
        let mut drive_speed = speed_required;
        if power_dodge_window {
            drive_speed = drive_speed.max(current_forward_speed.min(1200.0));
        }
        // Grid's double_jump reverses toward a target behind the car only
        // when it is badly misaligned and moving slowly; otherwise it never
        // needlessly turns a reachable back-side contact into a brake stop.
        let direction_sign = if shot.shot_type == ShotType::DoubleJump
            && local_final_target.y.atan2(local_final_target.x).abs() >= 1.6
            && current_forward_speed <= 1000.0
            && car.local_location(live_ball.location).x < -250.0
        {
            -1.0
        } else {
            1.0
        };
        drive_speed *= direction_sign;
        // Close-in overshoot is not permission to reverse. If the target has
        // fallen behind because of a stale slice, keep useful forward speed
        // long enough to complete the contact instead of brake-flipping away.
        if distance_remaining <= 0.0 && local_final_target.x > 0.0 {
            drive_speed = drive_speed.max(current_forward_speed.max(850.0));
        }
        drive_speed = cap(drive_speed, -MAX_SPEED, MAX_SPEED);

        let drive = default_drive(
            &car,
            DriveInput {
                target_speed: drive_speed,
                local_target: local_final_target,
                upside_down: false,
                up: None,
                handbrake: false,
                boost_accel,
            },
        );
        let velocity = if drive.car_speed.abs() < 0.001 { 1.0 } else { drive.car_speed };
        let local_offset_target = car.local_location(approach.flatten());
        let predicted_velocity = car.velocity + gravity * time_remaining;
        let local_predicted_velocity = car.local(predicted_velocity.flatten());
        let time_to_contact = distance_remaining
            / (velocity.abs() + dodge_impulse(car.velocity, MAX_SPEED)).max(1.0);

        let close_enough = velocity.abs() < 100.0
            && distance_remaining < car.hitbox.length * 0.5;
        let alignment_speed = if shot.shot_type == ShotType::DoubleJump {
            car.local_velocity().x
        } else {
            local_predicted_velocity.x
        };
        let alignment_sign = if shot.shot_type == ShotType::DoubleJump {
            direction_sign
        } else {
            1.0
        };
        let aligned = local_offset_target.y.abs() < 92.0
            && alignment_sign * alignment_speed
                >= alignment_sign * (local_offset_target.x - car.hitbox.length / 3.0)
            && alignment_sign * local_offset_target.x > 0.0;
        let launch_time_ready = if shot.shot_type == ShotType::DoubleJump {
            time_remaining <= needed_jump_time
        } else {
            time_remaining <= needed_jump_time * 1.08
        };
        let launch_ready = (close_enough || aligned) && launch_time_ready;

        if launch_ready {
            self.quick_shot = time_remaining <= 0.45;
            self.jump_counter = 0;
            self.jump_started_at = None;
            self.dodge_started_at = None;
            self.double_jump_prev_jump = false;
            self.double_jump_sent = false;
            if shot.shot_type == ShotType::Jump {
                self.set_phase(StrikePhase::JumpFlight, now);
            } else {
                self.set_phase(StrikePhase::JumpHold, now);
            }
            return controller_from_drive_output(drive, &car);
        }

        // Keep the actual CLib runtime check, but don't let a marginal
        // under-estimate suppress a close, high-speed contact in the final
        // 0.35 seconds. The original parser already validated this candidate;
        // this is a last-resort abort, not a second independent shot finder.
        let runtime_reachable = runtime_ground_reachability(
            time_remaining,
            boost_accel,
            &car,
            (final_target - car.location).normalize(),
            distance_remaining.max(0.0),
        ).is_some_and(|remaining| {
            if shot.shot_type == ShotType::DoubleJump {
                double_jump_shot_is_viable(
                    remaining,
                    gravity.z,
                    car.velocity.z,
                    car.hitbox.height * 0.5,
                    car.airborne,
                    approach.z,
                    true,
                )
            } else {
                jump_shot_is_viable(
                    remaining,
                    gravity.z,
                    car.velocity.z,
                    car.hitbox.height,
                    car.airborne,
                    approach.z,
                    true,
                )
            }
        });
        let close_late_contact = shot.shot_type == ShotType::Jump
            && time_remaining <= 0.35
            && local_offset_target.x > 0.0
            && current_forward_speed > 650.0
            && distance_remaining <= car.hitbox.length * 1.6;

        if time_remaining <= needed_jump_time
            || (time_to_jump > 0.0
                && distance_remaining > 50.0
                && !runtime_reachable
                && !close_late_contact)
        {
            if now - self.last_debug_at >= 0.25 {
                println!(
                    "[VOLT JUMP ABORT] T={:.3} Tjump={:.3} dist={:.1} speed={:.1} reachable={} close_late={}",
                    time_remaining, needed_jump_time, distance_remaining,
                    current_forward_speed, runtime_reachable, close_late_contact
                );
                self.last_debug_at = now;
            }
            self.reset();
            return if car.airborne {
                crate::drive::recover_in_air(&car, boost_accel, Some(live_ball.location))
            } else {
                ControllerState::default()
            };
        }

        if now - self.last_debug_at >= 0.25 {
            println!(
                "[VOLT JUMP APPROACH] T={:.3} Tjump={:.3} dist={:.1} speed_req={:.1} forward={:.1} lateral={:.1} t_contact={:.3} ready={} boost={:.0}",
                time_remaining, needed_jump_time, distance_remaining,
                speed_required, current_forward_speed, local_offset_target.y,
                time_to_contact, launch_ready, car.boost
            );
            self.last_debug_at = now;
        }

        controller_from_drive_output(drive, &car)
    }

    fn tick_single_jump(
        &mut self,
        car: CarState,
        shot: ShotCandidate,
        live_ball: BallState,
        gravity: Vector,
        boost_accel: f32,
        now: f32,
    ) -> ControllerState {
        let jump_started_at = *self.jump_started_at.get_or_insert(now);
        let jump_elapsed = (now - jump_started_at).max(0.0);
        let time_remaining = shot.time.clamp(-1.0, 6.0);
        let approach = self.current_jump_target;
        let shot_direction = self.current_shot_direction;
        let local_ball = car.local_location(shot.ball).flatten();
        let local_target = car.local_location(approach).flatten();
        let contact_distance = car.location.distance(approach);
        let contact_limit = (BALL_RADIUS + car.hitbox.length) * 1.02;

        if time_remaining <= -MAX_JUMP_SHOT_AGE
            || (!car.airborne && self.jump_counter >= 3)
        {
            if now - self.last_debug_at >= 0.25 {
                println!(
                    "[VOLT JUMP ABORT] phase=flight T={:.3} airborne={} counter={} elapsed={:.3}",
                    time_remaining, car.airborne, self.jump_counter, jump_elapsed
                );
                self.last_debug_at = now;
            }
            self.reset();
            return crate::drive::recover_in_air(&car, boost_accel, Some(live_ball.location));
        }

        let mut predicted_velocity = car.velocity + gravity * time_remaining;
        let mut predicted_position = car.location
            + car.velocity * time_remaining
            + gravity * (0.5 * time_remaining * time_remaining);
        if jump_elapsed <= 1.0e-6 {
            predicted_velocity = predicted_velocity + car.up() * JUMP_SPEED;
            predicted_position = predicted_position + car.up() * (JUMP_SPEED * time_remaining);
        }
        let hf = predicted_velocity.z;
        let tau = (JUMP_MAX_DURATION - jump_elapsed).max(0.0);
        predicted_position = predicted_position
            + car.up() * (JUMP_ACCEL * tau * (time_remaining - 0.5 * tau));

        let mut controller = ControllerState::default();
        if self.jump_counter < 3 && self.dodge_started_at.is_none() {
            let local_shot = car.local(shot_direction.flatten()).flatten().normalize();
            let (_, pd) = default_pd(&car, local_shot, false, None);
            controller.pitch = pd.pitch;
            controller.yaw = pd.yaw;
            controller.roll = pd.roll;

            let hold_s = shot.ball.z > 130.0
                && shot.ball.z > car.location.z + 65.0
                && !self.quick_shot;
            if hold_s {
                controller.pitch = 1.0;
                controller.roll = if pd.yaw.abs() > 0.08 {
                    -0.20 * pd.yaw.signum()
                } else {
                    0.0
                };
            }

            let delta_position = approach - predicted_position;
            let direction = delta_position.normalize();
            if car.forward().dot(direction).abs() > 0.5 && time_remaining > 1.0e-6 {
                let mut delta_velocity = delta_position.dot(car.forward()) / time_remaining;
                if car.boost > 0.0
                    && delta_velocity >= boost_accel * MIN_AIR_BOOST_TIME
                {
                    controller.boost = true;
                    delta_velocity -= boost_accel * MIN_AIR_BOOST_TIME;
                }
                if delta_velocity.abs() > AIR_THROTTLE_ACCEL * MIN_AIR_BOOST_TIME {
                    controller.throttle =
                        (delta_velocity / (AIR_THROTTLE_ACCEL * MIN_AIR_BOOST_TIME))
                            .clamp(-1.0, 1.0);
                }
            }
        }

        if self.jump_counter == 3 {
            let close_enough = contact_distance < contact_limit;
            let target_in_front = local_target.x > 0.0;
            if close_enough && target_in_front {
                let dodge = if local_ball.magnitude() > 1.0 {
                    local_ball.normalize()
                } else {
                    Vector::new(1.0, 0.0, 0.0)
                };
                self.dodge_started_at = Some(now);
                self.jump_counter = 4;
                controller.jump = true;
                controller.pitch = -dodge.x;
                controller.yaw = dodge.y;
                controller.roll = 0.0;
                controller.boost = car.boost > 0.0;
                println!(
                    "[VOLT JUMP FLIP] T={:.3} distance={:.1} local_ball=({:.2},{:.2}) pitch={:.2} yaw={:.2}",
                    time_remaining, contact_distance, dodge.x, dodge.y,
                    controller.pitch, controller.yaw
                );
            } else if time_remaining < 0.12
                && contact_distance > contact_limit + 150.0
            {
                self.jump_counter = 5;
            } else {
                let local_shot = car.local(shot_direction.flatten()).flatten().normalize();
                let (_, pd) = default_pd(&car, local_shot, false, None);
                controller.pitch = pd.pitch;
                controller.yaw = pd.yaw;
                controller.roll = pd.roll;
            }
        } else if self.jump_counter == 4 {
            // The flip was already triggered. Never issue pitch/yaw cancel
            // inputs here; allow the dodge impulse to finish naturally.
        }

        if self.jump_counter == 5 {
            self.reset();
            return crate::drive::recover_in_air(&car, boost_accel, Some(live_ball.location));
        }

        if jump_elapsed <= JUMP_MAX_DURATION
            && (jump_elapsed <= 0.06 || hf <= approach.z)
        {
            controller.jump = true;
        } else if self.jump_counter < 3 {
            self.jump_counter += 1;
        }

        controller
    }

    /// Update the active shot with the newest candidate/prediction and emit
    /// a clean controller state every tick.
    pub fn tick(
        &mut self,
        car: CarState,
        ball: BallState,
        candidate: Option<ShotCandidate>,
        prediction: &BallTrajectory,
        targets: ShotTargets,
        gravity: Vector,
        boost_accel: f32,
        now: f32,
    ) -> ControllerState {
        self.refresh_committed_candidate(candidate, prediction, now, gravity);
        let Some(shot) = self.candidate else {
            return ControllerState::default();
        };
        // Every parser candidate owns the target pair it was evaluated
        // against. This is especially important for find_pass_shot: replacing
        // its receiver lane with the normal goal posts here turns a pass back
        // into an accidental shot at goal during execution.
        let selected_targets = shot.parser.targets;

        if shot.time < -MAX_JUMP_SHOT_AGE {
            self.reset();
            return if car.airborne {
                crate::drive::recover_in_air(&car, boost_accel, Some(ball.location))
            } else {
                ControllerState::default()
            };
        }

        match self.phase {
            StrikePhase::Approach => {
                if shot.shot_type != ShotType::Ground {
                    let time = shot.time.clamp(0.000001, 6.0);
                    let (approach, direction, _) =
                        Self::refreshed_jump_geometry(&car, shot, ball, selected_targets, gravity, time);
                    self.current_jump_target = approach;
                    self.current_shot_direction = direction;
                }
                self.tick_ground_or_jump_approach(
                    car, ball, shot, selected_targets, gravity, boost_accel, now,
                )
            }
            StrikePhase::JumpFlight => {
                self.tick_single_jump(car, shot, ball, gravity, boost_accel, now)
            }
            StrikePhase::GroundFlip => {
                let Some(mut flip) = self.ground_flip.take() else {
                    self.reset();
                    return ControllerState::default();
                };
                let (controller, finished, recover) = flip.tick(now);
                if finished {
                    self.reset();
                    if recover || car.airborne {
                        crate::drive::recover_in_air(&car, boost_accel, Some(ball.location))
                    } else {
                        ControllerState::default()
                    }
                } else {
                    self.ground_flip = Some(flip);
                    controller
                }
            }
            StrikePhase::JumpHold => {
                let time_remaining = shot.time.clamp(-1.0, 6.0);
                // Grid's double_jump holds the first press for 24 120-Hz
                // frames, releases at least one frame, then sends one
                // distinct second press. The previous Rust implementation
                // shortened this to a few 60-ms phases, which often failed
                // to register as a true double jump.
                let elapsed = (now - self.phase_started).max(0.0);
                let jump_frames = (elapsed * 120.0).round().max(0.0) as u32;
                let mut controller = ControllerState::default();

                if jump_frames < 24 {
                    controller.jump = true;
                    self.double_jump_prev_jump = true;
                } else if !self.double_jump_sent && !car.doublejumped {
                    if self.double_jump_prev_jump {
                        controller.jump = false;
                        self.double_jump_prev_jump = false;
                    } else {
                        controller.jump = true;
                        self.double_jump_sent = true;
                        self.double_jump_prev_jump = true;
                    }
                } else {
                    controller.jump = false;
                    self.double_jump_prev_jump = false;
                }

                // Python defaultPD aims at the contact target itself, not
                // merely the goal vector. This keeps the nose on the chosen
                // lower/front hitbox contact throughout the two-jump launch.
                let local_contact = car.local_location(self.current_jump_target);
                let (_, orientation) = default_pd(&car, local_contact, false, None);
                controller.steer = orientation.steer;
                controller.pitch = orientation.pitch;
                controller.yaw = orientation.yaw;
                controller.roll = orientation.roll;

                if time_remaining > 0.0 {
                    let future_position = car.location
                        + car.velocity * time_remaining
                        + gravity * (0.5 * time_remaining * time_remaining);
                    let delta = self.current_jump_target - future_position;
                    let forward_need = delta.dot(car.forward()) / time_remaining;
                    if forward_need > boost_accel * MIN_AIR_BOOST_TIME
                        && car.boost > 0.0
                    {
                        controller.boost = true;
                    } else if forward_need.abs() > 1.0 {
                        controller.throttle = cap(
                            forward_need
                                / (AIR_THROTTLE_ACCEL * MIN_AIR_BOOST_TIME),
                            -1.0,
                            1.0,
                        );
                    }
                }

                if time_remaining <= -0.4 || (!car.airborne && jump_frames >= 30) {
                    self.reset();
                    if car.airborne {
                        crate::drive::recover_in_air(
                            &car, boost_accel, Some(ball.location),
                        )
                    } else {
                        ControllerState::default()
                    }
                } else {
                    controller
                }
            }
            StrikePhase::JumpRelease => {
                let mut controller = align_to_direction(&car, self.current_shot_direction);
                controller.jump = false;
                if now - self.phase_started >= 0.05 {
                    self.set_phase(StrikePhase::SecondJump, now);
                }
                controller
            }
            StrikePhase::SecondJump => {
                let mut controller = align_to_direction(&car, self.current_shot_direction);
                controller.jump = now - self.phase_started < 0.06;
                controller.boost = car.boost > 0.0
                    && car.local(self.current_shot_direction).x > 0.65;
                controller.throttle = 1.0;
                if now - self.phase_started >= 0.06 {
                    self.set_phase(StrikePhase::Flight, now);
                    controller.jump = false;
                }
                controller
            }
            StrikePhase::Flight => {
                if !car.airborne {
                    self.reset();
                    return ControllerState::default();
                }
                let target_delta = self.current_jump_target - car.location;
                let local_target = car.local(target_delta);
                let local_shot = car.local(self.current_shot_direction);
                let aim = if target_delta.magnitude() > 65.0 {
                    target_delta
                } else {
                    self.current_shot_direction
                };
                let mut controller = align_to_direction(&car, aim);
                controller.boost = car.boost > 0.0
                    && local_shot.x > 0.0
                    && local_shot.angle(Vector::new(1.0, 0.0, 0.0)) < 0.48
                    && car.velocity.dot(self.current_shot_direction) < 2180.0;
                controller.throttle = 1.0;
                let _ = local_target;
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
    /// 0 = build speed, 1 = wait for speedflip landing,
    /// 2 = close on ball, 3 = final kickoff dodge until contact.
    phase: u8,
    speedflip: Option<SpeedFlip>,
    phase3_started_at: Option<f32>,
    phase3_flip_started_at: Option<f32>,
    phase3_ball_start: Vector,
    phase3_velocity_start: Vector,
    finished: bool,
}

impl KickoffExecutor {
    pub fn reset(&mut self) {
        self.phase = 0;
        self.speedflip = None;
        self.phase3_started_at = None;
        self.phase3_flip_started_at = None;
        self.phase3_ball_start = Vector::ZERO;
        self.phase3_velocity_start = Vector::ZERO;
        self.finished = false;
    }

    fn face_ball(car: CarState, ball: Vector) -> ControllerState {
        let (_, pd) = default_pd(&car, car.local_location(ball), false, None);
        let mut controller = ControllerState::default();
        controller.steer = pd.steer;
        controller.pitch = pd.pitch;
        controller.yaw = pd.yaw;
        controller.roll = pd.roll;
        controller.throttle = 1.0;
        controller
    }

    fn begin_contact_phase(&mut self, ball: BallState, now: f32) {
        self.phase = 3;
        self.phase3_started_at = Some(now);
        self.phase3_flip_started_at = Some(now);
        self.phase3_ball_start = ball.location;
        self.phase3_velocity_start = ball.velocity;
    }

    pub fn tick_kicker(
        &mut self,
        car: CarState,
        ball: BallState,
        team: u32,
        boost_accel: f32,
        now: f32,
    ) -> ControllerState {
        let _ = boost_accel;
        if self.finished {
            return ControllerState::default();
        }

        let local_ball = car.local_location(ball.location);
        let distance = car.location.flat_distance(ball.location);
        let speed = car.velocity.magnitude();

        match self.phase {
            0 => {
                // Python speedflip_kickoff phase 0: accelerate straight at
                // the ball until 800 uu/s while it remains more than 1800 uu
                // away. This is deliberately not goto's generic speedflip
                // condition (which waits for 600-1600 forward speed and
                // a 3000 uu waypoint).
                let mut controller = Self::face_ball(car, ball.location);
                controller.boost = car.boost > 0.0;
                controller.jump = false;
                if speed > 800.0 && distance > 1800.0 {
                    let right_handed = car.location.x * (if team == 0 { 1.0 } else { -1.0 }) > 0.0;
                    self.speedflip = Some(SpeedFlip::new(right_handed, true));
                    self.phase = 1;
                    controller = self.speedflip.as_mut().unwrap().tick(car, now);
                }
                controller
            }
            1 => {
                if let Some(speedflip) = self.speedflip.as_mut() {
                    let controller = speedflip.tick(car, now);
                    if speedflip.finished() {
                        self.speedflip = None;
                    }
                    return controller;
                }
                // Python phase 1 waits for the speedflip to land before
                // beginning the final straight-line drive.
                if !car.airborne {
                    self.phase = 2;
                }
                ControllerState::default()
            }
            2 => {
                let mut controller = Self::face_ball(car, ball.location);
                controller.boost = speed < 2290.0 && car.boost > 0.0;
                if distance < 1000.0 {
                    self.begin_contact_phase(ball, now);
                }
                controller
            }
            3 => {
                let started = *self.phase3_started_at.get_or_insert(now);
                let flip_started = *self.phase3_flip_started_at.get_or_insert(now);
                let mut controller = Self::face_ball(car, ball.location);
                controller.boost = car.boost > 0.0
                    && local_ball.y.atan2(local_ball.x).abs() < 0.35;

                // Exact Python kickoff dodge: jump, three-frame release,
                // then one forward dodge press, followed by a clean release.
                let flip_time = (now - flip_started).max(0.0);
                if flip_time < 1.0 / 60.0 {
                    controller.jump = true;
                    controller.pitch = 0.0;
                } else if flip_time < 4.0 / 60.0 {
                    controller.jump = false;
                    controller.pitch = 0.0;
                } else if flip_time < 5.0 / 60.0 {
                    controller.jump = true;
                    controller.pitch = -1.0;
                } else {
                    controller.jump = false;
                    controller.pitch = 0.0;
                }

                let ball_displacement = (ball.location - self.phase3_ball_start).magnitude();
                let velocity_change = (ball.velocity - self.phase3_velocity_start).magnitude();
                let likely_contact = (ball_displacement > 100.0 && velocity_change > 250.0)
                    || velocity_change > 500.0;
                if likely_contact || now - started > 2.0 {
                    self.finished = true;
                    self.speedflip = None;
                    return ControllerState::default();
                }
                controller
            }
            _ => {
                self.finished = true;
                ControllerState::default()
            }
        }
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

    fn ground_candidate(time: f32, ball: Vector) -> ShotCandidate {
        let targets = ShotTargets {
            left: Vector::new(-820.0, 5120.0, 350.0),
            right: Vector::new(820.0, 5120.0, 350.0),
        };
        let parser = crate::shot_parser::ParsedShot {
            found: true,
            fast: false,
            shot_type: Some(ShotType::Ground),
            targets,
            ball_offset: ball,
            shot_vector: Vector::new(0.0, 1.0, 0.0),
            approach_target: ball - Vector::new(0.0, 150.0, 0.0),
            time,
        };
        ShotCandidate {
            time,
            ball,
            ball_velocity: Vector::ZERO,
            approach: parser.approach_target,
            shot_direction: parser.shot_vector,
            shot_type: ShotType::Ground,
            targeted: true,
            can_jump: true,
            fast: false,
            score: time,
            parser,
        }
    }

    fn jump_candidate(time: f32, ball: Vector) -> ShotCandidate {
        let targets = ShotTargets {
            left: Vector::new(-820.0, 5120.0, 350.0),
            right: Vector::new(820.0, 5120.0, 350.0),
        };
        let parser = crate::shot_parser::ParsedShot {
            found: true,
            fast: false,
            shot_type: Some(ShotType::Jump),
            targets,
            ball_offset: ball,
            shot_vector: Vector::new(1.0, 0.0, 0.0),
            approach_target: ball - Vector::new(150.0, 0.0, 0.0),
            time,
        };
        ShotCandidate {
            time,
            ball,
            ball_velocity: Vector::ZERO,
            approach: parser.approach_target,
            shot_direction: parser.shot_vector,
            shot_type: ShotType::Jump,
            targeted: true,
            can_jump: true,
            fast: false,
            score: time,
            parser,
        }
    }

    #[test]
    fn relative_goal_aim_uses_python_blue_and_orange_side_convention() {
        let targets = ShotTargets {
            left: Vector::new(-820.0, 5120.0, 350.0),
            right: Vector::new(820.0, 5120.0, 350.0),
        };
        let mut blue = car(Vector::ZERO, Vector::ZERO);
        blue.team = 0;
        let mut orange = blue;
        orange.team = 1;

        assert_eq!(
            ground_aim_point(&blue, Vector::new(0.0, -1000.0, 100.0), targets).z,
            310.0,
        );
        assert_eq!(
            ground_aim_point(&blue, Vector::new(0.0, 1000.0, 100.0), targets).z,
            470.0,
        );
        assert_eq!(
            ground_aim_point(&orange, Vector::new(0.0, 1000.0, 100.0), targets).z,
            310.0,
        );
        assert_eq!(
            ground_aim_point(&orange, Vector::new(0.0, -1000.0, 100.0), targets).z,
            470.0,
        );
    }

    #[test]
    fn kickoff_uses_python_speed_threshold_and_team_mirrored_handedness() {
        let mut blue = car(Vector::new(100.0, 0.0, 17.0), Vector::new(900.0, 0.0, 0.0));
        let ball = BallState { location: Vector::new(2200.0, 0.0, 92.0), velocity: Vector::ZERO };
        let mut kickoff = KickoffExecutor::default();

        let output = kickoff.tick_kicker(blue, ball, 0, 991.6667, 1.0);
        assert_eq!(kickoff.phase, 1);
        assert!(output.steer.abs() > 0.5);
        assert!(output.boost);

        let speedflip = kickoff.speedflip.as_ref().expect("kickoff must start speedflip");
        assert_eq!(speedflip.direction, 1.0);

        blue.team = 1;
        let mut orange_kickoff = KickoffExecutor::default();
        let _ = orange_kickoff.tick_kicker(blue, ball, 1, 991.6667, 1.0);
        assert_eq!(
            orange_kickoff.speedflip.as_ref().unwrap().direction,
            -1.0,
        );
    }

    #[test]
    fn kickoff_waits_for_speedflip_landing_then_runs_contact_dodge() {
        let mut kickoff = KickoffExecutor { phase: 1, ..Default::default() };
        let airborne = {
            let mut value = car(Vector::ZERO, Vector::new(900.0, 0.0, 200.0));
            value.airborne = true;
            value
        };
        let ball = BallState { location: Vector::new(1200.0, 0.0, 92.0), velocity: Vector::ZERO };

        let waiting = kickoff.tick_kicker(airborne, ball, 0, 991.6667, 1.0);
        assert_eq!(kickoff.phase, 1);
        assert!(!waiting.jump);

        let grounded = car(Vector::ZERO, Vector::new(1400.0, 0.0, 0.0));
        let landing = kickoff.tick_kicker(grounded, ball, 0, 991.6667, 1.01);
        assert_eq!(kickoff.phase, 2);
        assert!(!landing.jump);

        let close_ball = BallState { location: Vector::new(500.0, 0.0, 92.0), velocity: Vector::ZERO };
        let _ = kickoff.tick_kicker(grounded, close_ball, 0, 991.6667, 1.02);
        assert_eq!(kickoff.phase, 3);
        assert_eq!(kickoff.phase3_started_at, Some(1.02));

        let first_dodge = kickoff.tick_kicker(grounded, close_ball, 0, 991.6667, 1.025);
        assert!(first_dodge.jump);
        assert_eq!(first_dodge.pitch, 0.0);

        let released = kickoff.tick_kicker(grounded, close_ball, 0, 991.6667, 1.07);
        assert!(!released.jump);

        let dodge = kickoff.tick_kicker(grounded, close_ball, 0, 991.6667, 1.09);
        assert!(dodge.jump);
        assert_eq!(dodge.pitch, -1.0);

        let after_dodge = kickoff.tick_kicker(grounded, close_ball, 0, 991.6667, 1.11);
        assert!(!after_dodge.jump);
    }

    #[test]
    fn kickoff_finishes_on_ball_velocity_change_or_two_second_timeout() {
        let mut kickoff = KickoffExecutor {
            phase: 3,
            phase3_started_at: Some(1.0),
            phase3_flip_started_at: Some(1.0),
            phase3_ball_start: Vector::ZERO,
            phase3_velocity_start: Vector::ZERO,
            ..Default::default()
        };
        let car = car(Vector::ZERO, Vector::ZERO);
        let hit_ball = BallState {
            location: Vector::new(0.0, 120.0, 0.0),
            velocity: Vector::new(0.0, 300.0, 0.0),
        };
        let _ = kickoff.tick_kicker(car, hit_ball, 0, 991.6667, 1.1);
        assert!(kickoff.finished);
        assert!(!kickoff.tick_kicker(car, hit_ball, 0, 991.6667, 1.11).jump);

        kickoff.reset();
        kickoff.phase = 3;
        kickoff.phase3_started_at = Some(1.0);
        kickoff.phase3_flip_started_at = Some(1.0);
        let _ = kickoff.tick_kicker(car, BallState::default(), 0, 991.6667, 3.01);
        assert!(kickoff.finished);
    }

    #[test]
    fn double_jump_holds_24_frames_releases_then_sends_one_second_press() {
        let mut me = car(Vector::new(0.0, 0.0, 100.0), Vector::new(0.0, 800.0, 120.0));
        me.airborne = true;
        me.orientation = Matrix3::new(0.0, 1.5707963, 0.0);
        let ball = Vector::new(0.0, 700.0, 250.0);
        let mut candidate = jump_candidate(1.2, ball);
        candidate.shot_type = ShotType::DoubleJump;
        candidate.parser.shot_type = Some(ShotType::DoubleJump);

        let targets = candidate.parser.targets;
        let prediction = BallTrajectory { slices: Vec::new() };
        let mut executor = StrikeExecutor::new();
        executor.candidate = Some(candidate);
        executor.phase = StrikePhase::JumpHold;
        executor.phase_started = 1.0;
        executor.last_tick = Some(1.0);
        executor.current_jump_target = Vector::new(0.0, 850.0, 250.0);
        executor.current_shot_direction = Vector::new(0.0, 1.0, 0.0);

        let first_jump = executor.tick(
            me, BallState::default(), None, &prediction, targets,
            Vector::new(0.0, 0.0, -650.0), 991.6667, 1.0083333,
        );
        assert!(first_jump.jump);

        let release = executor.tick(
            me, BallState::default(), None, &prediction, targets,
            Vector::new(0.0, 0.0, -650.0), 991.6667, 1.2,
        );
        assert!(!release.jump);
        assert!(!executor.double_jump_sent);

        let second_jump = executor.tick(
            me, BallState::default(), None, &prediction, targets,
            Vector::new(0.0, 0.0, -650.0), 991.6667, 1.2083333,
        );
        assert!(second_jump.jump);
        assert!(executor.double_jump_sent);

        let release_after_second = executor.tick(
            me, BallState::default(), None, &prediction, targets,
            Vector::new(0.0, 0.0, -650.0), 991.6667, 1.2166667,
        );
        assert!(!release_after_second.jump);
    }

    #[test]
    fn ground_shot_uses_the_python_timed_directional_flip() {
        let mut me = car(Vector::ZERO, Vector::new(0.0, 1500.0, 0.0));
        me.orientation = Matrix3::new(0.0, 1.5707963, 0.0);
        let live_ball = BallState {
            location: Vector::new(0.0, 180.0, 50.0),
            velocity: Vector::ZERO,
        };
        let candidate = ground_candidate(0.20, live_ball.location);
        let targets = candidate.parser.targets;
        let prediction = BallTrajectory { slices: Vec::new() };
        let mut executor = StrikeExecutor::new();

        let _ = executor.tick(
            me,
            live_ball,
            Some(candidate),
            &prediction,
            targets,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            1.0,
        );
        assert_eq!(executor.phase, StrikePhase::GroundFlip);
        assert!(executor.ground_flip.is_some());

        let initial_jump = executor.tick(
            me,
            live_ball,
            None,
            &prediction,
            targets,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            1.0083333,
        );
        assert!(initial_jump.jump);

        let release = executor.tick(
            me,
            live_ball,
            None,
            &prediction,
            targets,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            1.09,
        );
        assert!(!release.jump);

        let mut dodge = ControllerState::default();
        for i in 0..3 {
            dodge = executor.tick(
                me,
                live_ball,
                None,
                &prediction,
                targets,
                Vector::new(0.0, 0.0, -650.0),
                991.6667,
                1.09 + (i as f32 + 1.0) / 120.0,
            );
        }
        dodge = executor.tick(
            me,
            live_ball,
            None,
            &prediction,
            targets,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            1.13,
        );
        assert!(dodge.jump);
        assert!(dodge.pitch.abs() > 0.5 || dodge.yaw.abs() > 0.5);
    }

    #[test]
    fn approach_candidate_survives_a_transient_empty_finder_result() {
        let me = car(Vector::ZERO, Vector::ZERO);
        let candidate = jump_candidate(4.5, Vector::new(1200.0, 0.0, 180.0));
        let prediction = crate::prediction::BallTrajectory {
            slices: vec![
                crate::prediction::BallSlice {
                    time: 4.48,
                    location: Vector::new(1200.0, 0.0, 180.0),
                    velocity: Vector::ZERO,
                },
                crate::prediction::BallSlice {
                    time: 4.50,
                    location: Vector::new(1201.0, 0.0, 180.0),
                    velocity: Vector::ZERO,
                },
            ],
        };
        let targets = candidate.parser.targets;
        let mut executor = StrikeExecutor::new();

        let _ = executor.tick(
            me, BallState::default(), Some(candidate), &prediction,
            targets, Vector::new(0.0, 0.0, -650.0), 991.6667, 1.0,
        );
        assert!(executor.active());
        assert!(executor.is_approaching());

        let _ = executor.tick(
            me, BallState::default(), None, &prediction,
            targets, Vector::new(0.0, 0.0, -650.0), 991.6667, 1.016,
        );
        assert!(executor.active());
        assert!(executor.active_candidate().is_some());
        assert!(executor.is_approaching());
    }

    #[test]
    fn jump_dodge_is_not_blocked_by_stale_has_dodged_flag_or_cancelled() {
        let mut me = car(Vector::ZERO, Vector::ZERO);
        me.airborne = true;
        me.has_dodged = true;
        let candidate = jump_candidate(0.21, Vector::new(180.0, 0.0, 110.0));
        let targets = candidate.parser.targets;
        let mut executor = StrikeExecutor::new();
        executor.candidate = Some(candidate);
        executor.phase = StrikePhase::JumpFlight;
        executor.jump_started_at = Some(0.0);
        executor.jump_counter = 3;
        executor.current_jump_target = Vector::new(150.0, 0.0, 0.0);
        executor.current_shot_direction = Vector::new(1.0, 0.0, 0.0);
        executor.last_tick = Some(0.19);

        let controller = executor.tick(
            me,
            BallState { location: candidate.ball, velocity: Vector::ZERO },
            None,
            &crate::prediction::BallTrajectory { slices: vec![] },
            targets,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            0.20,
        );

        assert!(controller.jump, "jump/dodge press must be sent");
        assert!(controller.pitch < -0.9, "front dodge must use a forward nose input");
        assert!(controller.roll.abs() < 0.01, "no flip-cancel roll may be injected");
        assert_eq!(executor.jump_counter, 4);
        assert!(executor.dodge_started_at.is_some());
    }

    #[test]
    fn teammate_role_yield_cannot_cancel_a_launched_jump() {
        let mut executor = StrikeExecutor::new();
        executor.candidate = Some(jump_candidate(
            0.35,
            Vector::new(180.0, 0.0, 110.0),
        ));
        executor.phase = StrikePhase::JumpFlight;
        executor.jump_started_at = Some(0.0);

        executor.cancel_approach();

        assert!(executor.active());
        assert!(!executor.is_approaching());
        assert_eq!(executor.phase, StrikePhase::JumpFlight);
    }

    #[test]
    fn first_jump_frame_emits_jump_instead_of_a_blank_controller() {
        let mut me = car(Vector::ZERO, Vector::ZERO);
        me.airborne = true;
        let candidate = jump_candidate(0.45, Vector::new(260.0, 0.0, 140.0));
        let targets = candidate.parser.targets;
        let mut executor = StrikeExecutor::new();
        executor.candidate = Some(candidate);
        executor.phase = StrikePhase::JumpFlight;
        executor.current_jump_target = Vector::new(200.0, 0.0, 0.0);
        executor.current_shot_direction = Vector::new(1.0, 0.0, 0.0);
        executor.last_tick = Some(0.0);

        let controller = executor.tick(
            me,
            BallState { location: candidate.ball, velocity: Vector::ZERO },
            None,
            &crate::prediction::BallTrajectory { slices: vec![] },
            targets,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            0.008333,
        );

        assert!(controller.jump, "first jump impulse must be emitted on the first flight tick");
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
