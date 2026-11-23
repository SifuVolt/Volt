//! Party Cannon-style aerial shot planning and execution.
//!
//! The planner evaluates RLBot's collision-aware ball slices with the same
//! boost-budget validation used by VirxERLU-CLib. Contact geometry follows
//! Party Cannon's hitbox-aware nose offset and predicts the outgoing ball path
//! before choosing the car's target point.

use crate::{
    control::{default_drive, default_pd, DriveInput},
    defense::GridDefense,
    math::Vector,
    prediction::{BallSlice, BallTrajectory},
    shot_physics::{aerial_shot_is_viable, ShotType},
    shot_parser::{parse_slice_for_shot, parse_slice_for_shot_with_target, ShotTargets},
    shot_planner::score_aerial_candidate,
    state::CarState,
};
use rlbot::flat::ControllerState;

const MAX_SPEED: f32 = 2300.0;
const AIR_THROTTLE_ACCEL: f32 = 66.667;
const JUMP_VEL: f32 = 291.667;
const JUMP_ACCEL: f32 = 1458.333;
const JUMP_DURATION: f32 = 0.2;
const BOOST_CONSUMPTION: f32 = 33.333_333;
// Aim the nose at the lower-middle of the ball instead of its centre.
const AERIAL_CONTACT_DROP: f32 = 46.375;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AerialCandidate {
    pub slice: BallSlice,
    pub shot_target: Vector,
    pub target_location: Vector,
    pub shot_direction: Vector,
    pub double_jump: bool,
    pub fast: bool,
    pub estimated_boost: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AerialShot {
    pub started_at: f32,
    pub last_update: f32,
    pub intercept_time: f32,
    pub ball: Vector,
    pub slice_velocity: Vector,
    pub slice_time: f32,
    pub shot_target: Vector,
    pub target_location: Vector,
    pub shot_direction: Vector,
    pub up: Vector,
    pub double_jump: bool,
    pub fast_aerial: bool,
    pub jump_type_fast: bool,
    pub backboard_mode: bool,
    pub aerialing: bool,
    pub elapsed: f32,
    pub jump_release_frames: u8,
    pub second_jump_frames: u8,
    pub jumped: bool,
    pub dodging: bool,
    pub double_jump_complete: bool,
    pub first_jump_impulse_applied: bool,
    pub double_jump_time: Option<f32>,
    pub side_wall_launched: bool,
    pub side_wall_active: bool,
    pub ceiling: bool,
    pub finished: bool,
}

impl AerialShot {
    pub fn new(
        car: CarState,
        candidate: AerialCandidate,
        game_time: f32,
        _gravity: Vector,
        _boost_accel: f32,
    ) -> Self {
        let ceiling = car.location.z > 2044.0 - car.hitbox.height * 2.0 && !car.jumped;
        let target_location = if ceiling {
            candidate.target_location - Vector::new(0.0, 0.0, 92.0)
        } else {
            candidate.target_location
        };

        Self {
            started_at: game_time,
            last_update: game_time,
            intercept_time: game_time + candidate.slice.time,
            ball: candidate.slice.location,
            slice_velocity: candidate.slice.velocity,
            slice_time: candidate.slice.time,
            shot_target: candidate.shot_target,
            target_location,
            shot_direction: candidate.shot_direction,
            up: car.up(),
            double_jump: candidate.fast,
            fast_aerial: candidate.fast,
            jump_type_fast: false,
            backboard_mode: false,
            aerialing: false,
            elapsed: 0.0,
            jump_release_frames: 0,
            second_jump_frames: 0,
            jumped: car.airborne,
            dodging: false,
            double_jump_complete: false,
            first_jump_impulse_applied: false,
            double_jump_time: None,
            side_wall_launched: false,
            side_wall_active: false,
            ceiling,
            finished: false,
        }
    }

    /// Refresh the committed aerial's contact from the latest absolute
    /// prediction, matching Python Aerial.run's nearest-timestamp selection.
    /// The intercept itself remains committed; only its predicted slice and
    /// hitbox-aware contact geometry are refreshed.
    pub fn refresh_prediction(
        &mut self,
        car: CarState,
        prediction: &BallTrajectory,
        game_time: f32,
        goal: Vector,
        gravity: Vector,
        live_ball_height: f32,
    ) {
        let time_remaining = self.intercept_time - game_time;
        if prediction.slices.is_empty() || time_remaining <= 0.0 {
            return;
        }
        let Some(slice) = prediction.slices.iter().copied().min_by(|a, b| {
            (a.time - time_remaining).abs().total_cmp(&(b.time - time_remaining).abs())
        }) else {
            return;
        };

        self.ball = slice.location;
        self.slice_velocity = slice.velocity;
        self.slice_time = slice.time;
        self.shot_target = goal;
        // Python disables the fast sequence if either the live ball or the
        // committed prediction has fallen to 350 uu or lower.
        if self.fast_aerial && (self.ball.z <= 350.0 || live_ball_height <= 350.0) {
            self.fast_aerial = false;
        }
        self.ceiling = car.location.z > 2044.0 - car.hitbox.height * 2.0 && !car.jumped;

        let foe_goal_y = if car.team == 0 { 5120.0 } else { -5120.0 };
        let foe_goal_center = Vector::new(0.0, foe_goal_y, 300.0);
        let own_goal_center = Vector::new(0.0, -foe_goal_y, 300.0);

        let (mut target_location, mut shot_direction) =
            contact_geometry(car, slice, goal, gravity);

        // Python Aerial.run replaces the normal attacking vector with a safe
        // outward clear when the predicted contact is within 2600 units of
        // our own goal, then applies the same wall correction and hitbox offset.
        if self.ball.flatten().distance(own_goal_center.flatten()) < 2600.0 {
            let clear_direction = (self.ball - own_goal_center).flatten().normalize();
            if clear_direction.magnitude() > 0.001 {
                let approach_distance = 92.75 + car.hitbox.length * 0.5;
                shot_direction = wall_shot_adjust_aerial(
                    self.ball, clear_direction, approach_distance,
                ).0;
                let hitbox_offset = car.forward() * car.hitbox.offset.x
                    + car.right() * car.hitbox.offset.y
                    + car.up() * car.hitbox.offset.z;
                target_location =
                    self.ball - shot_direction * approach_distance - hitbox_offset;
            }
        }

        // The nose should meet the lower-middle of the ball (halfway from
        // centre to bottom), not aim through the exact centre.
        target_location.z -= AERIAL_CONTACT_DROP;

        if self.ceiling {
            target_location.z -= 92.0;
        }
        self.target_location = target_location;
        self.shot_direction = shot_direction;

        // Backboard mode is only active for an airborne car when the ball is
        // genuinely rebounding away from the opponent goal, not merely
        // travelling toward the backboard.
        let goal_direction = (foe_goal_center - self.ball).normalize();
        let towards_goal = slice.velocity.dot(goal_direction);
        self.backboard_mode = car.airborne
            && (self.ball.y - foe_goal_center.y).abs() < 1100.0
            && self.ball.z > 450.0
            && self.ball.x.abs() < 3500.0
            && towards_goal < -150.0
            && 0.15 < time_remaining
            && time_remaining < 2.0;
    }

    fn run_side_wall_takeoff(
        &mut self,
        car: CarState,
        game_time: f32,
        time_remaining: f32,
        boost_accel: f32,
    ) -> Option<ControllerState> {
        self.side_wall_active = false;
        let ball = self.ball;
        let wall_side = if ball.x >= 0.0 { 1.0 } else { -1.0 };
        let wall_gap = 4096.0 - ball.x.abs() - 92.75;
        if !(-5.0..=700.0).contains(&wall_gap) || ball.z < 300.0 {
            return None;
        }

        let attached_wall = if car.location.x.abs() > 3500.0
            && car.up().x.abs() > 0.65
            && car.up().x * car.location.x < 0.0
        {
            Some(car.location.x.signum())
        } else {
            None
        };
        if car.airborne && attached_wall.is_none() {
            return None;
        }
        if attached_wall.is_some_and(|side| side != wall_side) || time_remaining <= 0.35 {
            return None;
        }
        self.side_wall_active = true;

        let wall_x = wall_side * 3850.0;
        let lane_y = self.target_location.y;
        let climb_z = (ball.z - 185.0).clamp(180.0, 1900.0);
        let (drive_target, attached) = if attached_wall.is_some() {
            (Vector::new(wall_x, lane_y, climb_z), true)
        } else {
            (Vector::new(wall_x, ball.y, car.location.z.max(100.0)), false)
        };
        let local_target = car.local_location(drive_target);
        let distance = car.location.distance(drive_target);
        let time_for_drive = (time_remaining - 0.20).max(0.18);
        let mut speed_required = (distance / time_for_drive).clamp(900.0, MAX_SPEED);
        let local_angle = local_target.y.atan2(local_target.x).abs();
        let turn_cap = if local_angle > 1.75 { 900.0 }
            else if local_angle > 1.30 { 1200.0 }
            else if local_angle > 0.80 { 1650.0 }
            else { MAX_SPEED };
        speed_required = speed_required.min(turn_cap);

        let wall_up = Vector::new(-wall_side, 0.0, 0.0);
        let mut drive = default_drive(
            &car,
            DriveInput {
                target_speed: speed_required,
                local_target,
                upside_down: false,
                up: if attached { Some(car.local(wall_up)) } else { None },
                handbrake: false,
                boost_accel,
            },
        );
        let boost_angle = if attached { drive.target_angles.yaw.abs() } else { local_angle };
        let needs_boost = car.boost > 0.0
            && distance > 550.0
            && car.velocity.magnitude() < speed_required - 100.0
            && boost_angle < 0.65
            && !drive.handbrake.unwrap_or(false);
        drive.boost = Some(needs_boost);
        if needs_boost { drive.throttle = Some(1.0); }

        let mut controller = ControllerState::default();
        controller.steer = drive.controls.steer;
        controller.pitch = drive.controls.pitch;
        controller.yaw = drive.controls.yaw;
        controller.roll = drive.controls.roll;
        controller.throttle = drive.throttle.unwrap_or(0.0);
        controller.boost = drive.boost.unwrap_or(false);
        controller.handbrake = drive.handbrake.unwrap_or(false);

        if !attached { return Some(controller); }
        let height_error = (car.location.z - climb_z).abs();
        let lane_error = (car.location.y - lane_y).abs();
        let launch_ready = (height_error < 190.0 && lane_error < 420.0)
            || distance < 330.0
            || (time_remaining < 0.55 && distance < 650.0);
        if !launch_ready { return Some(controller); }

        self.side_wall_launched = true;
        self.side_wall_active = false;
        self.aerialing = true;
        self.jump_type_fast = self.fast_aerial;
        self.started_at = game_time;
        self.elapsed = 0.0;
        self.up = car.up();
        self.jumped = false;
        self.first_jump_impulse_applied = false;
        self.double_jump_complete = false;
        self.double_jump_time = None;
        self.jump_release_frames = 0;
        self.second_jump_frames = 0;
        self.dodging = false;
        None
    }

    fn party_cannon_boost_estimate(
        &self,
        car: CarState,
        target: Vector,
        time_remaining: f32,
        final_position: Vector,
        final_velocity: Vector,
        boost_accel: f32,
    ) -> Option<f32> {
        if time_remaining <= 1.0 / 120.0 {
            return Some(0.0);
        }
        let offset = target - final_position;
        let distance = offset.magnitude();
        if distance <= 1.0 {
            return Some(0.0);
        }
        let direction = offset / distance;
        let angle = direction.dot(car.forward()).clamp(-1.0, 1.0).acos().max(0.0001);
        let turn_time = (0.6 * (2.0 * (angle / 9.0).sqrt()))
            .min((time_remaining - 1.0 / 120.0).max(0.0));
        let tau1 = turn_time * (1.0 - 0.4 / angle).clamp(0.0, 1.0);
        let usable_time = (time_remaining - tau1).max(1.0 / 120.0);
        let total_acceleration = boost_accel + AIR_THROTTLE_ACCEL;
        let required_acceleration = 2.0 * distance / (usable_time * usable_time);
        let ratio = required_acceleration / total_acceleration.max(1.0);
        if ratio.abs() >= 0.9 {
            return None;
        }
        let tau2 = time_remaining
            - (time_remaining - tau1) * (1.0 - ratio.clamp(0.0, 1.0)).sqrt();
        let boost_time = (tau2 - tau1).max(0.0);
        let boost_estimate = boost_time * BOOST_CONSUMPTION;
        let velocity_estimate =
            final_velocity + direction * (total_acceleration * boost_time);
        if velocity_estimate.magnitude() >= MAX_SPEED * 0.9
            || boost_estimate >= car.boost.max(0.0) * 0.9
        {
            None
        } else {
            Some(boost_estimate)
        }
    }

    pub fn run(
        &mut self,
        car: CarState,
        game_time: f32,
        gravity: Vector,
        boost_accel: f32,
        _steer: f32,
    ) -> ControllerState {
        self.last_update = game_time;
        let mut controls = ControllerState::default();
        let time_remaining = self.intercept_time - game_time;
        if self.finished {
            return controls;
        }

        // Python only attempts the side-wall route before the aerial
        // jump sequence starts; once aerialing owns the car, never re-enter
        // the wall-drive phase on subsequent packets.
        if !self.aerialing && !self.side_wall_launched && !self.jumped && !self.dodging {
            if let Some(wall_controls) =
                self.run_side_wall_takeoff(car, game_time, time_remaining, boost_accel)
            {
                return wall_controls;
            }
        }

        // Python Aerial.run starts the jump immediately from the floor. It
        // does not wait for a custom steering threshold or re-select single
        // versus fast aerial from a second, unrelated boost estimate.
        if !self.aerialing {
            self.aerialing = true;
            if car.airborne {
                self.jumped = true;
            } else {
                self.started_at = game_time;
                self.jump_type_fast = self.fast_aerial;
                self.jumped = false;
                self.first_jump_impulse_applied = false;
                self.double_jump_complete = false;
                self.double_jump_time = None;
                self.jump_release_frames = 0;
                self.second_jump_frames = 0;
            }
        }

        self.elapsed = if self.jumped {
            self.elapsed
        } else {
            (game_time - self.started_at).max(0.0)
        };
        let jump_elapsed = self.elapsed;
        // Python begins with the live ballistic state. Only the impulses that
        // have not yet appeared in the live packet are added to the forecast.
        let mut final_velocity = car.velocity + gravity * time_remaining;
        let mut final_position =
            car.location + car.velocity * time_remaining + gravity * (0.5 * time_remaining * time_remaining);
        let up = car.up();

        if !self.jumped {
            let tau = (JUMP_DURATION - jump_elapsed).max(0.0);
            if !self.first_jump_impulse_applied {
                final_velocity = final_velocity + up * JUMP_VEL;
                final_position = final_position + up * (JUMP_VEL * time_remaining);
                self.first_jump_impulse_applied = true;
            }
            final_velocity = final_velocity + up * (JUMP_ACCEL * tau);
            final_position = final_position
                + up * (JUMP_ACCEL * tau * (time_remaining - 0.5 * tau));

            if self.jump_type_fast {
                if car.doublejumped {
                    self.double_jump_complete = true;
                }
                if !self.double_jump_complete {
                    let second_jump_delay = if let Some(double_jump_time) = self.double_jump_time {
                        (double_jump_time - game_time).max(0.0)
                    } else if jump_elapsed <= JUMP_DURATION {
                        tau + 3.0 / 120.0
                    } else {
                        (3_i32 - self.jump_release_frames as i32).max(0) as f32 / 120.0
                    };
                    if time_remaining >= second_jump_delay {
                        final_velocity = final_velocity + up * JUMP_VEL;
                        final_position = final_position
                            + up * (JUMP_VEL * (time_remaining - second_jump_delay));
                    }
                }
            }

            if jump_elapsed <= JUMP_DURATION {
                controls.jump = true;
            } else if self.jump_type_fast && self.jump_release_frames < 3 {
                self.jump_release_frames += 1;
                controls.jump = false;
            } else if self.jump_type_fast && self.second_jump_frames < 3 {
                self.second_jump_frames += 1;
                controls.jump = true;
                self.dodging = true;
                if self.second_jump_frames == 1 && self.double_jump_time.is_none() {
                    self.double_jump_time = Some(game_time);
                }
            } else {
                self.dodging = false;
                self.jumped = true;
            }
        }

        self.ceiling = car.location.z > 2044.0 - car.hitbox.height * 2.0 && !car.jumped;
        let fast_pitchback = self.jump_type_fast
            && !self.jumped
            && 0.02 <= jump_elapsed
            && jump_elapsed <= 0.16
            && time_remaining > 0.30
            && self.target_location.z - final_position.z > 80.0
            && car.boost > 0.0
            && !self.backboard_mode
            && !self.ceiling
            && GridDefense::landing_plane(car.location, car.velocity, gravity.z) != 4;

        if fast_pitchback {
            let pitchback_time = time_remaining
                .min(0.16 - jump_elapsed)
                .min(car.boost / BOOST_CONSUMPTION)
                .max(0.0);
            let pitchback_angle: f32 = 0.35;
            let pitchback_direction =
                (car.forward() * pitchback_angle.cos() + up * pitchback_angle.sin()).normalize();
            let acceleration = pitchback_direction * boost_accel;
            final_velocity = final_velocity + acceleration * pitchback_time;
            final_position = final_position
                + acceleration * (pitchback_time * (time_remaining - 0.5 * pitchback_time));
        }

        let offset = self.target_location - final_position;
        let offset_length = offset.magnitude();
        let forward = car.forward();
        let aim_vector = if offset_length > 50.0 { offset } else { self.shot_direction };
        let local_aim = car.local(aim_vector);
        let (upside_down, up_reference) = if self.backboard_mode {
            (self.ball.z < car.location.z - 80.0, None)
        } else if GridDefense::landing_plane(car.location, car.velocity, gravity.z) == 4 {
            (true, None)
        } else {
            let up_reference = if self.jumped {
                (self.ball - car.location).normalize()
            } else {
                Vector::new(0.0, 0.0, 1.0)
            };
            (false, Some(car.local(up_reference)))
        };
        let (_, pd) = default_pd(&car, local_aim, upside_down, up_reference);
        controls.pitch = if fast_pitchback { 1.0 } else { pd.pitch };
        controls.yaw = pd.yaw;
        controls.roll = pd.roll;

        if self.dodging {
            controls.steer = 0.0;
            controls.pitch = 0.0;
            controls.yaw = 0.0;
            controls.roll = 0.0;
        }

        // The source checks boost budget for diagnostics/selection but keeps
        // tracking the contact even when the estimate is rejected.
        if time_remaining > 0.25 && !self.dodging {
            let _ = self.party_cannon_boost_estimate(
                car, self.target_location, time_remaining, final_position,
                final_velocity, boost_accel,
            );
        }

        if fast_pitchback && car.boost > 0.0 {
            controls.boost = true;
            controls.throttle = 1.0;
        } else if time_remaining > 0.0 && offset_length > 1.0 {
            let forward_need = offset.dot(forward) / time_remaining;
            let contact_direction = offset.normalize();
            let forward_angle = forward
                .dot(contact_direction)
                .clamp(-1.0, 1.0)
                .acos();
            controls.boost = forward_need
                >= (boost_accel + AIR_THROTTLE_ACCEL) * (13.0 / 120.0)
                && forward_angle < 0.4
                && car.boost > 0.0;
            controls.throttle = (forward_need
                / (AIR_THROTTLE_ACCEL * (1.0 / 120.0)))
                .clamp(-1.0, 1.0);
        } else {
            controls.boost = false;
            controls.throttle = 0.0;
        }

        if offset_length < 50.0 && !self.dodging {
            let (_, aligned) = default_pd(&car, car.local(self.shot_direction), false, None);
            controls.pitch = aligned.pitch;
            controls.yaw = aligned.yaw;
            controls.roll = aligned.roll;
        }

        let runtime_viable = aerial_shot_is_viable(
            &car, self.target_location, time_remaining, gravity, boost_accel, false,
        ).valid;
        if time_remaining <= -0.4 || (self.jumped && time_remaining > 1.5 && !runtime_viable) {
            self.finished = true;
            return controls;
        }

        let side = if car.team == 0 { -1.0 } else { 1.0 };
        let ceiling_flip = self.ceiling
            && car.location.distance(self.target_location) < 92.0 + car.hitbox.length
            && !car.doublejumped
            && car.location.z < self.ball.z + 92.0
            && self.target_location.y * side > -4240.0;
        let normal_flip = !self.ceiling && !car.doublejumped && time_remaining < 0.1;
        if ceiling_flip || normal_flip {
            let local_target = car.local_location(self.target_location).flatten().normalize();
            controls.pitch = -local_target.x;
            controls.yaw = local_target.y;
            controls.jump = true;
        }

        controls
    }

}

/// Exact port of Python Aerial._party_cannon_shot_direction.
///
/// This deliberately returns the raw Party Cannon prediction. Wall correction
/// belongs to jump_shot.run after it blends this prediction with the goal
/// direction; applying wall correction here would correct the jump shot twice.
pub fn party_cannon_shot_direction(
    car: CarState,
    ball: Vector,
    ball_velocity: Vector,
    time_remaining: f32,
    goal: Vector,
    gravity: Vector,
) -> Vector {
    let safe_time = time_remaining.max(1.0 / 120.0);
    // VirxERLU Vector.cap is component-wise, not a magnitude clamp.
    // Preserve the source's cap(0, max_speed), including its zero floor.
    let car_final_velocity = ((ball - car.location) / safe_time).cap(0.0, MAX_SPEED);

    let velocity_for_power = (car_final_velocity * 6.0 + ball_velocity) / 7.0;
    let direction_to_score = (goal - ball).flatten().normalize();
    let velocity_difference = (car_final_velocity - ball_velocity).magnitude();
    let score_speed = (
        velocity_difference * shot_power_modifier(velocity_difference)
            + velocity_for_power.dot(direction_to_score)
    ).clamp(500.0, 6000.0);
    let time_to_score = ball.flat_distance(goal) / score_speed;

    let predicted_after_hit_velocity = (car_final_velocity + ball_velocity * 2.0) / 3.0;
    let predicted_ball = ball
        + predicted_after_hit_velocity * time_to_score
        + gravity * (0.5 * time_to_score * time_to_score);
    let shot_direction = (goal - predicted_ball).normalize();

    if shot_direction.magnitude() < 0.001 {
        direction_to_score
    } else {
        shot_direction
    }
}

fn shot_power_modifier(value: f32) -> f32 {
    let value = value.clamp(0.0, 4600.0);
    if value <= 500.0 {
        0.65
    } else if value <= MAX_SPEED {
        let t = (value - 500.0) / 1800.0;
        0.65 + (0.55 - 0.65) * t
    } else {
        let t = (value - MAX_SPEED) / 2300.0;
        0.55 + (0.3 - 0.55) * t
    }
}

fn contact_geometry(
    car: CarState,
    slice: BallSlice,
    goal: Vector,
    gravity: Vector,
) -> (Vector, Vector) {
    let time = slice.time.max(0.001);
    // Match Python Vector.cap(0, max_speed): component-wise clamping.
    let car_final_velocity = ((slice.location - car.location) / time).cap(0.0, MAX_SPEED);
    let ball_after_velocity = (car_final_velocity * 6.0 + slice.velocity) / 7.0;
    let direction_to_goal = (goal - slice.location).flatten().normalize();
    let velocity_difference = (car_final_velocity - slice.velocity).magnitude();
    let scoring_speed = (velocity_difference * shot_power_modifier(velocity_difference)
        + ball_after_velocity.dot(direction_to_goal))
        .clamp(500.0, 6000.0);
    let distance_to_goal = goal.flat_distance(slice.location);
    let time_to_goal = distance_to_goal / scoring_speed;
    let ball_after_velocity = (car_final_velocity + slice.velocity * 2.0) / 3.0;
    let predicted_ball = slice.location
        + ball_after_velocity * time_to_goal
        + gravity * (0.5 * time_to_goal * time_to_goal);
    let mut shot_direction = (goal - predicted_ball).normalize();
    let approach_distance = 92.75 + car.hitbox.length * 0.5;
    shot_direction = wall_shot_adjust_aerial(
        slice.location, shot_direction, approach_distance,
    ).0;
    let hitbox_offset = car.forward() * car.hitbox.offset.x
        + car.right() * car.hitbox.offset.y
        + car.up() * car.hitbox.offset.z;
    let target_location = slice.location - shot_direction * approach_distance - hitbox_offset;
    (target_location, shot_direction)
}

/// Expose Party Cannon's predicted outgoing shot and hitbox-aware
/// contact target for the grounded jump-shot routine as well as aerial planning.
pub fn party_cannon_contact_geometry(
    car: CarState,
    slice: BallSlice,
    goal: Vector,
    gravity: Vector,
) -> (Vector, Vector) {
    contact_geometry(car, slice, goal, gravity)
}

fn nearest_wall(point: Vector) -> (Vector, Vector) {
    let candidates = [
        ((4096.0 - point.x).abs(), Vector::new(4096.0, point.y, point.z), Vector::new(-1.0, 0.0, 0.0)),
        ((-4096.0 - point.x).abs(), Vector::new(-4096.0, point.y, point.z), Vector::new(1.0, 0.0, 0.0)),
        ((5120.0 - point.y).abs(), Vector::new(point.x, 5120.0, point.z), Vector::new(0.0, -1.0, 0.0)),
        ((-5120.0 - point.y).abs(), Vector::new(point.x, -5120.0, point.z), Vector::new(0.0, 1.0, 0.0)),
    ];
    let best = candidates.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
    (best.1, best.2)
}

/// Port of Party Cannon WallShot.adjust_aerial with the car's actual
/// ball-radius-plus-half-hitbox standoff.
pub fn wall_shot_adjust_aerial(
    ball: Vector,
    shot_direction: Vector,
    approach_distance: f32,
) -> (Vector, Option<Vector>, f32) {
    let target = ball - shot_direction * approach_distance;
    let (surface, normal) = nearest_wall(target);
    let target_height = (target - surface).dot(normal);
    if target_height >= 50.0 { return (shot_direction, None, target_height); }
    let tangent = shot_direction - normal * shot_direction.dot(normal);
    if tangent.magnitude() < 0.001 { return (shot_direction, None, target_height); }
    let angle = ((target_height - 50.0) / approach_distance.max(1.0))
        .clamp(-1.0, 1.0).asin();
    let adjusted = (tangent.normalize() * angle.cos() + normal * angle.sin()).normalize();
    (adjusted, Some(normal), target_height)
}

fn opponent_eta(opponent: CarState, target: Vector) -> f32 {
    if opponent.demolished {
        return f32::INFINITY;
    }
    let delta = (target - opponent.location).flatten();
    let distance = delta.magnitude();
    let forward = opponent.forward().flatten().normalize();
    let direction = delta.normalize();
    let facing = if direction == Vector::ZERO || forward == Vector::ZERO {
        0.0
    } else {
        forward.dot(direction).clamp(-1.0, 1.0)
    };
    let speed = opponent.velocity.flatten().magnitude();
    distance / (speed + 650.0).clamp(850.0, 2100.0)
        + (1.0 - facing) * 0.28
}


/// Fast-zone aerial scan from routines.fast_intercept_finder. It keeps the
/// first normal aerial if one exists; otherwise it falls back to the first
/// fast aerial whose live and predicted ball heights both clear 350 units.
pub fn find_fast_aerial_candidate(
    car: CarState,
    prediction: &BallTrajectory,
    goal: Vector,
    gravity: Vector,
    boost_accel: f32,
    earliest_time: f32,
    horizon: f32,
    live_ball_height: f32,
    max_height: f32,
    can_aerial: bool,
) -> Option<AerialCandidate> {
    if !can_aerial || car.boost <= 0.0 || prediction.slices.is_empty() {
        return None;
    }
    let mut first_fast = None;
    let mut first_normal = None;
    for slice in prediction.slices.iter().copied() {
        if slice.time < earliest_time.max(1.0 / 120.0) { continue; }
        if slice.time > horizon.min(6.0) { break; }
        if slice.location.y.abs() > 5212.75 { break; }
        // Python fast_intercept_finder has no aerial-height ceiling; its
        // only special height gate is the >350 uu fast-aerial rule below.
        let _ = max_height;
        // Python validates each prediction slice at T = intercept - now - 1/120,
        // not at the slice timestamp itself.
        let time_remaining = slice.time - 1.0 / 120.0;
        if time_remaining <= 0.0 { continue; }
        // The Python fast_intercept_finder validates this candidate with
        // CLib parse_slice_for_shot(False, False, False, True, T, ...).
        // Do not validate the later Party Cannon nose target here: that
        // geometry belongs to the Aerial routine after candidate selection.
        let parsed = parse_slice_for_shot(
            &car,
            slice.location,
            time_remaining,
            91.25,
            boost_accel,
            gravity,
            false,
            false,
            false,
            true,
            true,
        );
        if !parsed.found || parsed.shot_type != Some(ShotType::Aerial) {
            continue;
        }
        let (target_location, shot_direction) = contact_geometry(car, slice, goal, gravity);
        let candidate = AerialCandidate {
            slice,
            shot_target: goal,
            target_location,
            shot_direction,
            double_jump: parsed.fast,
            fast: parsed.fast,
            estimated_boost: 0.0,
            score: slice.time + if parsed.fast { 0.025 } else { 0.0 },
        };
        if !parsed.fast {
            if first_normal.is_none() {
                first_normal = Some(candidate);
            }
        } else if slice.location.z > 350.0
            && live_ball_height > 350.0
            && first_fast.is_none()
        {
            first_fast = Some(candidate);
        }
    }

    // Choose the earliest reachable aerial. A later normal aerial must not
    // suppress an earlier, valid fast aerial.
    match (first_normal, first_fast) {
        (Some(normal), Some(fast)) => {
            Some(if fast.slice.time < normal.slice.time { fast } else { normal })
        }
        (Some(normal), None) => Some(normal),
        (None, Some(fast)) => Some(fast),
        (None, None) => None,
    }
}

pub fn find_aerial_candidate(
    car: CarState,
    prediction: &BallTrajectory,
    goal: Vector,
    targets: ShotTargets,
    cars: &[CarState],
    gravity: Vector,
    boost_accel: f32,
    game_time: f32,
    kickoff_contest_until: f32,
    min_height: f32,
    max_height: f32,
    horizon: f32,
    unlimited_boost: bool,
) -> Option<AerialCandidate> {
    if prediction.slices.is_empty() || (!unlimited_boost && car.boost <= 0.0) {
        return None;
    }

    let mut best: Option<AerialCandidate> = None;
    let mut last_sample = -999.0_f32;
    for slice in prediction.slices.iter().copied() {
        let time = slice.time;
        if time < 0.10 {
            continue;
        }
        if time > horizon.min(6.0) {
            break;
        }
        let sample_interval = if time < 1.5 { 0.05 } else { 0.10 };
        if time - last_sample + 1.0e-5 < sample_interval {
            continue;
        }
        last_sample = time;

        if slice.location.y.abs() > 5212.75 {
            break;
        }
        if !(min_height < slice.location.z && slice.location.z < max_height) {
            continue;
        }

        let time_remaining = time - 1.0 / 120.0;
        if time_remaining <= 0.0 {
            continue;
        }

        // Python calls parse_slice_for_shot_with_target with only the aerial
        // flag enabled. Keep that C-library reachability test separate from
        // Party Cannon's later hitbox-aware execution target.
        let mut parsed = parse_slice_for_shot_with_target(
            &car,
            slice.location,
            slice.velocity,
            time_remaining,
            92.75,
            boost_accel,
            gravity,
            targets,
            false,
            false,
            false,
            true,
            true,
            false,
        );
        if !parsed.found || parsed.shot_type != Some(ShotType::Aerial) {
            continue;
        }

        // Exact Volt post-launch reserve for aerials from
        // _shot_post_launch_boost_reserve(). Re-run the same parser with the
        // reduced approach budget, rather than estimating a separate race ETA.
        if !unlimited_boost {
            let height = (slice.location.z - 92.75 - car.location.z).max(0.0);
            let distance_3d = car.location.distance(slice.location);
            let distance_factor = ((distance_3d - 250.0) / 1000.0).clamp(0.0, 1.0);
            let height_factor = (height / 700.0).clamp(0.0, 1.0);
            let reserve = (car.boost * 0.18).clamp(2.0, 15.0)
                * distance_factor
                * height_factor;
            if reserve > 0.25 {
                let approach_budget = car.boost - reserve;
                if approach_budget < -0.01 {
                    continue;
                }
                let mut budgeted_car = car;
                budgeted_car.boost = approach_budget.max(0.0).trunc();
                let next = parse_slice_for_shot_with_target(
                    &budgeted_car,
                    slice.location,
                    slice.velocity,
                    time_remaining,
                    92.75,
                    boost_accel,
                    gravity,
                    targets,
                    false,
                    false,
                    false,
                    true,
                    true,
                    false,
                );
                if !next.found || next.shot_type != Some(ShotType::Aerial) {
                    continue;
                }
                parsed = next;
            }
        }

        let (target_location, shot_direction) = contact_geometry(car, slice, goal, gravity);
        let score = score_aerial_candidate(
            car,
            cars,
            slice.location,
            slice.velocity,
            time_remaining,
            targets,
            gravity.z,
            game_time < kickoff_contest_until,
        );
        let estimate = aerial_shot_is_viable(
            &car, target_location, time_remaining, gravity, boost_accel, true,
        );
        let candidate = AerialCandidate {
            slice,
            shot_target: goal,
            target_location,
            shot_direction,
            double_jump: parsed.fast,
            fast: parsed.fast,
            estimated_boost: estimate.validation.boost_estimate as f32,
            score,
        };
        if best.is_none_or(|current| candidate.score < current.score) {
            best = Some(candidate);
        }
    }
    best
}

/// Exact Party Cannon WallShot.adjust_jump correction used by jump_shot.
/// Only the four vertical side walls participate; floor and ceiling contacts
/// are handled by the normal jump/aerial contact model.
pub fn wall_shot_adjust_jump(ball: Vector, shot_direction: Vector) -> (Vector, Option<Vector>, f32) {
    let target = ball - shot_direction * 165.0;
    let walls = [
        ((4096.0 - target.x).abs(), Vector::new(4096.0, target.y, target.z), Vector::new(-1.0, 0.0, 0.0)),
        ((-4096.0 - target.x).abs(), Vector::new(-4096.0, target.y, target.z), Vector::new(1.0, 0.0, 0.0)),
        ((5120.0 - target.y).abs(), Vector::new(target.x, 5120.0, target.z), Vector::new(0.0, -1.0, 0.0)),
        ((-5120.0 - target.y).abs(), Vector::new(target.x, -5120.0, target.z), Vector::new(0.0, 1.0, 0.0)),
    ];
    let (_, _, normal) = walls.into_iter().min_by(|a,b| a.0.total_cmp(&b.0)).unwrap();
    let plane_point = match (normal.x, normal.y) {
        (-1.0, 0.0) => Vector::new(4096.0, ball.y, ball.z),
        (1.0, 0.0) => Vector::new(-4096.0, ball.y, ball.z),
        (0.0, -1.0) => Vector::new(ball.x, 5120.0, ball.z),
        _ => Vector::new(ball.x, -5120.0, ball.z),
    };
    let target_surface = match (normal.x, normal.y) {
        (-1.0, 0.0) => Vector::new(4096.0, target.y, target.z),
        (1.0, 0.0) => Vector::new(-4096.0, target.y, target.z),
        (0.0, -1.0) => Vector::new(target.x, 5120.0, target.z),
        _ => Vector::new(target.x, -5120.0, target.z),
    };
    let target_height = (target - target_surface).dot(normal);
    if target_height >= 40.0 {
        return (shot_direction, None, target_height);
    }
    let ball_height = (ball - plane_point).dot(normal);
    let tangent = shot_direction - normal * shot_direction.dot(normal);
    if tangent.magnitude() < 0.001 {
        return (shot_direction, None, target_height);
    }
    let angle = ((ball_height - 40.0) / 165.0).clamp(-1.0, 1.0).asin();
    let adjusted = (tangent.normalize() * angle.cos() + normal * angle.sin()).normalize();
    (adjusted, Some(normal), target_height)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math::Matrix3, state::Hitbox};
    use rlbot::flat::AirState;

    fn car(location: Vector, boost: f32) -> CarState {
        CarState {
            index: 0,
            team: 0,
            location,
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
            boost,
            hitbox: Hitbox { length: 118.0, width: 84.0, height: 36.0, offset: Vector::ZERO },
        }
    }

    #[test]
    fn party_cannon_wall_jump_adjusts_only_when_approach_crosses_wall_clearance() {
        let far = wall_shot_adjust_jump(
            Vector::new(0.0, 0.0, 500.0),
            Vector::new(0.0, 1.0, 0.0),
        );
        assert!(far.1.is_none());

        let near = wall_shot_adjust_jump(
            Vector::new(4000.0, 0.0, 500.0),
            Vector::new(-0.8, 0.6, 0.0),
        );
        assert_eq!(near.1, Some(Vector::new(-1.0, 0.0, 0.0)));
        assert!(near.0.x < -0.1);
        assert!(near.2 < 40.0);
    }

    #[test]
    fn party_cannon_aerial_contact_uses_actual_hitbox_standoff_and_offset() {
        let car = car(Vector::ZERO, 50.0);
        let slice = BallSlice {
            time: 1.0,
            location: Vector::new(0.0, 1000.0, 500.0),
            velocity: Vector::ZERO,
        };
        let (target, direction) = party_cannon_contact_geometry(
            car, slice, Vector::new(0.0, 5120.0, 300.0), Vector::new(0.0, 0.0, -650.0),
        );
        assert!((target.distance(slice.location) - (92.75 + car.hitbox.length * 0.5)).abs() < 0.01);
        assert!(direction.magnitude() > 0.99);
    }

    #[test]
    fn jump_shot_party_cannon_direction_is_raw_before_jump_wall_correction() {
        let mut me = car(Vector::new(3500.0, -1000.0, 17.0), 50.0);
        me.velocity = Vector::new(0.0, 1500.0, 0.0);
        let ball = Vector::new(4000.0, 0.0, 700.0);
        let goal = Vector::new(0.0, 5120.0, 300.0);
        let gravity = Vector::new(0.0, 0.0, -650.0);

        let raw = party_cannon_shot_direction(
            me, ball, Vector::ZERO, 1.0, goal, gravity,
        );
        let slice = BallSlice { time: 1.0, location: ball, velocity: Vector::ZERO };
        let (_, aerial_corrected) = party_cannon_contact_geometry(me, slice, goal, gravity);

        assert!(raw.magnitude() > 0.99);
        assert!(aerial_corrected.magnitude() > 0.99);
        assert!(
            raw.distance(aerial_corrected) > 0.001,
            "jump-shot direction must not inherit the aerial wall correction"
        );
    }

    #[test]
    fn party_cannon_wall_aerial_correction_uses_vertical_wall_clearance() {
        let ball = Vector::new(4050.0, 0.0, 700.0);
        let direction = Vector::new(0.0, 1.0, 0.0);
        let (adjusted, normal, clearance) = wall_shot_adjust_aerial(ball, direction, 151.75);
        assert!(normal.is_some());
        assert!(adjusted.magnitude() > 0.99);
        assert!(clearance < 50.0);
    }

    #[test]
    fn party_cannon_contact_target_uses_hitbox_aware_standoff() {
        let slice = BallSlice {
            time: 1.5,
            location: Vector::new(0.0, 0.0, 900.0),
            velocity: Vector::ZERO,
        };
        let car_state = car(Vector::new(0.0, -1500.0, 17.0), 100.0);
        let (target, direction) = contact_geometry(
            car_state,
            slice,
            Vector::new(0.0, 5120.0, 300.0),
            Vector::new(0.0, 0.0, -650.0),
        );
        let lower_middle = slice.location - Vector::new(0.0, 0.0, AERIAL_CONTACT_DROP);
        assert!((target.distance(lower_middle) - (92.75 + car_state.hitbox.length * 0.5)).abs() < 0.01);
        assert!(direction.magnitude() > 0.99);
    }

    #[test]
    fn aerial_candidate_requires_a_valid_boost_budget() {
        let mut car = car(Vector::new(0.0, -1200.0, 17.0), 100.0);
        // Face downfield (+Y in this test setup), matching the approach lane.
        car.orientation = Matrix3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0);
        let prediction = BallTrajectory {
            slices: vec![
                BallSlice {
                    time: 1.5,
                    location: Vector::new(0.0, 0.0, 900.0),
                    velocity: Vector::ZERO,
                },
                BallSlice {
                    time: 2.0,
                    location: Vector::new(500.0, 0.0, 1100.0),
                    velocity: Vector::ZERO,
                },
                BallSlice {
                    time: 3.0,
                    location: Vector::new(0.0, 0.0, 900.0),
                    velocity: Vector::ZERO,
                },
            ],
        };
        let result = find_aerial_candidate(
            car,
            &prediction,
            Vector::new(0.0, 5120.0, 300.0),
            ShotTargets {
                left: Vector::new(-893.0, 5120.0, 300.0),
                right: Vector::new(893.0, 5120.0, 300.0),
            },
            &[],
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            1.0,
            -999.0,
            350.0,
            f32::INFINITY,
            6.0,
            false,
        );
        assert!(result.is_some());
        assert!(result.unwrap().estimated_boost >= 0.0);
    }

    #[test]
    fn fast_intercept_aerial_uses_clib_slice_time_and_skips_expired_sample() {
        let mut me = car(Vector::ZERO, 100.0);
        me.orientation = Matrix3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0);
        let prediction = BallTrajectory {
            slices: vec![
                BallSlice {
                    time: 1.0 / 120.0,
                    location: Vector::new(0.0, 0.0, 900.0),
                    velocity: Vector::ZERO,
                },
                BallSlice {
                    time: 1.8,
                    location: Vector::new(0.0, 0.0, 900.0),
                    velocity: Vector::ZERO,
                },
            ],
        };
        let candidate = find_fast_aerial_candidate(
            me,
            &prediction,
            Vector::new(0.0, 5120.0, 300.0),
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            0.0,
            6.0,
            900.0,
            f32::INFINITY,
            true,
        );
        assert!(candidate.is_some());
        assert!((candidate.unwrap().slice.time - 1.8).abs() < 1.0e-5);
    }

    #[test]
    fn committed_aerial_refreshes_to_the_nearest_live_prediction_slice() {
        let car = car(Vector::ZERO, 100.0);
        let candidate = AerialCandidate {
            slice: BallSlice {
                time: 2.0,
                location: Vector::new(0.0, 0.0, 900.0),
                velocity: Vector::ZERO,
            },
            shot_target: Vector::new(0.0, 5120.0, 300.0),
            target_location: Vector::new(0.0, 0.0, 745.0),
            shot_direction: Vector::new(0.0, 1.0, 0.0),
            double_jump: false,
            fast: false,
            estimated_boost: 20.0,
            score: 2.0,
        };
        let mut shot = AerialShot::new(
            car, candidate, 1.0, Vector::new(0.0, 0.0, -650.0), 991.6667,
        );
        let prediction = BallTrajectory {
            slices: vec![
                BallSlice {
                    time: 1.0,
                    location: Vector::new(0.0, 0.0, 800.0),
                    velocity: Vector::ZERO,
                },
                BallSlice {
                    time: 1.5,
                    location: Vector::new(200.0, 100.0, 950.0),
                    velocity: Vector::new(0.0, 100.0, 0.0),
                },
                BallSlice {
                    time: 2.0,
                    location: Vector::new(500.0, 0.0, 1100.0),
                    velocity: Vector::ZERO,
                },
            ],
        };
        shot.refresh_prediction(
            car, &prediction, 1.5, Vector::new(0.0, 5120.0, 300.0),
            Vector::new(0.0, 0.0, -650.0),
            950.0,
);
        assert_eq!(shot.ball, Vector::new(200.0, 100.0, 950.0));
        assert_eq!(shot.slice_velocity, Vector::new(0.0, 100.0, 0.0));
        assert_eq!(shot.slice_time, 1.5);
    }

    #[test]
    fn aerial_refresh_detects_a_real_backboard_rebound() {
        let mut me = car(Vector::new(0.0, 0.0, 700.0), 80.0);
        me.airborne = true;
        me.jumped = true;
        me.air_state = AirState::InAir;
        let candidate = AerialCandidate {
            slice: BallSlice {
                time: 1.0,
                location: Vector::new(0.0, 4200.0, 600.0),
                velocity: Vector::new(0.0, -500.0, 0.0),
            },
            shot_target: Vector::new(0.0, 5120.0, 300.0),
            target_location: Vector::new(0.0, 4000.0, 500.0),
            shot_direction: Vector::new(0.0, 1.0, 0.0),
            double_jump: false,
            fast: false,
            estimated_boost: 20.0,
            score: 1.0,
        };
        let mut shot = AerialShot::new(
            me, candidate, 1.0, Vector::new(0.0, 0.0, -650.0), 991.6667,
        );
        let prediction = BallTrajectory { slices: vec![candidate.slice] };
        shot.refresh_prediction(
            me, &prediction, 1.0, Vector::new(0.0, 5120.0, 300.0),
            Vector::new(0.0, 0.0, -650.0),
            900.0,
);
        assert!(shot.backboard_mode);
    }

    #[test]
    fn aerial_refresh_uses_outward_clear_direction_near_own_goal() {
        let me = car(Vector::new(0.0, -2500.0, 500.0), 80.0);
        let candidate = AerialCandidate {
            slice: BallSlice {
                time: 1.0,
                location: Vector::new(1000.0, -3500.0, 500.0),
                velocity: Vector::ZERO,
            },
            shot_target: Vector::new(0.0, 5120.0, 300.0),
            target_location: Vector::new(0.0, -3300.0, 400.0),
            shot_direction: Vector::new(0.0, 1.0, 0.0),
            double_jump: false,
            fast: false,
            estimated_boost: 20.0,
            score: 1.0,
        };
        let mut shot = AerialShot::new(
            me, candidate, 1.0, Vector::new(0.0, 0.0, -650.0), 991.6667,
        );
        let prediction = BallTrajectory { slices: vec![candidate.slice] };
        shot.refresh_prediction(
            me, &prediction, 1.0, Vector::new(0.0, 5120.0, 300.0),
            Vector::new(0.0, 0.0, -650.0),
            600.0,
);
        assert!(shot.shot_direction.x > 0.4);
        assert!(shot.shot_direction.y > 0.4);
    }

    #[test]
    fn party_cannon_shot_power_uses_the_2300_speed_band() {
        assert!((shot_power_modifier(3450.0) - 0.425).abs() < 1e-6);
        assert!((shot_power_modifier(2300.0) - 0.55).abs() < 1e-6);
    }

    #[test]
    fn fast_aerial_releases_three_frames_then_holds_second_jump_three_frames() {
        let me = car(Vector::ZERO, 100.0);
        let candidate = AerialCandidate {
            slice: BallSlice {
                time: 3.0,
                location: Vector::new(0.0, 0.0, 900.0),
                velocity: Vector::ZERO,
            },
            shot_target: Vector::new(0.0, 5120.0, 300.0),
            target_location: Vector::new(0.0, 0.0, 745.0),
            shot_direction: Vector::new(0.0, 1.0, 0.0),
            double_jump: true,
            fast: true,
            estimated_boost: 20.0,
            score: 1.0,
        };
        let mut shot = AerialShot::new(
            me, candidate, 1.0, Vector::new(0.0, 0.0, -650.0), 991.6667,
        );

        assert!(shot.run(me, 1.0, Vector::new(0.0, 0.0, -650.0), 991.6667, 0.0).jump);
        for frame in 1..=3 {
            let controls = shot.run(
                me, 1.20 + frame as f32 / 120.0,
                Vector::new(0.0, 0.0, -650.0), 991.6667, 0.0,
            );
            assert!(!controls.jump, "release frame {frame} must be unpressed");
        }
        for frame in 1..=3 {
            let controls = shot.run(
                me, 1.23 + frame as f32 / 120.0,
                Vector::new(0.0, 0.0, -650.0), 991.6667, 0.0,
            );
            assert!(controls.jump, "second-jump hold frame {frame} must be pressed");
            assert!(shot.dodging);
        }
        let finished = shot.run(
            me, 1.27, Vector::new(0.0, 0.0, -650.0), 991.6667, 0.0,
        );
        assert!(!finished.jump);
        assert!(!shot.dodging);
        assert!(shot.jumped);
    }

    #[test]
    fn aerial_throttle_uses_python_fixed_dt_and_separate_boost_window() {
        let mut me = car(Vector::ZERO, 100.0);
        me.airborne = true;
        me.air_state = AirState::InAir;
        let candidate = AerialCandidate {
            slice: BallSlice {
                time: 2.0,
                location: Vector::new(156.75, 0.0, 0.0),
                velocity: Vector::ZERO,
            },
            shot_target: Vector::new(0.0, 5120.0, 300.0),
            target_location: Vector::new(1.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            double_jump: false,
            fast: false,
            estimated_boost: 10.0,
            score: 1.0,
        };
        let mut shot = AerialShot::new(
            me, candidate, 1.0, Vector::new(0.0, 0.0, -650.0), 991.6667,
        );
        let controls = shot.run(
            me, 1.0, Vector::new(0.0, 0.0, -650.0), 991.6667, 0.0,
        );
        assert!((controls.throttle - 0.9).abs() < 0.02);
    }

    #[test]
    fn aerial_state_emits_jump_and_boost_controls() {
        let car = car(Vector::ZERO, 100.0);
        let candidate = AerialCandidate {
            slice: BallSlice {
                time: 2.0,
                location: Vector::new(0.0, 0.0, 900.0),
                velocity: Vector::ZERO,
            },
            shot_target: Vector::new(0.0, 5120.0, 300.0),
            target_location: Vector::new(0.0, 0.0, 745.0),
            shot_direction: Vector::new(0.0, 0.0, 1.0),
            double_jump: false,
            fast: false,
            estimated_boost: 20.0,
            score: 1.5,
        };
        let mut shot = AerialShot::new(car, candidate, 1.0, Vector::new(0.0, 0.0, -650.0), 991.6667);
        let controls = shot.run(
            car,
            1.05,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
            0.0,
        );
        assert!(controls.jump);
    }
}
