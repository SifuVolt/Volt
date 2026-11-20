//! Active Volt decision and shot-execution layer.
//!
//! Kickoff assignment, team roles, defensive priorities, prediction-based
//! ground/jump contacts, finite-boost reachability, contact geometry and
//! timed shot execution all feed the live RLBot controller.

use crate::{
    control::{default_drive, default_pd, DriveInput},
    math::{utils::cap_in_field, Vector},
    physics::drive::{curvature, dodge_impulse, throttle_acceleration},
    prediction::BallTrajectory,
    shot_physics::get_jump_time,
    state::{BallState, BoostPadState, CarState, MatchState},
};
use rlbot::flat::ControllerState;

const BALL_RADIUS: f32 = 92.75;
const MAX_SPEED: f32 = 2300.0;
const BOOST_ACCEL: f32 = 991.6667;
const JUMP_SPEED: f32 = 291.6667;
const JUMP_ACCEL: f32 = 1458.3333;
const JUMP_MAX_DURATION: f32 = 0.20;
const JUMP_GRAVITY: f32 = -650.0;
const MIN_AIR_BOOST_TIME: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Challenger,
    Goalie,
    Transition,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeamRoles {
    pub challenger: usize,
    pub goalie: usize,
    pub transition: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Decision {
    KickoffKicker,
    KickoffCheat,
    KickoffBackBoost,
    EmergencyChallenge,
    GroundChallenge,
    Goalkeeper,
    Transition,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GroundIntercept {
    time: f32,
    ball: Vector,
    ball_velocity: Vector,
    approach: Vector,
    shot_direction: Vector,
    contact_quality: f32,
    contact_point: Vector,
    contact_face: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Plan {
    decision: Decision,
    target: Vector,
    target_speed: f32,
    shot: Option<GroundIntercept>,
    jump: Option<GroundIntercept>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FlipState {
    started_at: f32,
    direction: Vector,
    cancel: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct JumpShotState {
    started_at: f32,
    intercept_time: f32,
    ball: Vector,
    approach: Vector,
    shot_direction: Vector,
    quick: bool,
    dodge_started_at: Option<f32>,
    dodge_direction: Vector,
}

#[derive(Debug, Default)]
pub struct BotBrain {
    kickoff_complete: bool,
    pub match_state: MatchState,
    pub boost_accel: f32,
    pub boost_pads: Vec<BoostPadState>,
    last_kickoff_time: f32,
    ground_flip: Option<FlipState>,
    jump_shot: Option<JumpShotState>,
}

impl BotBrain {
    pub fn new() -> Self {
        Self {
            kickoff_complete: false,
            match_state: MatchState::default(),
            boost_accel: BOOST_ACCEL,
            boost_pads: Vec::new(),
            last_kickoff_time: -999.0,
            ground_flip: None,
            jump_shot: None,
        }
    }

    pub fn new_with_boost_accel(boost_accel: f32) -> Self {
        let mut brain = Self::new();
        brain.boost_accel = boost_accel;
        brain
    }

    /// Produce the active controller for this tick. The car list must contain
    /// all players from the current packet, including this bot.
    pub fn tick(
        &mut self,
        me: CarState,
        cars: &[CarState],
        ball: BallState,
        prediction: &BallTrajectory,
        game_time: f32,
        match_state: MatchState,
        boost_pads: &[BoostPadState],
    ) -> ControllerState {
        self.match_state = match_state;
        self.boost_pads.clear();
        self.boost_pads.extend_from_slice(boost_pads);
        let plan = self.choose_plan(me, cars, ball, prediction, game_time, match_state);

        // Keep a committed jump shot, but refresh its moving contact while the
        // same predicted opportunity remains nearby. This prevents stale
        // ball/approach coordinates without allowing every planner tick to
        // replace the mechanic with a different shot.
        if let (Some(mut state), Some(candidate)) = (self.jump_shot, plan.jump) {
            if state.dodge_started_at.is_none() {
                let candidate_time = game_time + candidate.time;
                if (candidate_time - state.intercept_time).abs() <= 0.28
                    && candidate.ball.distance(state.ball) <= 450.0
                {
                    state.intercept_time = candidate_time;
                    state.ball = candidate.ball;
                    state.approach = candidate.approach;
                    state.shot_direction = candidate.shot_direction;
                    self.jump_shot = Some(state);
                }
            }
        }

        self.drive_to(me, plan, game_time)
    }

    fn choose_plan(
        &mut self,
        me: CarState,
        cars: &[CarState],
        ball: BallState,
        prediction: &BallTrajectory,
        game_time: f32,
        match_state: MatchState,
    ) -> Plan {
        let side = team_side(me.team);
        let opponents: Vec<CarState> = cars
            .iter()
            .copied()
            .filter(|car| car.team != me.team && !car.demolished)
            .collect();
        // Use RLBot's match phase, not ball-position heuristics: kickoff
        // remains authoritative during kickoff countdown/settling frames.
        let kickoff_active = match_state.round_active && match_state.kickoff;

        if kickoff_active && game_time - self.last_kickoff_time > 3.0 {
            self.kickoff_complete = false;
        } else if !kickoff_active {
            if !self.kickoff_complete {
                self.last_kickoff_time = game_time;
            }
            self.kickoff_complete = true;
        }

        if kickoff_active && !self.kickoff_complete {
            let mut teammates: Vec<CarState> = cars
                .iter()
                .copied()
                .filter(|car| car.team == me.team && !car.demolished)
                .collect();
            teammates.sort_by(|a, b| {
                a.location
                    .distance(ball.location)
                    .total_cmp(&b.location.distance(ball.location))
                    .then_with(|| a.index.cmp(&b.index))
            });

            let kicker = teammates.first().map(|car| car.index);
            let cheater = teammates.get(1).map(|car| car.index);

            if kicker == Some(me.index) {
                return Plan {
                    decision: Decision::KickoffKicker,
                    target: ball.location,
                    target_speed: MAX_SPEED,
                    shot: None,
                jump: None,
                };
            }

            if teammates.len() <= 2 || cheater == Some(me.index) {
                return Plan {
                    decision: Decision::KickoffCheat,
                    target: Vector::new(0.0, -side * 1650.0, 0.0),
                    target_speed: 1200.0,
                    shot: None,
                jump: None,
                };
            }

            let x_sign = if me.location.x == 0.0 {
                1.0
            } else {
                me.location.x.signum()
            };
            return Plan {
                decision: Decision::KickoffBackBoost,
                target: Vector::new(x_sign * 3000.0, side * 4200.0, 0.0),
                target_speed: MAX_SPEED,
                shot: None,
                jump: None,
            };
        }

        let living_teammates: Vec<CarState> = cars
            .iter()
            .copied()
            .filter(|car| car.team == me.team && !car.demolished)
            .collect();
        let roles = assign_roles(&living_teammates, ball, side);

        let ball_depth = ball.location.y * side;
        let urgent_save = (ball_depth > 4200.0 && ball.location.z < 350.0)
            || (ball_depth > 1200.0
                && ball.location.z < 350.0
                && ball.velocity.y * side > 250.0);

        let is_challenger = roles
            .map(|r| r.challenger == me.index)
            .unwrap_or(true);
        let is_goalie = roles.map(|r| r.goalie == me.index).unwrap_or(false);

        if urgent_save && is_challenger {
            return Plan {
                decision: Decision::EmergencyChallenge,
                target: select_contact_approach(me, ball.location, shot_aim_point(ball.location, side)),
                target_speed: MAX_SPEED,
                shot: None,
                jump: None,
            };
        }

        if is_goalie && !is_challenger {
            return Plan {
                decision: Decision::Goalkeeper,
                target: Vector::new(0.0, side * 4970.0, 0.0),
                target_speed: 1500.0,
                shot: None,
                jump: None,
            };
        }

        if roles
            .and_then(|r| r.transition)
            .is_some_and(|index| index == me.index)
        {
            let x_sign = if me.location.x == 0.0 {
                1.0
            } else {
                me.location.x.signum()
            };
            return Plan {
                decision: Decision::Transition,
                target: Vector::new(x_sign * 900.0, side * 5050.0, 0.0),
                target_speed: 1700.0,
                shot: None,
                jump: None,
            };
        }

        if is_challenger {
            let aim = shot_aim_point(ball.location, side);
            let intercept = choose_ground_intercept(me, ball, prediction, aim, &opponents);
            let jump = if urgent_save { None } else { choose_jump_intercept(me, prediction, aim, &opponents) };
            let use_jump = jump.filter(|candidate| {
                intercept.is_none_or(|ground| candidate.time < ground.time)
            });
            let selected = use_jump.or(intercept);
            let target = selected
                .map(|shot| shot.approach)
                .unwrap_or_else(|| select_contact_approach(me, ball.location, aim));
            return Plan {
                decision: if urgent_save {
                    Decision::EmergencyChallenge
                } else {
                    Decision::GroundChallenge
                },
                target,
                target_speed: MAX_SPEED,
                shot: if use_jump.is_none() && !urgent_save { intercept } else { None },
                jump: use_jump,
            };
        }

        // In 1v1 there is no separate goalie: the sole player challenges.
        let aim = shot_aim_point(ball.location, side);
        let intercept = choose_ground_intercept(me, ball, prediction, aim, &opponents);
        let jump = choose_jump_intercept(me, prediction, aim, &opponents);
        let use_jump = jump.filter(|candidate| {
            intercept.is_none_or(|ground| candidate.time < ground.time)
        });
        let selected = use_jump.or(intercept);
        Plan {
            decision: Decision::GroundChallenge,
            target: selected
                .map(|shot| shot.approach)
                .unwrap_or_else(|| select_contact_approach(me, ball.location, aim)),
            target_speed: MAX_SPEED,
            shot: if use_jump.is_none() { intercept } else { None },
            jump: use_jump,
        }
    }

    fn drive_to(&mut self, car: CarState, plan: Plan, game_time: f32) -> ControllerState {
        if let Some(state) = self.jump_shot {
            return self.run_jump_shot(car, state, game_time);
        }

        if let Some(jump) = plan.jump {
            let t = jump.time;
            let jump_time = get_jump_time(
                jump.ball.z - car.location.z,
                car.velocity.z,
                -650.0,
            ).time as f32;
            let local_ball = car.local_location(jump.ball);
            let local_approach = car.local_location(jump.approach);
            let distance = jump.approach.flat_distance(car.location);
            let speed = car.velocity.flatten().magnitude().max(850.0);
            let approach_direction = local_approach.flatten().normalize();
            let eta = distance / (speed + if car.boost > 0.0 { 300.0 } else { 0.0 })
                + (1.0 - approach_direction.x.clamp(-1.0, 1.0)) * 0.20;
            let aligned_for_launch = local_approach.x > 0.0
                && local_approach.x < 450.0
                && local_approach.y.abs() < 92.75;
            let travel_time = (t - jump_time).max(0.10);
            let required_speed = (distance / travel_time).clamp(0.0, MAX_SPEED);

            if !car.airborne
                && t <= jump_time * 1.08
                && local_ball.x > 0.0
                && aligned_for_launch
                && eta <= t + 0.10
            {
                let shot_direction = jump.shot_direction;
                let state = JumpShotState {
                    started_at: game_time,
                    intercept_time: game_time + t,
                    ball: jump.ball,
                    approach: jump.approach,
                    shot_direction,
                    quick: t <= 0.45,
                    dodge_started_at: None,
                    dodge_direction: Vector::new(1.0, 0.0, 0.0),
                };
                self.jump_shot = Some(state);
                return self.run_jump_shot(car, state, game_time);
            }

            let output = controlled_drive(
                &car,
                jump.approach,
                required_speed.max(900.0),
            );
            return controller_from_drive(output, car);
        }

        if let Some(shot) = plan.shot {
            if let Some(flip) = self.ground_flip {
                return self.run_ground_flip(game_time, flip);
            }
            let t = shot.time;
            let local_offset = car.local_location(shot.approach.flatten());
            let distance_remaining = (shot.approach - car.location).flat_distance(Vector::ZERO)
                - car.hitbox.length / 3.0;
            let drive_target = shot_approach_target(car, shot.ball, shot.approach, shot.shot_direction, t);
            let drive_distance_remaining = (drive_target - car.location).flat_distance(Vector::ZERO)
                - car.hitbox.length / 3.0;
            let approach_direction = (shot.approach - car.location).flatten().normalize();
            let target_speed_along_approach = shot.ball_velocity.flatten().dot(approach_direction);
            let closing_speed = car.velocity.flatten().dot(approach_direction)
                - target_speed_along_approach;
            let time_distance_speed = drive_distance_remaining.max(0.0) / t.max(0.001);
            // Closed-loop speed demand: compensate for current closing speed,
            // but avoid asking for maximum speed when the intercept is already
            // comfortably reachable. default_throttle applies the curvature cap.
            let contact_power_adjustment = (0.72 - shot.contact_quality).max(0.0) * 350.0;
            let ball_progress = shot.ball_velocity.flatten().dot(shot.shot_direction);
            let progress_adjustment = (ball_progress - 800.0).clamp(0.0, 1000.0) * 0.12;
            let mut required_speed = (time_distance_speed
                + (time_distance_speed - closing_speed) * 0.55
                + contact_power_adjustment
                - progress_adjustment)
                .clamp(450.0, MAX_SPEED);
            if t < 0.28 {
                required_speed = required_speed.max(1900.0);
            }
            let local_ball = car.local_location(shot.ball);
            let local_contact_point = car.local_location(shot.contact_point);
            let local_future_velocity = car.local(car.velocity + Vector::new(0.0, 0.0, -650.0) * t).flatten();
            let local_shot_direction = car.local(shot.shot_direction).flatten().normalize();
            let angle = local_offset.y.atan2(local_offset.x).abs();

            let output = controlled_drive(&car, drive_target, required_speed);
            let velocity = output.car_speed;
            let impulse = dodge_impulse(car.velocity.flatten(), MAX_SPEED);
            let contact_surface_aligned = local_contact_point.x > 0.0
                && local_contact_point.x < car.hitbox.length * 0.5 + BALL_RADIUS + 50.0
                && local_contact_point.y.abs() < car.hitbox.width * 0.5 + BALL_RADIUS;
            let aligned_contact = (velocity.abs() < 100.0
                && distance_remaining < car.hitbox.length / 2.0)
                || (contact_surface_aligned
                    && local_future_velocity.x >= local_offset.x - car.hitbox.length / 3.0
                    && local_offset.x > 0.0);

            let flip_speed_limit = if car.boost > 12.0 {
                required_speed.abs()
            } else {
                1900.0
            };

            if !car.airborne
                && (0.1..0.35).contains(&t)
                && (velocity.abs() + impulse).min(MAX_SPEED) <= flip_speed_limit
                && aligned_contact
                && local_ball.x > 0.0
            {
                self.ground_flip = Some(FlipState {
                    started_at: game_time,
                    direction: ground_dodge_direction(
                        local_ball.flatten(),
                        local_shot_direction,
                        shot.contact_quality,
                        shot.contact_face,
                    ),
                    cancel: angle > 2.0,
                });
                return self.run_ground_flip(game_time, self.ground_flip.unwrap());
            }

            let controller = controller_from_drive(output, car);
            if car.airborne {
                self.ground_flip = None;
            }
            return controller;
        }

        self.ground_flip = None;
        let target = plan.target.flatten();
        let local_target = car.local_location(target);
        let distance = local_target.flatten().magnitude();
        let angle = local_target.y.atan2(local_target.x).abs();

        let mut target_speed = plan.target_speed;
        if distance < 700.0 {
            target_speed = target_speed.min(1200.0);
        } else if angle > 1.7 {
            target_speed = target_speed.min(1050.0);
        } else if angle > 1.0 {
            target_speed = target_speed.min(1650.0);
        }

        let output = controlled_drive(&car, target, target_speed);

        let controller = controller_from_drive(output, car);

        // The selected decision is intentionally retained in Plan for
        // deterministic testing and future routine-specific execution.
        let _ = plan.decision;
        controller
    }

    fn run_jump_shot(
        &mut self,
        car: CarState,
        mut state: JumpShotState,
        game_time: f32,
    ) -> ControllerState {
        let elapsed = (game_time - state.started_at).max(0.0);
        let time_remaining = state.intercept_time - game_time;
        let local_ball = car.local_location(state.ball).flatten();
        let distance_to_contact = car.location.distance(state.approach);
        let mut controller = ControllerState::default();

        if time_remaining < -0.10
            || (elapsed > 0.30 && !car.airborne && state.dodge_started_at.is_none())
        {
            self.jump_shot = None;
            return controller;
        }

        // Keep the nose aimed along the planned outgoing shot vector. The
        // normal shot posture deliberately pitches back (S) while the ball is
        // above the car; quick shots skip that posture and go straight to the
        // contact/dodge sequence.
        let local_shot_direction = car
            .local(state.shot_direction.flatten())
            .flatten()
            .normalize();
        let (_, pd) = default_pd(&car, local_shot_direction, false, None);
        controller.pitch = pd.pitch;
        controller.yaw = pd.yaw;
        controller.roll = pd.roll;

        let hold_s = !state.quick
            && state.ball.z > car.location.z + 65.0
            && state.dodge_started_at.is_none();
        if hold_s {
            controller.pitch = 1.0;
            // Pair A/D yaw correction with a small opposite air-roll input,
            // matching Volt's Python jump-shot controller.
            controller.roll = if pd.yaw.abs() > 0.08 {
                -0.20 * pd.yaw.signum()
            } else {
                0.0
            };
        }

        if let Some(dodge_time) = state.dodge_started_at {
            let dodge_elapsed = game_time - dodge_time;
            controller.jump = dodge_elapsed <= 0.045;
            controller.pitch = -state.dodge_direction.x;
            controller.yaw = state.dodge_direction.y;
            controller.roll = 0.0;
            if dodge_elapsed > 0.12 || car.has_dodged {
                self.jump_shot = None;
            } else {
                self.jump_shot = Some(state);
            }
            return controller;
        }

        // Predict the remaining jump arc and use the remaining approach error
        // to add throttle/boost along the car's forward axis. This is the
        // important part missing from the previous "jump, wait, dodge" logic:
        // the car actively corrects its flight instead of merely pointing.
        let tau = (JUMP_MAX_DURATION - elapsed).max(0.0);
        let gravity = Vector::new(0.0, 0.0, JUMP_GRAVITY);
        let mut predicted_velocity = car.velocity + gravity * time_remaining.max(0.0);
        let mut predicted_car = car.location
            + car.velocity * time_remaining.max(0.0)
            + gravity * (0.5 * time_remaining.max(0.0).powi(2));

        // The packet on the launch frame has not yet incorporated the first
        // jump impulse. On later frames, the live velocity already contains it.
        if elapsed <= 1.0 / 120.0 {
            predicted_velocity = predicted_velocity + car.up() * JUMP_SPEED;
            predicted_car = predicted_car + car.up() * (JUMP_SPEED * time_remaining.max(0.0));
        }
        predicted_velocity = predicted_velocity + car.up() * (JUMP_ACCEL * tau);
        predicted_car = predicted_car + car.up()
            * (JUMP_ACCEL * tau * (time_remaining.max(0.0) - 0.5 * tau));

        let delta = state.approach - predicted_car;
        let delta_direction = delta.normalize();
        let forward = car.forward().normalize();
        if delta_direction.magnitude() > 0.001 && forward.dot(delta_direction) > 0.5 {
            let required_velocity = delta.dot(forward) / time_remaining.max(0.05);
            let mut delta_velocity = required_velocity;
            if car.boost > 0.0 && delta_velocity >= self.boost_accel * MIN_AIR_BOOST_TIME {
                controller.boost = true;
                delta_velocity -= self.boost_accel * MIN_AIR_BOOST_TIME;
            }
            let forward_speed = car.local_velocity().x;
            let acceleration = throttle_acceleration(forward_speed);
            if acceleration > 0.0 && delta_velocity.abs() > acceleration * MIN_AIR_BOOST_TIME {
                controller.throttle =
                    (delta_velocity / (acceleration * MIN_AIR_BOOST_TIME)).clamp(-1.0, 1.0);
            }
        }

        // Hold the first jump long enough to leave the ground, then release
        // until the real contact window. Never burn the second jump early.
        let desired_vertical_speed = (state.approach.z - car.location.z)
            / time_remaining.max(0.05);
        if elapsed <= 0.06
            || (elapsed <= JUMP_MAX_DURATION && predicted_velocity.z < desired_vertical_speed)
        {
            controller.jump = true;
        }

        let close_enough = distance_to_contact < (92.75 + car.hitbox.length) * 1.02;
        let local_target = car.local_location(state.approach).flatten();
        if car.airborne
            && elapsed >= 0.06
            && local_ball.x > 0.0
            && close_enough
            && time_remaining <= 0.18
            && !car.has_double_jumped
            && !car.has_dodged
            && local_target.x > 0.0
        {
            state.dodge_started_at = Some(game_time);
            state.dodge_direction = local_ball.normalize();
            controller.jump = true;
            controller.pitch = -state.dodge_direction.x;
            controller.yaw = state.dodge_direction.y;
            controller.throttle = 1.0;
            controller.boost = false;
            self.jump_shot = Some(state);
            return controller;
        }

        self.jump_shot = Some(state);
        controller
    }

    fn run_ground_flip(&mut self, game_time: f32, flip: FlipState) -> ControllerState {
        let elapsed = game_time - flip.started_at;
        let mut controller = ControllerState::default();

        if elapsed < 0.075 {
            controller.jump = true;
        } else if elapsed < 0.10 {
            controller.jump = false;
        } else if elapsed < if flip.cancel { 0.4 } else { 0.9 } {
            controller.jump = true;
            controller.pitch = -flip.direction.x;
            controller.yaw = flip.direction.y;
        } else {
            self.ground_flip = None;
        }

        controller
    }
}

/// Turn-aware drive execution used by ordinary navigation and shot setup.
/// At large heading errors it scrubs speed before continuing the turn, which
/// prevents the bot from carrying full speed around a missed target in a loop.
fn controlled_drive(car: &CarState, world_target: Vector, requested_speed: f32) -> crate::control::DriveOutput {
    let local_target = car.local_location(world_target.flatten());
    let distance = local_target.flatten().magnitude();
    let angle = local_target.y.atan2(local_target.x).abs();
    let forward_speed = car.local_velocity().x;
    // If the target is well behind and the car has already scrubbed speed,
    // reverse toward it instead of steering forward in a tight orbit. Mirror
    // the local target so PD faces the nose away from the reverse travel line.
    let reverse = angle > 2.35
        && local_target.x < 0.0
        && distance > 450.0
        && forward_speed.abs() < 800.0;

    let mut target_speed = requested_speed;
    let drive_target = if reverse {
        target_speed = -requested_speed.abs().clamp(350.0, 650.0);
        -local_target
    } else {
        if distance < 260.0 {
            target_speed = target_speed.min(450.0);
        }
        if angle > 2.45 {
            target_speed = target_speed.min(350.0);
        } else if angle > 1.85 {
            target_speed = target_speed.min(600.0);
        } else if angle > 1.25 {
            target_speed = target_speed.min(1000.0);
        } else if angle > 0.80 {
            target_speed = target_speed.min(1500.0);
        }
        local_target
    };

    default_drive(
        car,
        DriveInput {
            target_speed,
            local_target: drive_target,
            upside_down: false,
            up: None,
            // Let defaultThrottle engage its radius/velocity-based powerslide
            // only when a tight forward turn is physically appropriate.
            handbrake: !reverse && angle > 1.05,
            boost_accel: BOOST_ACCEL,
        },
    )
}

fn controller_from_drive(output: crate::control::DriveOutput, car: CarState) -> ControllerState {
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

fn team_side(team: u32) -> f32 {
    if team == 0 { -1.0 } else { 1.0 }
}

fn foe_goal(side: f32) -> Vector {
    Vector::new(0.0, -side * 5120.0, 0.0)
}

fn assign_roles(cars: &[CarState], ball: BallState, side: f32) -> Option<TeamRoles> {
    if cars.is_empty() {
        return None;
    }
    if cars.len() == 1 {
        return Some(TeamRoles {
            challenger: cars[0].index,
            goalie: cars[0].index,
            transition: None,
        });
    }

    let own_goal = Vector::new(0.0, side * 5120.0, 0.0);
    let mut challenger = cars[0];
    let mut challenger_score = f32::INFINITY;
    let mut goalie = cars[0];
    let mut goalie_score = f32::INFINITY;

    for car in cars {
        // Mirrors the live Noob Black role score's facing and bad-position
        // penalties, while preserving the actual source's closest-ball basis.
        let facing_badly = car.forward().y * side > 0.0;
        let on_goal_post = car.location.flat_distance(Vector::new(
            if car.location.x >= 0.0 { 900.0 } else { -900.0 },
            side * 5050.0,
            0.0,
        )) <= 100.0;
        let facing_own_goal_from_line = car.location.y * side >= 5050.0
            && car.location.x.abs() <= 900.0
            && car.forward().y * side > 0.0;
        let challenger_candidate = car.location.distance(ball.location)
            + if facing_badly || on_goal_post || facing_own_goal_from_line {
                5000.0
            } else {
                0.0
            };

        if challenger_candidate < challenger_score {
            challenger_score = challenger_candidate;
            challenger = *car;
        }

        let deepest_candidate = car.location.distance(own_goal).min(900.0)
            + (car.location.y - own_goal.y).abs();
        if deepest_candidate < goalie_score {
            goalie_score = deepest_candidate;
            goalie = *car;
        }
    }

    // The live Python agent allows the goalie to challenge only when the ball
    // is in our half and the goalie's nose is already best aligned to it.
    if challenger.index != goalie.index && ball.location.y * side > 0.0 {
        let goalie_angle = (ball.location - goalie.location).angle(goalie.forward());
        let best_other = cars
            .iter()
            .filter(|car| car.index != goalie.index)
            .map(|car| (ball.location - car.location).angle(car.forward()))
            .fold(f32::INFINITY, f32::min);
        if goalie_angle + 0.001 < best_other {
            challenger = goalie;
        }
    }

    let transition = cars
        .iter()
        .find(|car| car.index != challenger.index && car.index != goalie.index)
        .map(|car| car.index);

    Some(TeamRoles {
        challenger: challenger.index,
        goalie: goalie.index,
        transition,
    })
}

/// Volt's relative goal aim: use a higher central target while the ball
/// remains in our half, then return to the normal ground-shot height.
fn shot_aim_point(ball: Vector, side: f32) -> Vector {
    let goal = foe_goal(side);
    Vector::new(goal.x, goal.y, if ball.y * side <= 0.0 { 470.0 } else { 300.0 })
}

/// Port of tools.select_ball_contact. Score the centre, left, right, high and
/// low ball surfaces against both the car's actual approach side and the
/// desired outgoing shot direction. This avoids blindly asking every car to
/// hit the exact centre line of the ball.
fn select_contact_approach(car: CarState, ball: Vector, target: Vector) -> Vector {
    let shot_direction = (target - ball).flatten().normalize();
    if shot_direction.magnitude() < 0.001 {
        return ball.flatten();
    }

    let lateral = Vector::new(-shot_direction.y, shot_direction.x, 0.0).normalize();
    let car_from_ball = (car.location - ball).normalize();
    let candidates = [
        -shot_direction,
        (-shot_direction + lateral * 0.55).normalize(),
        (-shot_direction - lateral * 0.55).normalize(),
        (-shot_direction + Vector::new(0.0, 0.0, 0.30)).normalize(),
        (-shot_direction - Vector::new(0.0, 0.0, 0.30)).normalize(),
    ];

    let normal = candidates
        .into_iter()
        .max_by(|a, b| {
            let score = |normal: Vector| {
                car_from_ball.dot(normal) * 0.75
                    + (-normal).dot(shot_direction) * 0.25
            };
            score(*a).total_cmp(&score(*b))
        })
        .unwrap_or(-shot_direction);

    // Python uses max(120, agent.best_shot_value). Volt's standard
    // best_shot_value is represented here by the same 300-unit approach
    // standoff used by its shot planning configuration.
    cap_in_field(&car, (ball + normal * 300.0).flatten())
}

/// Select a reachable single-jump contact using Volt's C-library jump-time
/// model. The contact height is limited to the single-jump window; higher
/// contacts are left for the later double-jump/aerial port.
fn choose_jump_intercept(
    car: CarState,
    prediction: &BallTrajectory,
    goal: Vector,
    opponents: &[CarState],
) -> Option<GroundIntercept> {
    if car.airborne || prediction.slices.is_empty() {
        return None;
    }

    let mut best: Option<(f32, GroundIntercept)> = None;
    for slice in prediction.slices.iter().filter(|slice| {
        slice.time >= 0.18
            && slice.time <= 3.0
            && slice.location.z > 190.0
            && slice.location.z <= 430.0
    }) {
        let height = slice.location.z - car.location.z;
        if height <= 0.0 || height > 270.0 {
            continue;
        }

        let jump_time = get_jump_time(height, car.velocity.z, -650.0).time as f32;
        if jump_time <= 0.0 || jump_time + 0.12 > slice.time {
            continue;
        }

        let aim = defender_adjusted_aim(slice.location, slice.velocity, goal, opponents);
        // Use the same seven-point front-hitbox contact search as ground
        // shots. This makes jump shots choose centre/top/left/right contact
        // geometry instead of approaching every ball through its centre.
        let (mut approach, _normal, contact_quality, contact_point, contact_face) =
            select_ground_contact(car, slice.location, slice.velocity, aim, slice.time);
        // Ground steering uses the flattened projection, but the jump routine
        // must retain the predicted contact height for its launch geometry.
        approach.z = slice.location.z;
        let delta = (approach - car.location).flatten();
        let distance = delta.magnitude();
        let direction = delta.normalize();
        let forward = car.forward().flatten().normalize();
        let facing_angle = if direction.magnitude() > 0.001 && forward.magnitude() > 0.001 {
            direction.dot(forward).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        };
        let speed = car.velocity.flatten().magnitude();
        let effective_speed = (speed + 650.0).clamp(850.0, 2100.0);
        let eta = distance / effective_speed + (1.0 - facing_angle.cos()) * 0.28;
        if eta > slice.time {
            continue;
        }

        let opponent_race = opponents
            .iter()
            .map(|foe| opponent_arrival_time(*foe, slice.location))
            .fold(f32::INFINITY, f32::min);
        let race_pressure = if opponent_race + 0.08 < eta {
            ((eta - opponent_race) / eta.max(0.10)).clamp(0.0, 1.0) * 0.18
        } else {
            0.0
        };
        let lane_pressure = shot_lane_pressure(slice.location, aim, opponents);
        let score = slice.time
            + facing_angle * 0.08
            + distance / 30000.0
            + (1.0 - contact_quality) * 0.16
            + race_pressure
            + lane_pressure;
        if best.is_none_or(|(best_score, _)| score < best_score) {
            best = Some((score, GroundIntercept {
                time: slice.time,
                ball: slice.location,
                ball_velocity: slice.velocity,
                approach,
                shot_direction: (aim - slice.location).flatten().normalize(),
                contact_quality,
                contact_point,
                contact_face,
            }));
        }
    }

    best.map(|(_, intercept)| intercept)
}

/// Predict a useful point inside the opponent goal window using the
/// ball's velocity. This is deliberately conservative when the ball is not
/// travelling toward the goal plane.
fn predicted_goal_aim(ball: Vector, velocity: Vector, goal: Vector) -> Vector {
    let dy = goal.y - ball.y;
    let time = if velocity.y.abs() > 100.0 {
        dy / velocity.y
    } else {
        -1.0
    };

    if (0.05..3.0).contains(&time) {
        Vector::new(
            (ball.x + velocity.x * time).clamp(-780.0, 780.0),
            goal.y,
            (ball.z + velocity.z * time - 325.0 * time * time).clamp(120.0, 550.0),
        )
    } else {
        Vector::new(0.0, goal.y, 300.0)
    }
}

/// Shift the aim away from an opponent occupying the goal mouth. This is
/// deliberately a modest adjustment: it prefers open space without aiming
/// outside the playable goal aperture or abandoning a viable central shot.
fn defender_adjusted_aim(
    ball: Vector,
    velocity: Vector,
    goal: Vector,
    opponents: &[CarState],
) -> Vector {
    let mut aim = predicted_goal_aim(ball, velocity, goal);
    let keeper = opponents
        .iter()
        .filter(|foe| {
            foe.location.flat_distance(goal) < 2100.0
                && (foe.location.y - goal.y).abs() < 1900.0
                && foe.location.x.abs() < 1750.0
        })
        .min_by(|a, b| {
            a.location.flat_distance(goal)
                .total_cmp(&b.location.flat_distance(goal))
        });

    if let Some(keeper) = keeper {
        let blocked_side = if (keeper.location.x - aim.x).abs() < 120.0 {
            if keeper.location.x >= 0.0 { 1.0 } else { -1.0 }
        } else {
            (keeper.location.x - aim.x).signum()
        };
        aim.x = (aim.x - blocked_side * 430.0).clamp(-720.0, 720.0);
    }
    aim
}

/// Approximate an opponent's arrival time using heading correction, current
/// forward speed, throttle acceleration and only the boost they can actually
/// spend. It is a pressure estimate, not a replacement for full physics.
fn opponent_arrival_time(foe: CarState, target: Vector) -> f32 {
    let local = foe.local_location(target).flatten();
    let distance = local.magnitude();
    if distance < 1.0 {
        return 0.0;
    }
    let angle = local.y.atan2(local.x).abs();
    let speed = foe.velocity.flatten().magnitude();
    let turn_rate = if speed > 1400.0 { 2.0 } else { 3.5 };
    let turn_time = (angle / turn_rate).min(1.4);
    let forward_speed = foe.local_velocity().x.max(0.0);
    let distance_after_turn = (distance - forward_speed * turn_time * 0.45).max(0.0);
    let acceleration = if foe.boost > 0.0 { 1150.0 } else { 650.0 };
    let travel_time = if acceleration > 0.0 {
        ((forward_speed * forward_speed + 2.0 * acceleration * distance_after_turn)
            .sqrt()
            - forward_speed)
            / acceleration
    } else {
        distance_after_turn / forward_speed.max(1.0)
    };
    let boost_credit = (foe.boost / 33.333333).min(travel_time.max(0.0)) * 0.06;
    (turn_time + travel_time - boost_credit).max(0.0)
}

/// Penalise a shot lane occupied by a defender between the ball and the goal.
/// The projection excludes defenders behind the ball or already past the
/// target plane, and smoothly fades with lateral distance from the lane.
fn shot_lane_pressure(ball: Vector, aim: Vector, opponents: &[CarState]) -> f32 {
    let lane = (aim - ball).flatten();
    let lane_length_sq = lane.dot(lane);
    if lane_length_sq < 1.0 {
        return 0.0;
    }
    opponents
        .iter()
        .map(|foe| {
            let relative = (foe.location - ball).flatten();
            let along = relative.dot(lane) / lane_length_sq;
            if !(0.08..0.92).contains(&along) {
                return 0.0;
            }
            let closest = ball + lane * along;
            let lateral_distance = foe.location.flat_distance(closest);
            ((650.0 - lateral_distance).max(0.0) / 650.0)
                * (0.12 + 0.10 * (1.0 - along).clamp(0.0, 1.0))
        })
        .fold(0.0, f32::max)
}

/// Select a ground contact using the predicted arrival position of the car,
/// not just its current location. Side contacts are considered when they give
/// a materially better approach while retaining a useful outgoing angle.
fn rotate_axis(vector: Vector, axis: Vector, angle: f32) -> Vector {
    let axis = axis.normalize();
    if axis.magnitude() < 0.001 || angle.abs() < 0.001 {
        return vector;
    }
    let (sine, cosine) = angle.sin_cos();
    vector * cosine
        + axis.cross(vector) * sine
        + axis * (axis.dot(vector) * (1.0 - cosine))
}

/// Select a ground contact on the actual front face of the car's hitbox.
/// This ports Volt's seven-point Party Cannon contact search, including
/// short-horizon orientation extrapolation and the hitbox-centre offset.
fn select_ground_contact(
    car: CarState,
    ball: Vector,
    ball_velocity: Vector,
    aim: Vector,
    time: f32,
) -> (Vector, Vector, f32, Vector, &'static str) {
    let shot_forward = (aim - ball).flatten().normalize();
    if shot_forward.magnitude() < 0.001 {
        return (cap_in_field(&car, ball.flatten()), Vector::ZERO, 0.0, ball, "centre");
    }

    let horizon = time.clamp(0.0, 0.18);
    let angular_world = car.forward() * car.angular_velocity.x
        + car.right() * car.angular_velocity.y
        + car.up() * car.angular_velocity.z;
    let angular_speed = angular_world.magnitude();
    let mut impact_forward = car.forward();
    let mut impact_up = car.up();
    if angular_speed > 0.05 {
        let angle = (angular_speed * horizon).min(0.65);
        impact_forward = rotate_axis(impact_forward, angular_world, angle).normalize();
        impact_up = rotate_axis(impact_up, angular_world, angle).normalize();
    }

    // RLBot local +Y is car-right (up cross forward).
    let impact_right = impact_up.cross(impact_forward).normalize();
    impact_up = impact_forward.cross(impact_right).normalize();
    if impact_forward.magnitude() < 0.001 || impact_right.magnitude() < 0.001 {
        return (
            cap_in_field(&car, ball - shot_forward * 300.0),
            -shot_forward,
            0.0,
            ball - shot_forward * BALL_RADIUS,
            "centre",
        );
    }

    let half_length = car.hitbox.length.max(80.0) * 0.5;
    let half_width = car.hitbox.width.max(60.0) * 0.5;
    let half_height = car.hitbox.height.max(24.0) * 0.5;
    let hitbox_offset = impact_forward * car.hitbox.offset.x
        + impact_right * car.hitbox.offset.y
        + impact_up * car.hitbox.offset.z;
    let faces: [(&str, f32, f32); 7] = [
        ("centre", 0.0, 0.0),
        ("top_left", -half_width * 0.82, half_height * 0.82),
        ("top_right", half_width * 0.82, half_height * 0.82),
        ("left", -half_width * 0.88, 0.0),
        ("right", half_width * 0.88, 0.0),
        ("top", 0.0, half_height * 0.88),
        ("bottom", 0.0, -half_height * 0.70),
    ];

    let predicted_car = car.location + car.velocity * horizon;
    let relative_velocity = (car.velocity - ball_velocity).flatten().normalize();
    let mut best: Option<(f32, Vector, Vector, Vector, &'static str)> = None;
    for (label, lateral, vertical) in faces {
        let front_offset = impact_forward * half_length
            + impact_right * lateral
            + impact_up * vertical
            + hitbox_offset;
        let predicted_contact = predicted_car + front_offset;
        let incoming_normal = (ball - predicted_contact).normalize();
        if incoming_normal.magnitude() < 0.001 {
            continue;
        }

        let shot_alignment = incoming_normal.dot(shot_forward);
        let front_alignment = incoming_normal.dot(impact_forward);
        let required_car_centre =
            ball - shot_forward * BALL_RADIUS - front_offset;
        let reach_cost = required_car_centre.flat_distance(predicted_car);
        let closing_alignment = if relative_velocity.magnitude() > 0.001 {
            relative_velocity.dot(incoming_normal).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let score = shot_alignment
            + front_alignment.clamp(-1.0, 1.0) * 0.12
            + closing_alignment * 0.08
            - reach_cost * 0.00020;

        if best.is_none_or(|(best_score, ..)| score > best_score) {
            let contact_point = ball - shot_forward * BALL_RADIUS;
            best = Some((
                score,
                required_car_centre,
                incoming_normal,
                contact_point,
                label,
            ));
        }
    }

    let Some((score, approach, normal, contact_point, label)) = best else {
        return (
            cap_in_field(&car, ball - shot_forward * 300.0),
            -shot_forward,
            0.0,
            ball - shot_forward * BALL_RADIUS,
            "centre",
        );
    };

    // Keep quality bounded and useful as a weight in planner scoring.
    let quality = ((score + 0.25) / 1.25).clamp(0.0, 1.0);
    (cap_in_field(&car, approach.flatten()), normal, quality, contact_point, label)
}

/// Choose the dodge vector from the ball offset and intended shot direction.
/// A lateral dodge is reserved for a genuine side contact or a strongly
/// angled shot; ordinary contacts stay forward to avoid unnecessary side flips.
fn ground_dodge_direction(
    local_ball: Vector,
    local_shot_direction: Vector,
    contact_quality: f32,
    contact_face: &'static str,
) -> Vector {
    let ball = local_ball.flatten();
    let shot = local_shot_direction.flatten().normalize();
    if ball.x <= 0.0 {
        return Vector::new(1.0, 0.0, 0.0);
    }

    let lateral_ratio = ball.y / ball.x.max(1.0);
    let face_lateral = if contact_face.contains("left") {
        -1.0
    } else if contact_face.contains("right") {
        1.0
    } else {
        0.0
    };
    let direction = if lateral_ratio.abs() > 0.34 && ball.y.abs() > 35.0 {
        Vector::new(1.0, lateral_ratio.clamp(-0.72, 0.72), 0.0)
    } else if shot.x > 0.25 && shot.y.abs() > 0.42 && contact_quality > 0.35 {
        Vector::new(shot.x.max(0.35), shot.y.clamp(-0.75, 0.75), 0.0)
    } else if face_lateral != 0.0 && contact_quality > 0.55 {
        Vector::new(1.0, face_lateral * 0.30, 0.0)
    } else {
        Vector::new(1.0, 0.0, 0.0)
    };

    direction.normalize()
}

/// Port of Python _shot_approach_target: when the car is poorly aligned
/// with the desired shot, temporarily route around the ball on the side the
/// car is already occupying. The offset fades out before contact.
fn shot_approach_target(
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

/// Estimate arrival time with turn-in cost and a short forward-acceleration
/// simulation. Boost is only credited when the target is reasonably aligned.
fn estimate_ground_eta(car: CarState, target: Vector) -> f32 {
    let local = car.local_location(target).flatten();
    let distance = local.magnitude();
    if distance < 1.0 {
        return 0.0;
    }

    let angle = local.y.atan2(local.x).abs();
    let speed = car.velocity.flatten().magnitude();
    // Convert the same curvature table used by live steering into an expected
    // yaw rate. A low-speed floor covers steering response near a standstill.
    let curvature_turn_rate = speed * curvature(speed);
    let base_turn_rate = if speed < 250.0 {
        2.6
    } else {
        curvature_turn_rate.clamp(1.6, 4.5)
    };
    let turn_sign = local.y.signum();
    let current_yaw_rate = car.angular_velocity.z * turn_sign;
    let turn_rate = (base_turn_rate + current_yaw_rate.clamp(-1.5, 1.5) * 0.45).max(0.8);
    let raw_turn_time = (angle / turn_rate).min(1.6);
    // Fast cars need room to scrub speed before a sharp turn. This mirrors
    // defaultThrottle's curvature speed cap instead of assuming instant steering.
    let braking_angle_blend = ((angle - 0.55) / 0.80).clamp(0.0, 1.0);
    let braking_time = if speed > 1000.0 {
        // A fast car cannot instantly take a tight line. Account for the
        // time needed to scrub toward a controllable cornering speed; scale
        // the cost down for shallow corrections where braking is unnecessary.
        ((speed - 1000.0) / 3500.0).clamp(0.0, 0.45) * braking_angle_blend
    } else {
        0.0
    };
    let turn_time = (raw_turn_time + braking_time).min(1.9);
    let target_direction = (target - car.location).flatten().normalize();
    let projected_speed = car.velocity.flatten().dot(target_direction).max(0.0);
    let distance_during_turn = projected_speed * raw_turn_time * 0.45;
    let mut remaining = (distance - distance_during_turn).max(0.0);
    let mut velocity = projected_speed;
    let can_boost = angle < 0.55;
    // Boost is finite: never let the planner spend more than the car actually has.
    let mut boost_remaining = car.boost.max(0.0) / 33.333333;
    let mut elapsed = turn_time;

    for _ in 0..360 {
        if remaining <= 0.0 {
            break;
        }
        let throttle = throttle_acceleration(velocity);
        let boost = if can_boost && boost_remaining > 0.0 && velocity < 2200.0 {
            boost_remaining = (boost_remaining - 1.0 / 120.0).max(0.0);
            991.6667
        } else {
            0.0
        };
        velocity = (velocity + (throttle + boost) / 120.0).min(MAX_SPEED);
        remaining -= velocity / 120.0;
        elapsed += 1.0 / 120.0;
    }

    elapsed
}

/// Select the earliest useful predicted ground contact. Candidates are
/// ranked by arrival feasibility, contact geometry and the resulting shot
/// angle rather than simply choosing the first reachable slice.
fn choose_ground_intercept(
    car: CarState,
    _ball: BallState,
    prediction: &BallTrajectory,
    goal: Vector,
    opponents: &[CarState],
) -> Option<GroundIntercept> {
    if prediction.slices.is_empty() {
        return None;
    }

    let mut best: Option<(f32, GroundIntercept)> = None;
    for slice in prediction.slices.iter().filter(|slice| {
        slice.time >= 0.04 && slice.time <= 3.0 && slice.location.z <= 190.0
    }) {
        let aim = defender_adjusted_aim(
            slice.location,
            slice.velocity,
            goal,
            opponents,
        );
        let (approach, _normal, contact_quality, contact_point, contact_face) =
            select_ground_contact(car, slice.location, slice.velocity, aim, slice.time);
        let shot_direction = (aim - slice.location).flatten().normalize();
        let distance = approach.flat_distance(car.location);
        let local = car.local_location(approach);
        let direction = (approach - car.location).flatten().normalize();
        let forward = car.forward().flatten().normalize();
        let facing_angle = if direction.magnitude() > 0.001 && forward.magnitude() > 0.001 {
            direction.dot(forward).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        };

        let drive_target =
            shot_approach_target(car, slice.location, approach, shot_direction, slice.time);
        let eta = estimate_ground_eta(car, drive_target);
        if eta > slice.time + 0.08 {
            continue;
        }

        let ball_progress = slice.velocity.flatten().dot(shot_direction);
        let urgency = (slice.time - eta).max(0.0);
        let opponent_race = opponents
            .iter()
            .map(|foe| opponent_arrival_time(*foe, slice.location))
            .fold(f32::INFINITY, f32::min);
        let race_pressure = if opponent_race + 0.08 < eta {
            ((eta - opponent_race) / eta.max(0.10)).clamp(0.0, 1.0) * 0.22
        } else {
            0.0
        };
        let lane_pressure = shot_lane_pressure(slice.location, aim, opponents);
        let score = slice.time * 0.22
            + facing_angle * 0.12
            + distance / 40000.0
            + (1.0 - contact_quality) * 0.24
            - (ball_progress / 3000.0).clamp(-0.08, 0.08)
            + urgency * 0.08
            + (local.y.abs() / 5000.0).min(0.08)
            + race_pressure
            + lane_pressure;

        let intercept = GroundIntercept {
            time: slice.time,
            ball: slice.location,
            ball_velocity: slice.velocity,
            approach,
            shot_direction,
            contact_quality,
            contact_point,
            contact_face,
        };

        if best.is_none_or(|(best_score, _)| score < best_score) {
            best = Some((score, intercept));
        }

    }

    best.map(|(_, intercept)| intercept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn car(index: usize, team: u32, location: Vector) -> CarState {
        CarState {
            index,
            team,
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
            boost: 50.0,
            hitbox: Hitbox { length: 118.0, width: 84.0, height: 36.0, offset: Vector::ZERO },
        }
    }

    #[test]
    fn one_player_is_always_the_challenger() {
        let only = car(2, 0, Vector::ZERO);
        let roles = assign_roles(&[only], BallState::default(), -1.0).unwrap();
        assert_eq!(roles.challenger, 2);
        assert_eq!(roles.goalie, 2);
        assert_eq!(roles.transition, None);
    }

    #[test]
    fn deepest_player_holds_goalie_and_third_player_transitions() {
        let cars = [
            car(0, 0, Vector::new(0.0, -1000.0, 0.0)),
            car(1, 0, Vector::new(1000.0, 500.0, 0.0)),
            car(2, 0, Vector::new(-1000.0, 1000.0, 0.0)),
        ];
        let roles = assign_roles(&cars, BallState { location: Vector::new(0.0, 0.0, 100.0), velocity: Vector::ZERO }, -1.0).unwrap();
        assert_eq!(roles.goalie, 0);
        assert!(roles.transition.is_some());
    }

    #[test]
    fn contact_approach_is_behind_the_ball_relative_to_goal() {
        let car = car(0, 0, Vector::new(0.0, -1000.0, 0.0));
        let ball = Vector::new(0.0, 0.0, 100.0);
        let target = select_contact_approach(car, ball, Vector::new(0.0, 5120.0, 300.0));
        assert!(target.y < ball.y);
        assert_eq!(target.z, 0.0);
    }

    #[test]
    fn contact_selection_can_choose_a_flank_when_car_is_offset() {
        let car = car(0, 0, Vector::new(800.0, -500.0, 0.0));
        let ball = Vector::new(0.0, 0.0, 100.0);
        let target = select_contact_approach(car, ball, Vector::new(0.0, 5120.0, 300.0));
        assert!(target.x.abs() > 10.0);
    }

    #[test]
    fn ground_contact_accounts_for_predicted_car_motion() {
        let mut me = car(0, 0, Vector::new(500.0, -600.0, 0.0));
        me.velocity = Vector::new(-500.0, 900.0, 0.0);
        let ball = Vector::new(0.0, 0.0, 100.0);
        let aim = Vector::new(0.0, 5120.0, 300.0);
        let (approach, normal, quality, contact_point, face) =
            select_ground_contact(me, ball, Vector::ZERO, aim, 0.5);

        assert!(approach.x.is_finite() && approach.y.is_finite());
        assert!(normal.magnitude() > 0.9);
        assert!(quality.is_finite());
        assert!(contact_point.distance(ball) <= BALL_RADIUS + 0.01);
        assert!(!face.is_empty());
    }

    #[test]
    fn hitbox_offset_moves_the_selected_car_centre_target() {
        let centred = car(0, 0, Vector::new(0.0, -500.0, 0.0));
        let mut offset = centred;
        offset.hitbox.offset.x = 20.0;
        let ball = Vector::new(0.0, 0.0, 100.0);
        let aim = Vector::new(0.0, 5120.0, 300.0);
        let (plain, _, _, _, _) =
            select_ground_contact(centred, ball, Vector::ZERO, aim, 0.2);
        let (shifted, _, _, _, _) =
            select_ground_contact(offset, ball, Vector::ZERO, aim, 0.2);

        assert!((shifted.x - (plain.x - 20.0)).abs() < 0.01);
    }

    #[test]
    fn ground_dodge_stays_forward_for_centre_contact() {
        let direction = ground_dodge_direction(
            Vector::new(150.0, 20.0, 0.0),
            Vector::new(1.0, 0.05, 0.0),
            0.9,
            "centre",
        );
        assert!(direction.x > 0.99);
        assert!(direction.y.abs() < 0.01);
    }

    #[test]
    fn ground_dodge_uses_lateral_direction_only_for_side_geometry() {
        let right = ground_dodge_direction(
            Vector::new(120.0, 90.0, 0.0),
            Vector::new(1.0, 0.0, 0.0),
            0.8,
            "centre",
        );
        let left = ground_dodge_direction(
            Vector::new(120.0, -90.0, 0.0),
            Vector::new(1.0, 0.0, 0.0),
            0.8,
            "centre",
        );
        assert!(right.y > 0.0);
        assert!(left.y < 0.0);
    }

    #[test]
    fn selected_front_face_can_bias_a_clean_side_dodge() {
        let direction = ground_dodge_direction(
            Vector::new(180.0, 10.0, 0.0),
            Vector::new(1.0, 0.0, 0.0),
            0.9,
            "top_right",
        );
        assert!(direction.y > 0.0);
    }

    #[test]
    fn ground_eta_uses_finite_boost_and_existing_turn_direction() {
        let mut no_boost = car(0, 0, Vector::ZERO);
        no_boost.boost = 0.0;
        let mut boosted = no_boost;
        boosted.boost = 100.0;
        // The car's forward axis is world +X at zero yaw, so keep this
        // target aligned to ensure the boost comparison tests boost itself.
        let target = Vector::new(2200.0, 0.0, 0.0);
        assert!(estimate_ground_eta(no_boost, target) > estimate_ground_eta(boosted, target));

        let mut already_turning = no_boost;
        already_turning.angular_velocity.z = 1.0;
        let turning_target = Vector::new(1500.0, 1000.0, 0.0);
        let turning_eta = estimate_ground_eta(already_turning, turning_target);
        let neutral_eta = estimate_ground_eta(no_boost, turning_target);
        assert!(turning_eta < neutral_eta);
    }

    #[test]
    fn ground_eta_accounts_for_speed_scrub_before_a_sharp_turn() {
        let mut fast = car(0, 0, Vector::ZERO);
        fast.velocity = Vector::new(2000.0, 0.0, 0.0);
        let slow = car(1, 0, Vector::ZERO);
        let target = Vector::new(0.0, 1500.0, 0.0);

        assert!(estimate_ground_eta(fast, target) > estimate_ground_eta(slow, target));
    }

    #[test]
    fn predicted_goal_aim_leads_a_lateral_ball_toward_open_goal_space() {
        let aim = predicted_goal_aim(
            Vector::new(-300.0, 0.0, 200.0),
            Vector::new(300.0, 1800.0, 0.0),
            Vector::new(0.0, 5120.0, 300.0),
        );

        assert!(aim.x > 300.0);
        assert!((120.0..=550.0).contains(&aim.z));
        assert_eq!(aim.y, 5120.0);
    }

    #[test]
    fn ground_contact_prefers_forward_velocity_toward_ball() {
        let mut me = car(0, 0, Vector::new(0.0, -700.0, 0.0));
        me.velocity = Vector::new(0.0, 900.0, 0.0);
        let (_, normal, quality, _, _) = select_ground_contact(
            me,
            Vector::new(0.0, 0.0, 100.0),
            Vector::ZERO,
            Vector::new(0.0, 5120.0, 300.0),
            0.2,
        );

        // The selected front-face normal points from the car's contact
        // surface toward the ball, which is in +Y from this car.
        assert!(normal.y > 0.5);
        assert!(quality > 0.7);
    }

    #[test]
    fn shot_approach_flank_fades_out_as_contact_nears() {
        let me = car(0, 0, Vector::new(900.0, -500.0, 0.0));
        let ball = Vector::new(0.0, 0.0, 100.0);
        let final_target = Vector::new(0.0, -300.0, 0.0);
        let shot_direction = Vector::new(0.0, 1.0, 0.0);

        let early = shot_approach_target(me, ball, final_target, shot_direction, 1.5);
        let late = shot_approach_target(me, ball, final_target, shot_direction, 0.30);
        assert!(early.x.abs() > late.x.abs());
        assert_eq!(late, final_target);
    }

    #[test]
    fn goal_aim_moves_away_from_a_goalkeeper() {
        let ball = Vector::new(0.0, 0.0, 150.0);
        let goal = Vector::new(0.0, 5120.0, 0.0);
        let keeper = car(1, 1, Vector::new(0.0, 4900.0, 0.0));
        let aim = defender_adjusted_aim(ball, Vector::new(0.0, 1800.0, 0.0), goal, &[keeper]);
        assert!(aim.x.abs() > 300.0);
        assert!(aim.x.abs() <= 720.0);
    }

    #[test]
    fn shot_lane_pressure_only_counts_defenders_between_ball_and_goal() {
        let ball = Vector::new(0.0, 0.0, 100.0);
        let aim = Vector::new(0.0, 5120.0, 300.0);
        let blocker = car(1, 1, Vector::new(0.0, 2400.0, 0.0));
        let behind = car(2, 1, Vector::new(0.0, -300.0, 0.0));
        assert!(shot_lane_pressure(ball, aim, &[blocker]) > 0.1);
        assert_eq!(shot_lane_pressure(ball, aim, &[behind]), 0.0);
    }

    #[test]
    fn opponent_arrival_time_rewards_a_facing_defender() {
        let facing = car(1, 1, Vector::new(0.0, 1000.0, 0.0));
        let turned_away = car(2, 1, Vector::new(0.0, 1000.0, 0.0));
        let target = Vector::new(0.0, 2000.0, 0.0);
        let mut facing = facing;
        facing.orientation = Matrix3::new(0.0, std::f32::consts::PI / 2.0, 0.0);
        let mut turned_away = turned_away;
        turned_away.orientation = Matrix3::new(0.0, -std::f32::consts::PI / 2.0, 0.0);
        assert!(opponent_arrival_time(facing, target) < opponent_arrival_time(turned_away, target));
    }

    #[test]
    fn ground_eta_increases_for_a_large_heading_change() {
        let me = car(0, 0, Vector::ZERO);
        // At zero yaw, +X is straight ahead and +Y requires a 90-degree turn.
        let straight = estimate_ground_eta(me, Vector::new(1500.0, 0.0, 0.0));
        let turning = estimate_ground_eta(me, Vector::new(0.0, 1500.0, 0.0));
        assert!(turning > straight);
    }

    #[test]
    fn jump_intercept_uses_single_jump_height_and_timing_window() {
        let me = car(0, 0, Vector::new(0.0, -1000.0, 0.0));
        let prediction = BallTrajectory {
            slices: vec![
                crate::prediction::BallSlice {
                    time: 0.50,
                    location: Vector::new(0.0, -300.0, 260.0),
                    velocity: Vector::ZERO,
                },
                crate::prediction::BallSlice {
                    time: 1.50,
                    location: Vector::new(0.0, -200.0, 260.0),
                    velocity: Vector::ZERO,
                },
            ],
        };
        let shot = choose_jump_intercept(
            me,
            &prediction,
            Vector::new(0.0, 5120.0, 300.0),
            &[],
        ).unwrap();

        assert_eq!(shot.time, 1.50);
        assert!(shot.ball.z > 190.0);
        assert_eq!(shot.approach.z, shot.ball.z);
    }

    #[test]
    fn turn_aware_drive_reduces_speed_for_a_target_behind_the_car() {
        let me = car(0, 0, Vector::ZERO);
        let output = controlled_drive(&me, Vector::new(-1800.0, 0.0, 0.0), 2300.0);
        assert!(output.target_speed < 0.0);
        assert_eq!(output.boost, Some(false));
    }

    #[test]
    fn jump_contact_selection_uses_a_real_front_face() {
        let mut me = car(0, 0, Vector::new(250.0, -500.0, 0.0));
        me.velocity = Vector::new(0.0, 700.0, 0.0);
        let prediction = BallTrajectory {
            slices: vec![
                crate::prediction::BallSlice {
                    time: 0.55,
                    location: Vector::new(0.0, 0.0, 260.0),
                    velocity: Vector::ZERO,
                },
                crate::prediction::BallSlice {
                    time: 1.45,
                    location: Vector::new(0.0, 0.0, 260.0),
                    velocity: Vector::ZERO,
                },
            ],
        };
        let shot = choose_jump_intercept(
            me,
            &prediction,
            Vector::new(0.0, 5120.0, 300.0),
            &[],
        ).unwrap();
        assert!(shot.contact_quality.is_finite());
        assert!(shot.contact_point.distance(shot.ball) <= BALL_RADIUS + 0.01);
        assert!(!shot.contact_face.is_empty());
    }

    #[test]
    fn jump_shot_holds_jump_then_uses_directional_dodge() {
        let mut brain = BotBrain::new();
        let mut me = car(0, 0, Vector::ZERO);
        me.airborne = true;
        me.air_state = AirState::InAir;
        let state = JumpShotState {
            started_at: 1.0,
            intercept_time: 1.2,
            ball: Vector::new(100.0, 0.0, 220.0),
            approach: Vector::new(50.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            quick: true,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
        };

        let controller = brain.run_jump_shot(me, state, 1.10);
        assert!(controller.jump);
        assert_eq!(controller.pitch, -1.0);
        assert_eq!(controller.yaw, 0.0);
        assert!(brain.jump_shot.unwrap().dodge_started_at.is_some());
    }

    #[test]
    fn jump_shot_uses_air_boost_to_correct_the_contact_path() {
        let mut brain = BotBrain::new();
        let mut me = car(0, 0, Vector::ZERO);
        me.airborne = true;
        me.air_state = AirState::InAir;
        let state = JumpShotState {
            started_at: 1.0,
            intercept_time: 2.0,
            ball: Vector::new(1000.0, 0.0, 300.0),
            approach: Vector::new(1000.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            quick: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
        };

        let controller = brain.run_jump_shot(me, state, 1.10);
        assert!(controller.boost);
        assert!(controller.throttle > 0.0);
    }

    #[test]
    fn expired_jump_shot_is_cleared_instead_of_looping() {
        let mut brain = BotBrain::new();
        let me = car(0, 0, Vector::ZERO);
        let state = JumpShotState {
            started_at: 1.0,
            intercept_time: 1.0,
            ball: Vector::new(300.0, 0.0, 200.0),
            approach: Vector::new(200.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            quick: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
        };

        let controller = brain.run_jump_shot(me, state, 1.20);
        assert!(!controller.jump);
        assert!(brain.jump_shot.is_none());
    }

    #[test]
    fn normal_jump_shot_uses_s_tilt_but_quick_shot_does_not() {
        let mut brain = BotBrain::new();
        let me = car(0, 0, Vector::ZERO);
        let state = JumpShotState {
            started_at: 1.0,
            intercept_time: 1.5,
            ball: Vector::new(300.0, 0.0, 200.0),
            approach: Vector::new(200.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 1.0, 0.0),
            quick: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
        };

        let normal = brain.run_jump_shot(me, state, 1.01);
        assert!(normal.jump);
        assert_eq!(normal.pitch, 1.0);
        assert!(normal.yaw > 0.0);
        assert!(normal.roll < 0.0);

        let quick = JumpShotState { quick: true, ..state };
        let fast = brain.run_jump_shot(me, quick, 1.01);
        assert_ne!(fast.pitch, 1.0);
        assert_eq!(fast.roll, 0.0);
    }

    #[test]
    fn ground_flip_releases_jump_before_directional_dodge() {
        let mut brain = BotBrain::new();
        let flip = FlipState {
            started_at: 1.0,
            direction: Vector::new(0.8, 0.6, 0.0),
            cancel: false,
        };

        assert!(brain.run_ground_flip(1.05, flip).jump);
        assert!(!brain.run_ground_flip(1.08, flip).jump);

        let dodge = brain.run_ground_flip(1.12, flip);
        assert!(dodge.jump);
        assert!((dodge.pitch + 0.8).abs() < 1e-6);
        assert!((dodge.yaw - 0.6).abs() < 1e-6);
    }

    #[test]
    fn kickoff_selects_nearest_teammate_as_kicker() {
        let mut brain = BotBrain::new();
        let me = car(0, 0, Vector::new(-3000.0, -3000.0, 0.0));
        let mate = car(1, 0, Vector::new(2000.0, -2500.0, 0.0));
        let ball = BallState::default();
        let plan = brain.choose_plan(
            me,
            &[me, mate],
            ball,
            &BallTrajectory { slices: Vec::new() },
            1.0,
            MatchState { round_active: true, kickoff: true, ..MatchState::default() },
        );
        assert_eq!(plan.decision, Decision::KickoffCheat);
        assert_eq!(plan.target, Vector::new(0.0, 1650.0, 0.0));
    }
    #[test]
    fn kickoff_uses_authoritative_match_phase_even_when_ball_is_not_at_centre() {
        let mut brain = BotBrain::new();
        let me = car(0, 0, Vector::new(0.0, -1000.0, 0.0));
        let teammate = car(1, 0, Vector::new(1000.0, -2000.0, 0.0));
        let ball = BallState {
            location: Vector::new(300.0, 200.0, 120.0),
            velocity: Vector::new(400.0, 0.0, 0.0),
        };

        let plan = brain.choose_plan(
            me,
            &[me, teammate],
            ball,
            &BallTrajectory { slices: Vec::new() },
            1.0,
            MatchState { round_active: true, kickoff: true, ..MatchState::default() },
        );

        assert_eq!(plan.decision, Decision::KickoffKicker);
    }

    #[test]
    fn centred_slow_ball_does_not_fake_a_kickoff_outside_kickoff_phase() {
        let mut brain = BotBrain::new();
        let me = car(0, 0, Vector::new(0.0, -1000.0, 0.0));
        let ball = BallState::default();

        let plan = brain.choose_plan(
            me,
            &[me],
            ball,
            &BallTrajectory { slices: Vec::new() },
            1.0,
            MatchState { round_active: true, kickoff: false, ..MatchState::default() },
        );

        assert_ne!(plan.decision, Decision::KickoffKicker);
    }

}
