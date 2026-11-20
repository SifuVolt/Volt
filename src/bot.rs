//! Active Volt decision and shot-execution layer.
//!
//! Kickoff assignment, team roles, defensive priorities, prediction-based
//! ground/jump contacts, finite-boost reachability, contact geometry and
//! timed shot execution all feed the live RLBot controller.

use crate::{
    control::{default_drive, default_pd, DriveInput},
    math::Vector,
    physics::drive::{dodge_impulse, throttle_acceleration},
    prediction::BallTrajectory,
    shot_physics::get_jump_time,
    state::{BallState, CarState},
};
use rlbot::flat::ControllerState;

const BALL_RADIUS: f32 = 92.75;
const MAX_SPEED: f32 = 2300.0;
const BOOST_ACCEL: f32 = 991.6667;

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
    last_kickoff_time: f32,
    ground_flip: Option<FlipState>,
    jump_shot: Option<JumpShotState>,
}

impl BotBrain {
    pub fn new() -> Self {
        Self {
            kickoff_complete: false,
            last_kickoff_time: -999.0,
            ground_flip: None,
            jump_shot: None,
        }
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
    ) -> ControllerState {
        let plan = self.choose_plan(me, cars, ball, prediction, game_time);
        self.drive_to(me, plan, game_time)
    }

    fn choose_plan(
        &mut self,
        me: CarState,
        cars: &[CarState],
        ball: BallState,
        prediction: &BallTrajectory,
        game_time: f32,
    ) -> Plan {
        let side = team_side(me.team);
        let kickoff_position = ball.location.flat_distance(Vector::ZERO) < 100.0;
        let kickoff_ball_slow = ball.velocity.magnitude() < 100.0;
        let kickoff_active = kickoff_position && kickoff_ball_slow;

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
            let intercept = choose_ground_intercept(me, ball, prediction, aim);
            let jump = if urgent_save { None } else { choose_jump_intercept(me, prediction, aim) };
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
        let intercept = choose_ground_intercept(me, ball, prediction, aim);
        let jump = choose_jump_intercept(me, prediction, aim);
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
                let shot_direction = (shot_aim_point(jump.ball, team_side(car.team)) - jump.ball)
                    .flatten()
                    .normalize();
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

            let output = default_drive(
                &car,
                DriveInput {
                    target_speed: required_speed.max(900.0),
                    local_target: local_approach.flatten(),
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel: BOOST_ACCEL,
                },
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
            let local_future_velocity = car.local(car.velocity + Vector::new(0.0, 0.0, -650.0) * t).flatten();
            let angle = local_offset.y.atan2(local_offset.x).abs();

            let output = default_drive(
                &car,
                DriveInput {
                    target_speed: required_speed,
                    local_target: car.local_location(drive_target.flatten()),
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel: BOOST_ACCEL,
                },
            );
            let velocity = output.car_speed;
            let impulse = dodge_impulse(car.velocity.flatten(), MAX_SPEED);
            let aligned_contact = (velocity.abs() < 100.0
                && distance_remaining < car.hitbox.length / 2.0)
                || (local_offset.y.abs() < BALL_RADIUS
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
                    direction: local_ball.flatten().normalize(),
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

        let output = default_drive(
            &car,
            DriveInput {
                target_speed,
                local_target,
                upside_down: false,
                up: None,
                handbrake: false,
                boost_accel: BOOST_ACCEL,
            },
        );

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
        let elapsed = game_time - state.started_at;
        let time_remaining = state.intercept_time - game_time;
        let local_ball = car.local_location(state.ball).flatten();
        let distance_to_contact = car.location.distance(state.approach);
        let mut controller = ControllerState::default();

        if time_remaining < -0.25 || (elapsed > 0.30 && !car.airborne && state.dodge_started_at.is_none()) {
            self.jump_shot = None;
            return controller;
        }

        let local_shot_direction = car.local(state.shot_direction).flatten();
        let (_, pd) = default_pd(&car, local_shot_direction, false, None);
        controller.pitch = pd.pitch;
        controller.yaw = pd.yaw;
        controller.roll = pd.roll;

        let hold_s = !state.quick && state.ball.z > car.location.z + 65.0;
        if hold_s {
            controller.pitch = 1.0;
            if pd.yaw.abs() > 0.08 {
                controller.roll = -0.20 * pd.yaw.signum();
            } else {
                controller.roll = 0.0;
            }
        }

        // Keep the first jump held briefly, then release it before the dodge.
        if elapsed <= 0.075 {
            controller.jump = true;
        }

        if let Some(dodge_time) = state.dodge_started_at {
            let dodge_elapsed = game_time - dodge_time;
            controller.jump = dodge_elapsed <= 0.045;
            controller.pitch = -state.dodge_direction.x;
            controller.yaw = state.dodge_direction.y;
            if dodge_elapsed > 0.12 {
                self.jump_shot = None;
            } else {
                self.jump_shot = Some(state);
            }
            return controller;
        }

        let close_enough = distance_to_contact < (92.75 + car.hitbox.length) * 1.02;
        if elapsed >= 0.075
            && car.airborne
            && local_ball.x > 0.0
            && close_enough
            && time_remaining <= 0.18
            && !car.has_double_jumped
            && !car.has_dodged
        {
            state.dodge_started_at = Some(game_time);
            state.dodge_direction = local_ball.normalize();
            controller.jump = true;
            controller.pitch = -state.dodge_direction.x;
            controller.yaw = state.dodge_direction.y;
            self.jump_shot = Some(state);
            return controller;
        }

        if time_remaining < -0.10 {
            self.jump_shot = None;
        } else {
            self.jump_shot = Some(state);
        }
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
    Vector::new(0.0, -side * 5120.0, if ball.y * side <= 0.0 { 470.0 } else { 300.0 })
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
    (ball + normal * 300.0).flatten()
}

/// Select a reachable single-jump contact using Volt's C-library jump-time
/// model. The contact height is limited to the single-jump window; higher
/// contacts are left for the later double-jump/aerial port.
fn choose_jump_intercept(
    car: CarState,
    prediction: &BallTrajectory,
    goal: Vector,
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

        let mut approach = select_contact_approach(car, slice.location, goal);
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

        let score = slice.time + facing_angle * 0.08 + distance / 30000.0;
        if best.is_none_or(|(best_score, _)| score < best_score) {
            best = Some((score, GroundIntercept {
                time: slice.time,
                ball: slice.location,
                ball_velocity: slice.velocity,
                approach,
                shot_direction: (goal - slice.location).flatten().normalize(),
                contact_quality: 1.0,
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

/// Select a ground contact using the predicted arrival position of the car,
/// not just its current location. Side contacts are considered when they give
/// a materially better approach while retaining a useful outgoing angle.
fn select_ground_contact(
    car: CarState,
    ball: Vector,
    ball_velocity: Vector,
    aim: Vector,
    time: f32,
) -> (Vector, Vector, f32) {
    let shot_direction = (aim - ball).flatten().normalize();
    if shot_direction.magnitude() < 0.001 {
        return (ball.flatten(), Vector::ZERO, 0.0);
    }

    let lateral = Vector::new(-shot_direction.y, shot_direction.x, 0.0);
    let projected_car = car.location + car.velocity.flatten() * time.min(0.35);
    let car_from_ball = (projected_car - ball).flatten().normalize();
    // Score the approach using closing velocity, not car velocity alone: a
    // moving ball changes which face of the car can meet it cleanly.
    let velocity_direction = (car.velocity - ball_velocity).flatten().normalize();
    let candidates = [
        -shot_direction,
        (-shot_direction + lateral * 0.55).normalize(),
        (-shot_direction - lateral * 0.55).normalize(),
    ];

    let mut best = (ball - shot_direction * 300.0, -shot_direction, f32::NEG_INFINITY);
    for normal in candidates {
        let approach_alignment = car_from_ball.dot(normal);
        let shot_alignment = (-normal).dot(shot_direction);
        let velocity_alignment = if velocity_direction.magnitude() > 0.001 {
            -velocity_direction.dot(normal)
        } else {
            approach_alignment
        };
        // The outgoing shot angle remains important, but a reachable flank
        // can beat a centre-line approach that would require a large cut.
        let score = approach_alignment * 0.52
            + shot_alignment * 0.38
            + velocity_alignment * 0.10;

        if score > best.2 {
            best = (ball + normal * 300.0, normal, score);
        }
    }

    (best.0.flatten(), best.1, best.2)
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
    let base_turn_rate = (2.8 - 0.75 * (speed / MAX_SPEED).clamp(0.0, 1.0)).max(1.4);
    let turn_sign = local.y.signum();
    let current_yaw_rate = car.angular_velocity.z * turn_sign;
    let turn_rate = (base_turn_rate + current_yaw_rate.clamp(-1.5, 1.5) * 0.45).max(0.8);
    let turn_time = (angle / turn_rate).min(1.6);
    let forward_speed = car.local_velocity().x.max(0.0);
    let distance_during_turn = forward_speed * turn_time * 0.35;
    let mut remaining = (distance - distance_during_turn).max(0.0);
    let mut velocity = forward_speed;
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
) -> Option<GroundIntercept> {
    if prediction.slices.is_empty() {
        return None;
    }

    let mut best: Option<(f32, GroundIntercept)> = None;
    for slice in prediction.slices.iter().filter(|slice| {
        slice.time >= 0.04 && slice.time <= 3.0 && slice.location.z <= 190.0
    }) {
        let aim = predicted_goal_aim(slice.location, slice.velocity, goal);
        let (approach, _normal, contact_quality) =
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
        let score = slice.time * 0.22
            + facing_angle * 0.12
            + distance / 40000.0
            + (1.0 - contact_quality) * 0.24
            - (ball_progress / 3000.0).clamp(-0.08, 0.08)
            + urgency * 0.08
            + (local.y.abs() / 5000.0).min(0.08);

        let intercept = GroundIntercept {
            time: slice.time,
            ball: slice.location,
            ball_velocity: slice.velocity,
            approach,
            shot_direction,
            contact_quality,
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
        let (approach, normal, quality) =
            select_ground_contact(me, ball, Vector::ZERO, aim, 0.5);

        assert!(approach.x.is_finite() && approach.y.is_finite());
        assert!(normal.magnitude() > 0.9);
        assert!(quality.is_finite());
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
        let (_, normal, quality) = select_ground_contact(
            me,
            Vector::new(0.0, 0.0, 100.0),
            Vector::ZERO,
            Vector::new(0.0, 5120.0, 300.0),
            0.2,
        );

        assert!(normal.y < -0.5);
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
        ).unwrap();

        assert_eq!(shot.time, 1.50);
        assert!(shot.ball.z > 190.0);
        assert_eq!(shot.approach.z, shot.ball.z);
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
            approach: Vector::ZERO,
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
    fn normal_jump_shot_uses_s_tilt_but_quick_shot_does_not() {
        let mut brain = BotBrain::new();
        let me = car(0, 0, Vector::ZERO);
        let state = JumpShotState {
            started_at: 1.0,
            intercept_time: 1.5,
            ball: Vector::new(300.0, 0.0, 200.0),
            approach: Vector::new(200.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            quick: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
        };

        let normal = brain.run_jump_shot(me, state, 1.01);
        assert!(normal.jump);
        assert_eq!(normal.pitch, 1.0);

        let quick = JumpShotState { quick: true, ..state };
        let fast = brain.run_jump_shot(me, quick, 1.01);
        assert_ne!(fast.pitch, 1.0);
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
        let plan = brain.choose_plan(me, &[me, mate], ball, &BallTrajectory { slices: Vec::new() }, 1.0);
        assert_eq!(plan.decision, Decision::KickoffCheat);
        assert_eq!(plan.target, Vector::new(0.0, 1650.0, 0.0));
    }
}
