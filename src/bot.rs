//! Active Volt decision layer.
//!
//! This is the first gameplay-connected port of Volt's current run_logic
//! priorities: kickoff assignment, Noob Black-style challenger/goalie roles,
//! urgent own-half saves, and Grid defensive positioning. Shot routines are
//! still being translated; until they are connected, the challenger uses the
//! authoritative RLBot prediction to select a reachable ground approach.

use crate::{
    control::{default_drive, DriveInput},
    physics::drive::dodge_impulse,
    math::Vector,
    prediction::BallTrajectory,
    state::{BallState, CarState},
};
use rlbot::flat::ControllerState;

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
    approach: Vector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Plan {
    decision: Decision,
    target: Vector,
    target_speed: f32,
    shot: Option<GroundIntercept>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FlipState {
    started_at: f32,
    direction: Vector,
    cancel: bool,
}

#[derive(Debug, Default)]
pub struct BotBrain {
    kickoff_complete: bool,
    last_kickoff_time: f32,
    ground_flip: Option<FlipState>,
}

impl BotBrain {
    pub fn new() -> Self {
        Self {
            kickoff_complete: false,
            last_kickoff_time: -999.0,
            ground_flip: None,
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
                };
            }

            if teammates.len() <= 2 || cheater == Some(me.index) {
                return Plan {
                    decision: Decision::KickoffCheat,
                    target: Vector::new(0.0, -side * 1650.0, 0.0),
                    target_speed: 1200.0,
                    shot: None,
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
            };
        }

        if is_goalie && !is_challenger {
            return Plan {
                decision: Decision::Goalkeeper,
                target: Vector::new(0.0, side * 4970.0, 0.0),
                target_speed: 1500.0,
                shot: None,
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
            };
        }

        if is_challenger {
            let intercept = choose_ground_intercept(me, ball, prediction, shot_aim_point(ball.location, side));
            let target = intercept
                .map(|shot| shot.approach)
                .unwrap_or_else(|| select_contact_approach(me, ball.location, shot_aim_point(ball.location, side)));
            return Plan {
                decision: if urgent_save {
                    Decision::EmergencyChallenge
                } else {
                    Decision::GroundChallenge
                },
                target,
                target_speed: MAX_SPEED,
                shot: if urgent_save { None } else { intercept },
            };
        }

        // In 1v1 there is no separate goalie: the sole player challenges.
        let intercept = choose_ground_intercept(me, ball, prediction, shot_aim_point(ball.location, side));
        Plan {
            decision: Decision::GroundChallenge,
            target: intercept
                .map(|shot| shot.approach)
                .unwrap_or_else(|| select_contact_approach(me, ball.location, shot_aim_point(ball.location, side))),
            target_speed: MAX_SPEED,
            shot: intercept,
        }
    }

    fn drive_to(&mut self, car: CarState, plan: Plan, game_time: f32) -> ControllerState {
        if let Some(shot) = plan.shot {
            if let Some(flip) = self.ground_flip {
                return self.run_ground_flip(game_time, flip);
            }
            let t = shot.time;
            let local_offset = car.local_location(shot.approach.flatten());
            let distance_remaining = (shot.approach - car.location).flat_distance(Vector::ZERO)
                - car.hitbox.length / 3.0;
            let required_speed = (distance_remaining.max(0.0) / t.max(0.001)).min(MAX_SPEED);
            let local_ball = car.local_location(shot.ball);
            let local_future_velocity = car.local(car.velocity + Vector::new(0.0, 0.0, -650.0) * t).flatten();
            let angle = local_offset.y.atan2(local_offset.x).abs();

            let output = default_drive(
                &car,
                DriveInput {
                    target_speed: required_speed,
                    local_target: car.local_location(plan.target.flatten()),
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

            if !car.airborne
                && (0.1..0.35).contains(&t)
                && (velocity.abs() + impulse).min(MAX_SPEED) <= required_speed.max(1900.0_f32.min(required_speed))
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

            let mut controller = controller_from_drive(output, car);
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

/// Select the earliest prediction sample that the current ground-drive model
/// can plausibly reach. The authoritative RLBot prediction already includes
/// collisions; this function only ranks its samples and does not resimulate it.
fn choose_ground_intercept(
    car: CarState,
    ball: BallState,
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
        let target = select_contact_approach(car, slice.location, goal);
        let delta = (target - car.location).flatten();
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
        if eta > slice.time + 0.10 {
            continue;
        }

        // Earlier feasible contacts win; alignment breaks close ties.
        let score = slice.time + facing_angle * 0.08 + distance / 30000.0;
        if best.is_none_or(|(best_score, _)| score < best_score) {
            best = Some((score, GroundIntercept {
                time: slice.time,
                ball: slice.location,
                approach: target,
            }));
        }
    }

    let _ = ball;
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
