//! Active Volt decision and shot-execution layer.
//!
//! Kickoff assignment, team roles, defensive priorities, prediction-based
//! ground/jump contacts, finite-boost reachability, contact geometry and
//! timed shot execution all feed the live RLBot controller.

use crate::{
    aerial::{find_aerial_candidate, find_fast_aerial_candidate, party_cannon_contact_geometry, party_cannon_shot_direction, wall_shot_adjust_jump, AerialCandidate, AerialShot},
    control::{default_drive, default_pd, DriveInput},
    defense::GridDefense,
    math::{utils::cap_in_field, Vector},
    physics::drive::{curvature, dodge_impulse, throttle_acceleration},
    prediction::{BallSlice, BallTrajectory},
    planning::{find_wall_drive, wall_approach_geometry, WallDriveMode, WallDrivePlan},
    shot_physics::{double_jump_shot_is_viable, get_jump_time, ground_shot_is_viable, jump_shot_is_viable, ShotType},
    shot_parser::{runtime_ground_reachability, ShotTargets},
    shot_planner::{find_best_any_shot, find_best_pass_shot, find_best_shot, find_fast_intercept, find_fast_intercept_candidate, grid_jump_time, opponent_reaches_before, ShotCandidate},
    state::{BallState, BoostPadState, CarState, MatchState},
};
use std::collections::HashMap;
use crate::state::TeamplayMessage;
use rlbot::flat::ControllerState;

const BALL_RADIUS: f32 = 92.75;
const MAX_SPEED: f32 = 2300.0;
const BOOST_ACCEL: f32 = 991.6667;
const JUMP_SPEED: f32 = 291.6667;
const JUMP_ACCEL: f32 = 1458.3333;
const JUMP_MAX_DURATION: f32 = 0.20;
const MIN_AIR_BOOST_TIME: f32 = 1.0 / 30.0;

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
    AerialChallenge,
    WallDrive,
    Boost,
    TransitionBump,
    TransitionReceive,
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
    shot_type: ShotType,
    side_shot: Option<SideShotPlan>,
    targets: ShotTargets,
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
struct PendingDoubleJump {
    shot: GroundIntercept,
    intercept_time: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FlipState {
    started_at: f32,
    direction: Vector,
    cancel: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SideShotPlan {
    side_sign: f32,
    lateral_distance: f32,
    shot_direction: Vector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct JumpShotState {
    started_at: f32,
    intercept_time: f32,
    ball: Vector,
    ball_velocity: Vector,
    approach: Vector,
    shot_direction: Vector,
    contact_face: &'static str,
    contact_face_label: Option<&'static str>,
    goal_aim_point: Option<Vector>,
    needed_jump_time: f32,
    jumping: bool,
    jump_time: Option<f32>,
    quick: bool,
    counter: u8,
    patience_applied: bool,
    dodge_started_at: Option<f32>,
    dodge_direction: Vector,
    side_mode: bool,
    side_sign: f32,
    side_lateral_distance: f32,
    targets: ShotTargets,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DoubleJumpState {
    started_at: f32,
    intercept_time: f32,
    ball: Vector,
    offset_target: Vector,
    shot_direction: Vector,
    jump_time: f32,
    jump_frames: u32,
    second_jump_sent: bool,
    prev_jump: bool,
    targets: ShotTargets,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct KickoffDodgeState {
    started_at: f32,
    direction: Vector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SpeedFlipState {
    step_start: f32,
    step_index: u8,
    direction: f32,
    use_boost: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HalfFlipState {
    started_at: f32,
    release_frames: u8,
    flip_started_at: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct EmergencyClearState {
    started_at: f32,
    target: Option<Vector>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PowerClearState {
    started_at: f32,
    phase: u8,
    initial_ball_velocity: Vector,
    initial_ball_z: f32,
    contact_time: Option<f32>,
}

#[derive(Debug, Default)]
pub struct BotBrain {
    kickoff_complete: bool,
    pub match_state: MatchState,
    pub boost_accel: f32,
    pub unlimited_boost: bool,
    shot_targets: ShotTargets,
    pub boost_pads: Vec<BoostPadState>,
    last_kickoff_time: f32,
    kickoff_was_active: bool,
    kickoff_ball_start: Option<Vector>,
    kickoff_contest_until: f32,
    ground_flip: Option<FlipState>,
    jump_shot: Option<JumpShotState>,
    double_jump_shot: Option<DoubleJumpState>,
    pending_double_jump: Option<PendingDoubleJump>,
    ground_shot_patience_applied: bool,
    wave_dash: Option<crate::mechanics::WaveDash>,
    defense: GridDefense,
    defense_flip: Option<FlipState>,
    last_flip_time: f32,
    defense_recovery_target: Option<Vector>,
    ball_recovery: bool,
    half_flip: Option<HalfFlipState>,
    speed_flip: Option<SpeedFlipState>,
    navigation_speedflip_used: bool,
    kickoff_dodge: Option<KickoffDodgeState>,
    aerial_candidate: Option<AerialCandidate>,
    aerial_shot: Option<AerialShot>,
    wall_drive_candidate: Option<WallDrivePlan>,
    boost_target: Option<BoostPadState>,
    teammate_comms: HashMap<usize, TeamplayMessage>,
    teamplay_status: String,
    teamplay_score: f32,
    teamplay_contact_time: f32,
    teamplay_cover_released: bool,
    teamplay_cover_owner: Option<usize>,
    teamplay_receive: Option<(Vector, Vector, Vector)>,
    transition_attacker: Option<usize>,
    transition_contact_at: f32,
    transition_hit_time: Option<f32>,
    transition_last_ball: Option<BallState>,
    transition_followup_until: f32,
    transition_last_target: Option<Vector>,
    transition_direction: Option<Vector>,
    transition_bump_cooldown_until: f32,
    emergency_clear: Option<EmergencyClearState>,
    last_emergency_clear_time: f32,
    emergency_save_zone_active: bool,
    power_clear: Option<PowerClearState>,
    last_goalie_role_time: f32,
    was_airborne: bool,
    last_landing_time: f32,
}

fn shot_patience_adjustment(
    car: CarState,
    slices: &[BallSlice],
    game_time: f32,
    intercept_time: f32,
    goal_y: f32,
) -> Option<(f32, BallSlice)> {
    if slices.is_empty() {
        return None;
    }

    let current_time = (intercept_time - game_time).max(0.0);
    let start = ((current_time * 120.0).round() as isize - 1)
        .clamp(0, slices.len() as isize - 1) as usize;
    if start >= slices.len() - 1 {
        return None;
    }

    let current = slices[start];
    let current_pos = current.location;
    let current_vel = current.velocity;
    let sign_x = if current_pos.x < 0.0 {
        -1.0
    } else if current_pos.x > 0.0 {
        1.0
    } else {
        0.0
    };
    let alignment_wait = current_pos.x.abs() < 3000.0
        && current_pos.x.abs() - 890.0 + 92.75 * 2.0_f32.sqrt()
            > (current_pos.y - goal_y).abs()
        && current_vel.x * sign_x < 0.0;
    let falling_wait = current_pos.z > 180.0 && current_vel.z < -80.0;
    if !alignment_wait && !falling_wait {
        return None;
    }

    let speed = car.velocity.flatten().magnitude();
    let estimated_speed = (speed + 650.0).clamp(1000.0, 2300.0);
    // Python's range(start + 1, max_index + 1) includes max_index.
    // Preserve that inclusive upper bound at prediction slice 600.
    let end_index = slices.len().min(601);
    for (i, future) in slices.iter().copied().enumerate().take(end_index).skip(start + 1) {
        let future_time = (i + 1) as f32 / 120.0;
        if future_time <= current_time + 0.15 {
            continue;
        }
        let pos = future.location;
        let vel = future.velocity;
        let distance = car.location.flat_distance(pos);
        if distance / estimated_speed > future_time + 0.15 {
            continue;
        }

        let aligned = alignment_wait
            && pos.x.abs() - 890.0 + 92.75 * 2.0_f32.sqrt()
                <= (pos.y - goal_y).abs();
        let fallen = falling_wait && pos.z <= 180.0 && vel.z <= 80.0;
        if aligned || fallen {
            return Some((game_time + future_time, future));
        }
    }

    None
}

impl BotBrain {
    pub fn new() -> Self {
        Self {
            kickoff_complete: false,
            match_state: MatchState::default(),
            boost_accel: BOOST_ACCEL,
            unlimited_boost: false,
            shot_targets: ShotTargets::default(),
            boost_pads: Vec::new(),
            last_kickoff_time: -999.0,
            kickoff_was_active: false,
            kickoff_ball_start: None,
            kickoff_contest_until: -999.0,
            ground_flip: None,
            jump_shot: None,
            double_jump_shot: None,
            pending_double_jump: None,
            ground_shot_patience_applied: false,
            wave_dash: None,
            defense: GridDefense::default(),
            defense_flip: None,
            last_flip_time: -999.0,
            defense_recovery_target: None,
            ball_recovery: false,
            half_flip: None,
            speed_flip: None,
            navigation_speedflip_used: false,
            kickoff_dodge: None,
            aerial_candidate: None,
            aerial_shot: None,
            wall_drive_candidate: None,
            boost_target: None,
            teammate_comms: HashMap::new(),
            teamplay_status: "NO_SHOT".to_owned(),
            teamplay_score: 0.0,
            teamplay_contact_time: 10.0,
            teamplay_cover_released: false,
            teamplay_cover_owner: None,
            teamplay_receive: None,
            transition_attacker: None,
            transition_contact_at: -999.0,
            transition_hit_time: None,
            transition_last_ball: None,
            transition_followup_until: -999.0,
            transition_last_target: None,
            transition_direction: None,
            transition_bump_cooldown_until: -999.0,
            emergency_clear: None,
            last_emergency_clear_time: -999.0,
            emergency_save_zone_active: false,
            power_clear: None,
            last_goalie_role_time: -999.0,
            was_airborne: false,
            last_landing_time: -999.0,
        }
    }

    pub fn set_unlimited_boost(&mut self, unlimited: bool) {
        self.unlimited_boost = unlimited;
    }

    pub fn set_teammate_comms(&mut self, messages: &HashMap<usize, TeamplayMessage>) {
        self.teammate_comms.clone_from(messages);
    }

 
    fn set_teamplay_shot(
        &mut self,
        me: CarState,
        ball: BallState,
        candidate: ShotCandidate,
        goal: Vector,
        roles: Option<TeamRoles>,
        urgent_save: bool,
    ) {
        // Python find_teamplay_opportunity scores the live ball, not the
        // future contact slice. The future slice is used only for the receive
        // plan sent to the transition player.
        let distance = me.location.distance(ball.location);
        let local_ball = me.local_location(ball.location);
        let facing_angle = local_ball.y.atan2(local_ball.x).abs();
        let approach = (ball.location - me.location).flatten().normalize();
        let receive_aim = if (candidate.parser.targets.left - candidate.parser.targets.right).magnitude() > 1.0 {
            (candidate.parser.targets.left + candidate.parser.targets.right) * 0.5
        } else { goal };
        let goal_direction = (goal - ball.location).flatten().normalize();
        let alignment = if approach.magnitude() > 0.001 && goal_direction.magnitude() > 0.001 {
            approach.dot(goal_direction).clamp(-1.0, 1.0)
        } else { 1.0 };
        let shooting_penalty = (1.0 - alignment) * 14.0;
        let shot_type = if candidate.targeted { "SHOT_FOUND" } else { "ANY_SHOT_FOUND" };
        let mut score = if candidate.targeted { 78.0 } else { 64.0 };
        score -= candidate.time.min(4.0) * 7.0;
        score -= facing_angle.min(3.14) * 7.0;
        score -= (distance / 3000.0).min(1.0) * 10.0;
        score -= if candidate.targeted { shooting_penalty } else { shooting_penalty * 0.35 };
        if me.boost < 25.0 { score -= 4.0; }
        if urgent_save { score += 15.0; }
        self.teamplay_status = shot_type.to_owned();
        self.teamplay_score = score.clamp(0.0, 100.0);
        self.teamplay_contact_time = candidate.time.clamp(0.0, 10.0);
        self.teamplay_cover_released = true;
        self.teamplay_cover_owner = roles.and_then(|r| if r.goalie != r.challenger { Some(r.goalie) } else { None });
        let _ = ball;
        self.teamplay_receive = receive_plan(me, candidate.ball, receive_aim);
    }


    fn set_teamplay_aerial(
        &mut self,
        me: CarState,
        ball: BallState,
        candidate: AerialCandidate,
        goal: Vector,
        aim: Vector,
        roles: Option<TeamRoles>,
        urgent_save: bool,
        targeted: bool,
    ) {
        let distance = me.location.distance(ball.location);
        let local_ball = me.local_location(ball.location);
        let facing = local_ball.y.atan2(local_ball.x).abs();
        let approach = (ball.location - me.location).flatten().normalize();
        let goal_direction = (goal - ball.location).flatten().normalize();
        let alignment = if approach.magnitude() > 0.001 && goal_direction.magnitude() > 0.001 {
            approach.dot(goal_direction).clamp(-1.0, 1.0)
        } else { 1.0 };
        let shot_status = if targeted { "SHOT_FOUND" } else { "ANY_SHOT_FOUND" };
        let mut score = if targeted { 78.0 } else { 64.0 }
            - candidate.slice.time.min(4.0) * 7.0
            - facing.min(3.14) * 7.0 - (distance / 3000.0).min(1.0) * 10.0
            - (1.0 - alignment) * 14.0 * if targeted { 1.0 } else { 0.35 };
        if me.boost < 25.0 { score -= 4.0; }
        if urgent_save { score += 15.0; }
        self.teamplay_status = shot_status.to_owned();
        self.teamplay_score = score.clamp(0.0, 100.0);
        self.teamplay_contact_time = candidate.slice.time.clamp(0.0, 10.0);
        self.teamplay_cover_released = true;
        self.teamplay_cover_owner = roles.and_then(|r| if r.goalie != r.challenger { Some(r.goalie) } else { None });
        self.teamplay_receive = receive_plan(me, candidate.slice.location, goal);
    }

    fn yield_teamplay_opportunity(&mut self) {
        self.teamplay_status = "NO_SHOT".to_owned();
        self.teamplay_score = 0.0;
        self.teamplay_contact_time = 10.0;
        self.teamplay_cover_released = false;
        self.teamplay_cover_owner = None;
        self.teamplay_receive = None;
        self.aerial_candidate = None;
    }

    pub fn teamplay_message(&self, now: f32) -> TeamplayMessage {
        let (receive, direction, contact) = self.teamplay_receive
            .map(|(receive, direction, contact)| (Some(receive), Some(direction), Some(contact)))
            .unwrap_or((None, None, None));
        TeamplayMessage {
            status: self.teamplay_status.clone(),
            shot_type: if matches!(self.teamplay_status.as_str(), "SHOT_FOUND" | "ANY_SHOT_FOUND" | "WALL_PASS") {
                self.teamplay_status.clone()
            } else { "NO_SHOT".to_owned() },
            score: self.teamplay_score,
            contact_time: self.teamplay_contact_time,
            received_at: now,
            cover_owner: self.teamplay_cover_owner,
            cover_released: self.teamplay_cover_released,
            receive_x: receive.map(|v| v.x),
            receive_y: receive.map(|v| v.y),
            receive_dir_x: direction.map(|v| v.x),
            receive_dir_y: direction.map(|v| v.y),
            contact_x: contact.map(|v| v.x),
            contact_y: contact.map(|v| v.y),
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
        shot_targets: ShotTargets,
    ) -> ControllerState {
        self.match_state = match_state;
        self.shot_targets = shot_targets;
        if self.was_airborne && !me.airborne {
            self.last_landing_time = game_time;
        }
        self.was_airborne = me.airborne;
        if let Some(mut dash) = self.wave_dash {
            let result = dash.tick(me, game_time);
            if result.finished {
                self.wave_dash = None;
                if result.recover {
                    let target = result.recovery_target.unwrap_or_else(|| {
                        me.location + me.velocity.normalize()
                    });
                    self.defense_recovery_target = Some(target);
                    return GridDefense::recovery_controller(
                        me, target, match_state.gravity.z,
                    );
                }
            } else {
                self.wave_dash = Some(dash);
            }
            return result.controller;
        }
        if let Some(state) = self.kickoff_dodge {
            return self.run_kickoff_dodge(state, game_time);
        }
        if let Some(state) = self.speed_flip {
            return self.run_speed_flip(me, state, game_time);
        }
        if let Some(state) = self.half_flip {
            return self.run_half_flip(me, state, game_time, match_state.gravity.z);
        }

        let side = team_side(me.team);
        let own_goal = Vector::new(0.0, side * 5120.0, 0.0);
        let ball_in_goal_danger = ball.location.flat_distance(own_goal) < 400.0
            && ball.velocity.y * side > 250.0;
        if ball_in_goal_danger {
            if !self.emergency_save_zone_active {
                self.emergency_save_zone_active = true;
                if self.aerial_shot.as_ref().is_none_or(|shot| !shot.aerialing)
                    && self.ground_flip.is_none()
                    && self.defense_flip.is_none()
                    && self.defense_recovery_target.is_none()
                {
                    self.jump_shot = None;
                    self.double_jump_shot = None;
                    self.aerial_shot = None;
                    self.power_clear = None;
                    self.last_emergency_clear_time = game_time;
                    self.emergency_clear = Some(EmergencyClearState {
                        started_at: game_time,
                        target: None,
                    });
                }
            }
        } else {
            self.emergency_save_zone_active = false;
        }

        if let Some(state) = self.emergency_clear {
            return self.run_emergency_clear(me, ball, game_time, state);
        }
        if let Some(state) = self.power_clear {
            return self.run_power_clear(me, cars, ball, prediction, game_time, state);
        }

        self.boost_pads.clear();
        self.boost_pads.extend_from_slice(boost_pads);
        let mut plan = self.choose_plan(me, cars, ball, prediction, game_time, match_state, shot_targets);

        // Python evaluates boost collection before its rolling-ball clear.
        // If no boost route was selected, preserve that exact priority order.
        if plan.decision != Decision::Boost
            && self.jump_shot.is_none()
            && self.double_jump_shot.is_none()
            && self.pending_double_jump.is_none()
            && self.aerial_shot.is_none()
            && self.ground_flip.is_none()
            && self.defense_flip.is_none()
            && self.defense_recovery_target.is_none()
            && self.ball_rolling_toward_me(me, cars, ball, prediction, game_time)
        {
            let state = PowerClearState {
                started_at: game_time,
                phase: 1,
                initial_ball_velocity: ball.velocity,
                initial_ball_z: ball.location.z,
                contact_time: None,
            };
            self.power_clear = Some(state);
            return self.run_power_clear(me, cars, ball, prediction, game_time, state);
        }

        if let Some(mut shot) = plan.shot {
            if !self.ground_shot_patience_applied {
                let foe_goal_y = -side * 5120.0;
                if let Some((intercept_time, later_slice)) = shot_patience_adjustment(
                    me,
                    &prediction.slices,
                    game_time,
                    game_time + shot.time,
                    foe_goal_y,
                ) {
                    shot.time = intercept_time - game_time;
                    shot.ball = later_slice.location;
                    shot.ball_velocity = later_slice.velocity;
                    let aim = relative_goal_aim_point(me, shot.ball, shot.targets);
                    shot.shot_direction = (aim - shot.ball).flatten().normalize();
                    shot.approach = contact_approach_from_label(
                        shot.ball, shot.shot_direction, shot.contact_face, 92.75,
                    ).unwrap_or(shot.ball - shot.shot_direction * 92.75);
                    plan.target = shot.approach;
                    plan.shot = Some(shot);
                    self.ground_shot_patience_applied = true;
                }
            }
        } else {
            self.ground_shot_patience_applied = false;
        }

        // A newly reachable wall approach replaces ordinary ground/jump
        // setups, but never a launched aerial, flip, or recovery.

        // Aerial aborts in the Python routine stack push ball_recovery,
        // which is distinct from ordinary recovery because it may boost
        // downward toward a predicted floor landing.
        if self.ball_recovery {
            let controller = GridDefense::ball_recovery_controller(
                me, match_state.gravity.z,
            );
            if !me.airborne {
                self.ball_recovery = false;
                self.defense_recovery_target = None;
            }
            return controller;
        }

        // VirxERLU routines are stack-prioritized: a recovery pushed by
        // grid_defense remains active until the car regains wheel contact,
        // even if the strategic role changes during flight.
        if let Some(target) = self.defense_recovery_target {
            if me.airborne {
                return GridDefense::recovery_controller(me, target, match_state.gravity.z);
            }
            self.defense_recovery_target = None;
        }

        if self.defense_flip.is_some() {
            return self.run_grid_defense(me, cars, ball, game_time, plan.decision);
        }

        // A launched aerial is a committed, uninterruptible mechanic. Keep
        // executing its own prediction/boost controller until contact or a
        // physically impossible recovery, just like the source routine stack.
        if let Some(mut pending) = self.pending_double_jump {
            let time_remaining = pending.intercept_time - game_time;
            if time_remaining <= 0.0 || prediction.slices.is_empty() {
                self.pending_double_jump = None;
            } else {
                let slice_index = ((time_remaining * 120.0).round() as isize - 1)
                    .clamp(0, prediction.slices.len() as isize - 1) as usize;
                let slice = prediction.slices[slice_index];
                pending.shot.time = time_remaining;
                pending.shot.ball = slice.location;
                pending.shot.ball_velocity = slice.velocity;
                let controller = self.drive_double_jump(me, ball, pending.shot, prediction, game_time);
                if self.double_jump_shot.is_some() {
                    self.pending_double_jump = None;
                } else {
                    self.pending_double_jump = Some(pending);
                }
                return controller;
            }
        }

        if let Some(state) = self.double_jump_shot {
            return self.run_double_jump(me, state, game_time);
        }

        if let Some(mut aerial) = self.aerial_shot {
            let aerial_goal = relative_goal_aim_point(me, ball.location, self.shot_targets);
            aerial.refresh_prediction(
                me, prediction, game_time, aerial_goal, match_state.gravity, ball.location.z,
            );
            let drive = default_drive(
                &me,
                DriveInput {
                    target_speed: MAX_SPEED,
                    local_target: me.local_location(aerial.target_location),
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel: self.boost_accel,
                },
            );
            if aerial.fast_aerial
                && (aerial.ball.z <= 350.0 || ball.location.z <= 350.0)
            {
                aerial.fast_aerial = false;
                aerial.double_jump = false;
            }
            let controls = aerial.run(
                me, game_time, match_state.gravity, self.boost_accel,
                drive.controls.steer,
            );
            if aerial.finished {
                self.aerial_shot = None;
                if me.airborne {
                    self.defense_recovery_target = None;
                    self.ball_recovery = true;
                    return GridDefense::ball_recovery_controller(
                        me, match_state.gravity.z,
                    );
                }
                self.ball_recovery = false;
                return controller_from_drive(drive, me);
            }
            self.aerial_shot = Some(aerial);
            if !aerial.aerialing && !aerial.side_wall_active {
                return controller_from_drive(drive, me);
            }
            return controls;
        }

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
                    state.ball_velocity = candidate.ball_velocity;
                    state.shot_direction = candidate.shot_direction;
                    if state.side_mode {
                        let lateral = Vector::new(-state.shot_direction.y, state.shot_direction.x, 0.0).normalize();
                        state.approach = candidate.ball - lateral * state.side_lateral_distance * state.side_sign;
                    } else {
                        state.approach = candidate.approach;
                    }
                    self.jump_shot = Some(state);
                }
            }
        }

        if let Some(mut state) = self.jump_shot {
            if !state.patience_applied {
                let foe_goal_y = -team_side(me.team) * 5120.0;
                if let Some((new_intercept_time, later_slice)) = shot_patience_adjustment(
                    me,
                    &prediction.slices,
                    game_time,
                    state.intercept_time,
                    foe_goal_y,
                ) {
                    state.intercept_time = new_intercept_time;
                    state.ball = later_slice.location;
                    state.ball_velocity = later_slice.velocity;
                    state.patience_applied = true;
                }
                self.jump_shot = Some(state);
            }
        }

        // A Python jump_shot is a committed routine-stack entry. Once it
        // exists, new defensive/transition plans cannot steal its controller.
        if let Some(state) = self.jump_shot {
            return self.run_jump_shot(me, ball, prediction, state, game_time);
        }

        if self.jump_shot.is_none()
            && matches!(plan.decision, Decision::Goalkeeper | Decision::Transition)
        {
            return self.run_grid_defense(me, cars, ball, game_time, plan.decision);
        }

        self.drive_to(me, plan, game_time, ball, prediction)
    }

    fn run_kickoff_dodge(
        &mut self,
        state: KickoffDodgeState,
        game_time: f32,
    ) -> ControllerState {
        let elapsed = game_time - state.started_at;
        let mut controller = ControllerState::default();
        if elapsed < 0.05 {
            controller.jump = true;
        } else if elapsed < 0.083 {
            controller.jump = false;
        } else if elapsed < 0.15 {
            controller.jump = true;
            controller.pitch = -state.direction.x;
            controller.yaw = state.direction.y;
        } else {
            self.kickoff_dodge = None;
            self.kickoff_complete = true;
        }
        controller
    }

    fn run_speed_flip(
        &mut self,
        car: CarState,
        mut state: SpeedFlipState,
        game_time: f32,
    ) -> ControllerState {
        const DURATIONS: [f32; 6] = [
            5.0 / 60.0,
            1.0 / 60.0,
            0.05,
            1.0 / 60.0,
            0.6,
            0.2,
        ];

        let mut controller = ControllerState::default();
        controller.throttle = 1.0;
        controller.boost = state.use_boost && car.boost > 0.0;

        match state.step_index {
            0 => controller.steer = -state.direction,
            1 => controller.jump = true,
            2 => {}
            3 => {
                controller.yaw = state.direction;
                controller.jump = true;
                controller.pitch = -1.0;
            }
            4 => controller.pitch = 1.0,
            5 => {
                controller.yaw = state.direction;
                controller.roll = state.direction;
            }
            _ => {
                self.speed_flip = None;
                return controller;
            }
        }

        if game_time - state.step_start > DURATIONS[state.step_index as usize] {
            state.step_index += 1;
            state.step_start = game_time;
            if state.step_index >= DURATIONS.len() as u8 {
                self.speed_flip = None;
                return controller;
            }
        }

        self.speed_flip = Some(state);
        controller
    }

    fn run_half_flip(
        &mut self,
        car: CarState,
        mut state: HalfFlipState,
        game_time: f32,
        gravity_z: f32,
    ) -> ControllerState {
        let elapsed = game_time - state.started_at;
        let mut controller = ControllerState::default();

        // Python half_flip: initial jump, three release frames, backwards
        // dodge, then cancel by reversing pitch.
        if elapsed < 0.075 {
            controller.jump = true;
            controller.pitch = 1.0;
            self.half_flip = Some(state);
            return controller;
        }

        if state.release_frames < 3 {
            controller.jump = false;
            state.release_frames += 1;
            self.half_flip = Some(state);
            return controller;
        }

        if elapsed < 0.20 {
            controller.jump = true;
            controller.pitch = 1.0;
            self.half_flip = Some(state);
            return controller;
        }

        if elapsed < 0.40 {
            if state.flip_started_at.is_none() {
                state.flip_started_at = Some(game_time);
                self.last_flip_time = game_time;
            }
            controller.jump = true;
            controller.pitch = -1.0;
            self.half_flip = Some(state);
            return controller;
        }

        self.half_flip = None;
        let target = if car.velocity.magnitude() > 1.0 {
            car.location + car.velocity.normalize() * 1000.0
        } else {
            car.location + car.forward() * 1000.0
        };
        if game_time - self.last_flip_time >= 0.5 && car.airborne {
            self.defense_recovery_target = Some(target);
            return GridDefense::recovery_controller(car, target, gravity_z);
        }
        controller
    }

    fn run_grid_defense(
        &mut self,
        me: CarState,
        cars: &[CarState],
        ball: BallState,
        game_time: f32,
        decision: Decision,
    ) -> ControllerState {
        if let Some(flip) = self.defense_flip {
            let elapsed = game_time - flip.started_at;
            let mut controller = ControllerState::default();
            if elapsed < 0.075 {
                controller.jump = true;
                return controller;
            }
            if elapsed < 0.10 {
                controller.jump = false;
                return controller;
            }
            if elapsed < 0.9 {
                if self.last_flip_time < flip.started_at {
                    self.last_flip_time = game_time;
                }
                controller.jump = true;
                controller.pitch = -flip.direction.x;
                controller.yaw = flip.direction.y;
                return controller;
            }
            self.defense_flip = None;
        }

        let output = self.defense.run(
            me,
            cars,
            ball,
            &self.boost_pads,
            Vector::new(0.0, -team_side(me.team) * 5120.0, 300.0),
            decision == Decision::Goalkeeper,
            false,
        );
        if output.start_flip {
            let flip = FlipState {
                started_at: game_time,
                direction: Vector::new(1.0, 0.0, 0.0),
                cancel: false,
            };
            self.defense_flip = Some(flip);
            let mut controller = ControllerState::default();
            controller.jump = true;
            return controller;
        }

        // Python grid_defense pushes recovery after the flip lockout while
        // descending. Its landing plane is calculated with the original
        // VirxERLU-CLib equations in defense.rs.
        if me.airborne
            && game_time - self.last_flip_time > 78.0 / 120.0
            && me.velocity.z < -10.0
        {
            if let Some(target) = output.recovery_target {
                self.defense_recovery_target = Some(target);
                return GridDefense::recovery_controller(me, target, self.match_state.gravity.z);
            }
        }
        output.controller
    }

    fn shot_search_horizon(&self, game_time: f32) -> f32 {
        let intercept = self.jump_shot.map(|s| s.intercept_time)
            .or_else(|| self.double_jump_shot.map(|s| s.intercept_time))
            .or_else(|| self.aerial_shot.map(|s| s.intercept_time));
        if let Some(intercept) = intercept {
            let remaining = intercept - game_time;
            if (0.0..0.5).contains(&remaining) { return 0.0; }
            if remaining > 0.0 { return 6.0_f32.min(remaining - 0.05); }
        }
        6.0
    }

    fn fast_intercept_plan(
        &mut self,
        me: CarState,
        cars: &[CarState],
        ball: BallState,
        prediction: &BallTrajectory,
        game_time: f32,
        opponent_goal: Vector,
        roles: Option<TeamRoles>,
        urgent_save: bool,
    ) -> Option<Plan> {
        let can_aerial = me.airborne || game_time - self.last_landing_time > 0.5;
        let found = find_fast_intercept_candidate(
            me,
            prediction,
            ball.location,
            self.match_state.gravity,
            self.boost_accel,
            0.0,
            self.shot_search_horizon(game_time),
            self.unlimited_boost,
            game_time < self.kickoff_contest_until,
            can_aerial,
        )?;
        let slice = found.slice;
        let parsed = found.parsed;
        let shot_type = parsed.shot_type?;

        if shot_type == ShotType::Aerial {
            let aim = relative_goal_aim_point(me, ball.location, self.shot_targets);
            let candidate = AerialCandidate {
                slice,
                shot_target: aim,
                target_location: parsed.approach_target,
                shot_direction: parsed.shot_vector,
                double_jump: parsed.fast,
                fast: parsed.fast,
                estimated_boost: 0.0,
                score: slice.time,
            };
            self.set_teamplay_aerial(
                me, ball, candidate, opponent_goal, opponent_goal, roles, urgent_save, false,
            );
            if !should_commit_to_shot(
                me, cars, ball, roles, urgent_save, self.teamplay_score,
                self.teamplay_contact_time, &self.teammate_comms, game_time,
            ) {
                self.yield_teamplay_opportunity();
                return None;
            }
            self.aerial_candidate = Some(candidate);
            return Some(Plan {
                decision: Decision::AerialChallenge,
                target: candidate.target_location,
                target_speed: MAX_SPEED,
                shot: None,
                jump: None,
            });
        }

        let candidate = ShotCandidate {
            time: slice.time,
            ball: slice.location,
            ball_velocity: slice.velocity,
            approach: parsed.approach_target,
            shot_direction: parsed.shot_vector,
            shot_type,
            targeted: false,
            can_jump: shot_type != ShotType::Ground,
            fast: parsed.fast,
            score: slice.time,
            parser: parsed,
        };
        self.set_teamplay_shot(me, ball, candidate, opponent_goal, roles, urgent_save);
        if !should_commit_to_shot(
            me, cars, ball, roles, urgent_save, self.teamplay_score,
            self.teamplay_contact_time, &self.teammate_comms, game_time,
        ) {
            self.yield_teamplay_opportunity();
            return None;
        }
        let intercept = candidate_to_intercept(me, candidate, self.shot_targets);
        let is_ground = shot_type == ShotType::Ground;
        Some(Plan {
            decision: Decision::GroundChallenge,
            target: intercept.approach,
            target_speed: MAX_SPEED,
            shot: if is_ground { Some(intercept) } else { None },
            jump: if is_ground { None } else { Some(intercept) },
        })
    }

    fn choose_plan(
        &mut self,
        me: CarState,
        cars: &[CarState],
        ball: BallState,
        prediction: &BallTrajectory,
        game_time: f32,
        match_state: MatchState,
        shot_targets: ShotTargets,
    ) -> Plan {
        let committed_nonwall_shot_active = self.jump_shot.is_some()
            || self.double_jump_shot.is_some()
            || self.pending_double_jump.is_some()
            || self.aerial_shot.is_some();
        let committed_shot_active =
            committed_nonwall_shot_active || self.wall_drive_candidate.is_some();
        self.aerial_candidate = None;
        self.wall_drive_candidate = None;
        self.boost_target = None;
        if committed_shot_active {
            // Python update_teamplay keeps cover released while a shot
            // routine remains on the stack, even when the finder temporarily
            // returns no replacement candidate.
            self.teamplay_cover_released = true;
        } else {
            self.teamplay_status = if me.boost < 18.0 { "LOW_BOOST" } else { "NO_SHOT" }.to_owned();
            self.teamplay_score = 0.0;
            self.teamplay_contact_time = 10.0;
            self.teamplay_cover_released = false;
            self.teamplay_cover_owner = None;
            self.teamplay_receive = None;
        }
        let side = team_side(me.team);
        let opponents: Vec<CarState> = cars
            .iter()
            .copied()
            .filter(|car| car.team != me.team)
            .collect();
        // Use RLBot's match phase, not ball-position heuristics: kickoff
        // remains authoritative during kickoff countdown/settling frames.
        let kickoff_active = match_state.round_active && match_state.kickoff;

        if kickoff_active && !self.kickoff_was_active {
            self.kickoff_complete = false;
            self.kickoff_ball_start = Some(ball.location);
            self.kickoff_contest_until = -999.0;
        } else if !kickoff_active {
            self.kickoff_complete = true;
        }
        self.kickoff_was_active = kickoff_active;

        if let Some(start) = self.kickoff_ball_start {
            let resolved = ball.location.distance(start) > 300.0
                || ball.velocity.magnitude() > 350.0;
            if resolved && self.kickoff_contest_until < -900.0 {
                self.kickoff_contest_until = game_time + 10.0;
            }
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

            let target = kickoff_boost_target(me, side, &self.boost_pads)
                .unwrap_or(me.location.flatten());
            return Plan {
                decision: Decision::KickoffBackBoost,
                target,
                target_speed: if target == me.location.flatten() { 0.0 } else { MAX_SPEED },
                shot: None,
                jump: None,
            };
        }

        let living_teammates: Vec<CarState> = cars
            .iter()
            .copied()
            .filter(|car| car.team == me.team && !car.demolished)
            .collect();
        let own_goal = Vector::new(0.0, side * 5120.0, 350.0);
        let opponent_goal = Vector::new(0.0, -side * 5120.0, 300.0);
        let roles = if living_teammates.len() >= 2 {
            assign_roles_volt(
                &living_teammates,
                ball,
                side,
                own_goal,
                opponent_goal,
                prediction,
                game_time,
            )
        } else {
            None
        };

        let ball_depth = ball.location.y * side;
        let to_own_goal = (own_goal - ball.location).flatten();
        let toward_goal_speed = if to_own_goal.magnitude() > 1.0 {
            ball.velocity.flatten().dot(to_own_goal.normalize())
        } else {
            0.0
        };
        let extreme_danger = ball_depth > 4200.0 && ball.location.z < 350.0;
        let low_ball_threat = ball_depth > 1200.0
            && ball.location.z < 350.0
            && ball.velocity.y * side > 250.0;
        let urgent_save = extreme_danger || low_ball_threat;

        // The Python bot elects the challenger and deepest defender from
        // the same ball-dependency scores in every team size. Do not use a
        // separate 2v2 distance heuristic.
        let transition_followup_active = self.transition_followup_until > game_time;
        let is_challenger = transition_followup_active || if living_teammates.len() <= 1 {
            true
        } else {
            roles.map(|role| role.challenger == me.index).unwrap_or(false)
        };
        let is_goalie = roles.map(|r| r.goalie == me.index).unwrap_or(false);
        self.teamplay_cover_owner = roles.and_then(|r| {
            if r.goalie != r.challenger { Some(r.goalie) } else { None }
        });

        if urgent_save {
            self.teamplay_status = "DEFEND".to_owned();
            self.teamplay_score = 100.0;
            self.teamplay_contact_time = if is_challenger {
                (me.location.distance(ball.location) / (me.velocity.magnitude() + 1000.0).max(900.0))
                    .clamp(0.12, 2.0)
            } else { 10.0 };
            self.teamplay_cover_released = true;
        }

        // Port the Python boost decisions before ordinary shot search.
        // Only an empty routine stack may divert to a pad.
        let routine_stack_clear = self.jump_shot.is_none()
            && self.double_jump_shot.is_none()
            && self.aerial_shot.is_none()
            && self.ground_flip.is_none()
            && self.defense_flip.is_none()
            && self.defense_recovery_target.is_none()
            && !self.ball_recovery
            && self.wave_dash.is_none()
            && self.half_flip.is_none()
            && self.speed_flip.is_none()
            && self.kickoff_dodge.is_none()
            && self.emergency_clear.is_none()
            && self.power_clear.is_none();

        let friends_count = living_teammates.len().saturating_sub(1);
        let goalie_holding_net = friends_count > 0 && is_goalie;
        if goalie_holding_net {
            self.last_goalie_role_time = game_time;
        }
        let recently_released_goalie_under_threat = !goalie_holding_net
            && game_time - self.last_goalie_role_time <= 5.0
            && game_time >= self.last_goalie_role_time
            && ball_depth > 0.0
            && ball.velocity.y * side > 150.0;
        let cover_player_deep_defending =
            friends_count > 0 && goalie_holding_net && ball_depth > 2048.0;
        let defensive_boost_lock = goalie_holding_net
            || cover_player_deep_defending
            || recently_released_goalie_under_threat;
        let prioritize_ball = should_prioritize_ball(
            me,
            cars,
            ball,
            side,
            game_time,
            self.kickoff_contest_until,
            urgent_save,
            false,
            roles,
        );

        if routine_stack_clear
            && me.boost < 70.0
            && !defensive_boost_lock
            && !urgent_save
            && !(friends_count >= 2 && goalie_holding_net && !cover_player_deep_defending)
            && (prioritize_ball
                || (me.boost < 55.0 && ball_depth > 640.0 && ball.location.x.abs() > 1200.0))
        {
            if let Some(pad) = boost_on_route_to_ball(me, ball, &self.boost_pads, &opponents, side) {
                self.boost_target = Some(pad);
                return Plan {
                    decision: Decision::Boost,
                    target: pad.location,
                    target_speed: MAX_SPEED,
                    shot: None,
                    jump: None,
                };
            }
        }

        let normal_boost_threshold = if friends_count == 1 {
            23.0
        } else if friends_count > 1 {
            30.0
        } else {
            18.0
        };
        if routine_stack_clear
            && me.boost < normal_boost_threshold
            && !defensive_boost_lock
            && !prioritize_ball
            && !cover_player_deep_defending
        {
            if let Some(pad) = self
                .boost_pads
                .iter()
                .copied()
                .filter(|pad| pad.active && pad.large)
                .min_by(|a, b| {
                    me.location.distance(a.location)
                        .total_cmp(&me.location.distance(b.location))
                })
            {
                self.boost_target = Some(pad);
                return Plan {
                    decision: Decision::Boost,
                    target: pad.location,
                    target_speed: MAX_SPEED,
                    shot: None,
                    jump: None,
                };
            }
        }

        if urgent_save && is_challenger {
            let car_depth = me.location.y * side;
            let distance_to_ball = me.location.distance(ball.location);
            let clear_now = (ball_depth > 4200.0 && car_depth > ball_depth)
                || (game_time - self.last_emergency_clear_time >= 0.8
                    && (ball_depth > 2500.0 || distance_to_ball < 1100.0));
            if self.ground_flip.is_none()
                && self.defense_flip.is_none()
                && self.defense_recovery_target.is_none()
                && self.aerial_shot.as_ref().is_none_or(|shot| !shot.aerialing)
            {
                self.jump_shot = None;
                self.double_jump_shot = None;
                self.aerial_shot = None;
            }
            if clear_now {
                self.last_emergency_clear_time = game_time;
                return Plan {
                    decision: Decision::EmergencyChallenge,
                    target: select_contact_approach(me, ball.location, shot_aim_point(ball.location, side, me.airborne)),
                    target_speed: MAX_SPEED,
                    shot: None,
                    jump: None,
                };
            }
            return Plan {
                decision: if is_goalie { Decision::Goalkeeper } else { Decision::Transition },
                target: Vector::new(0.0, side * 5050.0, 0.0),
                target_speed: 1500.0,
                shot: None,
                jump: None,
            };
        }

        if is_goalie && !is_challenger {
            return Plan {
                decision: Decision::Goalkeeper,
                target: Vector::new(0.0, side * 5270.0, 0.0),
                target_speed: 1500.0,
                shot: None,
                jump: None,
            };
        }

        // In 2v2, should_prioritize_ball() can elect the teammate with
        // the better immediate challenge. The non-challenger runs the
        // non-goalie grid_defense lane rather than also driving at the ball.
        if living_teammates.len() == 2 && !is_challenger {
            return Plan {
                decision: Decision::Transition,
                target: Vector::new(
                    if me.location.x == 0.0 { 0.0 } else { me.location.x.signum() * 900.0 },
                    side * 5050.0,
                    0.0,
                ),
                target_speed: 1700.0,
                shot: None,
                jump: None,
            };
        }

        if roles.and_then(|r| r.transition).is_some_and(|index| index == me.index)
            && !transition_followup_active
        {
            let attacker = roles.map(|r| r.challenger);
            let attacker_message = attacker.and_then(|index| self.teammate_comms.get(&index));
            let active_attack = attacker_message.is_some_and(|message| {
                game_time - message.received_at <= 0.65
                    && message.cover_released
                    && matches!(message.status.as_str(), "SHOT_FOUND" | "ANY_SHOT_FOUND" | "WALL_PASS")
                    && (ball_depth < 1280.0 || message.status == "WALL_PASS")
            }) && !urgent_save;

            if active_attack {
                let attacker_index = attacker.unwrap();
                let attacker_car = cars.iter().copied().find(|car| car.index == attacker_index && !car.demolished);
                if self.transition_attacker != Some(attacker_index) {
                    self.transition_attacker = Some(attacker_index);
                    self.transition_hit_time = None;
                    self.transition_last_ball = Some(ball);
                    self.transition_contact_at = attacker_message.unwrap().received_at
                        + attacker_message.unwrap().contact_time;
                    self.transition_last_target = None;
                }
                if let Some(previous) = self.transition_last_ball {
                    let velocity_change = (ball.velocity - previous.velocity).magnitude();
                    let ball_shift = ball.location.flat_distance(previous.location);
                    if self.transition_hit_time.is_none()
                        && (velocity_change > 450.0
                            || (game_time >= self.transition_contact_at + 0.12 && ball_shift > 100.0)
                            || game_time >= self.transition_contact_at + 0.28)
                    {
                        self.transition_hit_time = Some(game_time);
                    }
                }
                self.transition_last_ball = Some(ball);

                if self.transition_hit_time.is_some_and(|hit| game_time >= hit + 0.18) {
                    self.transition_followup_until = game_time + 1.2;
                    self.transition_attacker = None;
                    self.transition_hit_time = None;
                    self.transition_last_ball = None;
                    self.transition_last_target = None;
                } else {
                    if game_time >= self.transition_bump_cooldown_until {
                        if let Some(attacker_car) = attacker_car {
                            if let Some(defender) = transition_bump_target(
                                me, attacker_car, ball, cars, opponent_goal,
                            ) {
                                self.transition_bump_cooldown_until = game_time + 2.25;
                                let distance = me.location.flat_distance(defender.location);
                                let speed = me.velocity.flatten().magnitude();
                                let lead_time = (distance / (speed + 650.0).max(1100.0)).clamp(0.12, 0.55);
                                return Plan {
                                    decision: Decision::TransitionBump,
                                    target: defender.location + defender.velocity.flatten() * lead_time,
                                    target_speed: MAX_SPEED,
                                    shot: None,
                                    jump: None,
                                };
                            }
                        }
                    }

                    let message = attacker_message.unwrap();
                    let direction = if self.transition_hit_time.is_some() {
                        let velocity = ball.velocity.flatten();
                        if velocity.magnitude() >= 250.0 { velocity.normalize() }
                        else { self.transition_last_ball.map(|b| b.velocity.flatten().normalize()).unwrap_or(Vector::new(0.0, -side, 0.0)) }
                    } else if let (Some(dx), Some(dy)) = (message.receive_dir_x, message.receive_dir_y) {
                        Vector::new(dx, dy, 0.0).normalize()
                    } else if let Some(attacker_car) = attacker_car {
                        attacker_car.forward().flatten().normalize()
                    } else {
                        ball.velocity.flatten().normalize()
                    };
                    let direction = if direction.magnitude() < 0.1 { Vector::new(0.0, -side, 0.0) } else { direction };
                    let mut target = if let (Some(x), Some(y)) = (message.receive_x, message.receive_y) {
                        Vector::new(x, y, 0.0)
                    } else {
                        ball.location + direction * 1450.0
                    };
                    target.x = target.x.clamp(-3900.0, 3900.0);
                    target.y = target.y.clamp(-4550.0, 4550.0);
                    if let Some(previous) = self.transition_last_target {
                        let shift = target - previous;
                        if shift.magnitude() < 1100.0 {
                            target = previous + shift * if self.transition_hit_time.is_some() { 0.58 } else { 0.38 };
                        }
                    }
                    self.transition_last_target = Some(target);
                    self.transition_direction = Some(direction);
                    return Plan {
                        decision: Decision::TransitionReceive,
                        target,
                        target_speed: 1700.0,
                        shot: None,
                        jump: None,
                    };
                }
            } else {
                self.transition_attacker = None;
                self.transition_hit_time = None;
                self.transition_last_ball = None;
                self.transition_last_target = None;
                self.transition_direction = None;
            }

            let x_sign = if me.location.x == 0.0 { 1.0 } else { me.location.x.signum() };
            self.transition_direction = None;
            return Plan {
                decision: Decision::Transition,
                target: Vector::new(x_sign * 900.0, side * 5050.0, 0.0),
                target_speed: 1700.0,
                shot: None,
                jump: None,
            };
        }

        // The Python run_logic() falls through to grid_defense() whenever
        // the elected challenger does not meet should_prioritize_ball().
        // Until the dedicated grid_defense controller is executed, preserve
        // its non-goalie lane target here rather than issuing a ball challenge.
        if !is_challenger {
            return Plan {
                decision: Decision::Transition,
                target: Vector::new(
                    if me.location.x == 0.0 { 0.0 } else { me.location.x.signum() * 900.0 },
                    side * 5050.0,
                    0.0,
                ),
                target_speed: 1700.0,
                shot: None,
                jump: None,
            };
        }

        if is_challenger {
            let aim = shot_aim_point(ball.location, side, me.airborne);
            let fast_intercept_zone = transition_followup_active || (ball_depth > 0.0 && ball_depth <= 1280.0);
            if fast_intercept_zone {
                if let Some(plan) = self.fast_intercept_plan(
                    me, cars, ball, prediction, game_time, opponent_goal, roles, urgent_save,
                ) {
                    return plan;
                }
                self.yield_teamplay_opportunity();
                return Plan {
                    decision: Decision::Transition,
                    target: Vector::new(
                        if me.location.x == 0.0 { 0.0 } else { me.location.x.signum() * 900.0 },
                        side * 5050.0,
                        0.0,
                    ),
                    target_speed: 1700.0,
                    shot: None,
                    jump: None,
                };
            }
            let mut candidate = if fast_intercept_zone {
                find_fast_intercept(
                    me, prediction, shot_targets, match_state.gravity, self.boost_accel,
                    if transition_followup_active { 0.03 } else { 0.10 },
                    self.shot_search_horizon(game_time), self.unlimited_boost,
                )
            } else {
                find_best_shot(
                    me, cars, ball.location, prediction, shot_targets,
                    match_state.gravity, self.boost_accel, game_time,
                    self.kickoff_contest_until, urgent_save, true, self.unlimited_boost, self.shot_search_horizon(game_time),
                ).or_else(|| {
                    if ball_depth < me.location.y * side {
                        find_best_any_shot(
                            me, cars, prediction, shot_targets, match_state.gravity,
                            self.boost_accel, game_time, self.kickoff_contest_until,
                            self.unlimited_boost, self.shot_search_horizon(game_time),
                        )
                    } else { None }
                })
            };
            if candidate.is_none() && !fast_intercept_zone {
                candidate = find_best_pass_shot(
                    me, cars, ball.location, prediction, match_state.gravity,
                    self.boost_accel, game_time, self.kickoff_contest_until,
                    urgent_save, true, self.unlimited_boost, self.shot_search_horizon(game_time),
                );
            }
            if candidate.is_none()
                && ball_depth > 640.0
                && opponent_is_vulnerable(me, ball, cars, own_goal)
            {
                candidate = find_best_any_shot(
                    me, cars, prediction, shot_targets, match_state.gravity,
                    self.boost_accel, game_time, self.kickoff_contest_until,
                    self.unlimited_boost, self.shot_search_horizon(game_time),
                );
            }
            if let Some(candidate) = candidate {
                self.set_teamplay_shot(me, ball, candidate, opponent_goal, roles, urgent_save);
                if !should_commit_to_shot(
                    me, cars, ball, roles, urgent_save, self.teamplay_score,
                    candidate.time, &self.teammate_comms, game_time,
                ) {
                    self.teamplay_status = if me.boost < 18.0 { "LOW_BOOST" } else { "NO_SHOT" }.to_owned();
                    self.teamplay_score = 0.0;
                    self.teamplay_contact_time = 10.0;
                    self.teamplay_cover_released = false;
                    self.teamplay_receive = None;
                    return Plan {
                        decision: Decision::Transition,
                        target: Vector::new(me.location.x.signum() * 900.0, side * 5050.0, 0.0),
                        target_speed: 1700.0,
                        shot: None,
                        jump: None,
                    };
                }
            }
            if candidate.is_none() && !urgent_save && self.ground_flip.is_none()
                && self.defense_flip.is_none() && self.defense_recovery_target.is_none()
            {
                let kickoff = game_time < self.kickoff_contest_until;
                let rejected_jump = self.jump_shot.is_some_and(|shot| {
                    opponent_reaches_before(
                        &opponents, shot.ball, shot.intercept_time - game_time,
                        ShotType::Jump, kickoff,
                    )
                });
                let rejected_double = self.double_jump_shot.is_some_and(|shot| {
                    opponent_reaches_before(
                        &opponents, shot.ball, shot.intercept_time - game_time,
                        ShotType::DoubleJump, kickoff,
                    )
                });
                if rejected_jump || rejected_double {
                    self.jump_shot = None;
                    self.double_jump_shot = None;
                    self.teamplay_status = "SHOT_ABANDONED".to_owned();
                    self.teamplay_score = 0.0;
                    self.teamplay_contact_time = 10.0;
                    self.teamplay_cover_released = false;
                    self.teamplay_receive = None;
                    return Plan {
                        decision: if is_goalie { Decision::Goalkeeper } else { Decision::Transition },
                        target: Vector::new(0.0, side * 5050.0, 0.0),
                        target_speed: 1500.0,
                        shot: None,
                        jump: None,
                    };
                }
            }
        let selected_score = candidate.map(|shot| shot.score);
        let selected = candidate.map(|candidate| candidate_to_intercept(me, candidate, shot_targets));
            let can_jump = candidate.is_some_and(|candidate| candidate.can_jump);
            let use_jump = selected.filter(|shot| shot.shot_type != ShotType::Ground || can_jump);
            let intercept = selected.filter(|shot| shot.shot_type == ShotType::Ground && !can_jump);

            // Noob Black's unfolded wall-shot scan competes with ordinary
            // ground/jump contacts, but never displaces an urgent save.
            if !committed_nonwall_shot_active
                && !urgent_save
                && self.ground_flip.is_none()
                && self.defense_flip.is_none()
                && self.defense_recovery_target.is_none()
            {
                if let Some(wall) = find_wall_drive(
                    &me, &prediction.slices, me.team, opponent_goal, 5.0,
                ) {

                    self.wall_drive_candidate = Some(wall);
                    self.teamplay_status = "WALL_PASS".to_owned();
                    self.teamplay_score = self.teamplay_score.max(78.0);
                    self.teamplay_contact_time = wall.intercept_time.clamp(0.0, 10.0);
                    self.teamplay_cover_released = true;
                    self.teamplay_receive = receive_plan(
                        me, wall.ball_slice.location, opponent_goal,
                    );
                    return Plan {
                        decision: Decision::WallDrive,
                        target: wall.ball_slice.location,
                        target_speed: MAX_SPEED,
                        shot: None,
                        jump: None,
                    };
                }
            }

            if fast_intercept_zone && !urgent_save
                && (me.airborne || game_time - self.last_landing_time > 0.5)
            {
                let max_height = if living_teammates.len() == 1 && opponents.len() == 1 {
                    1200.0
                } else { f32::INFINITY };
                if let Some(candidate) = find_fast_aerial_candidate(
                    me, prediction, aim, match_state.gravity, self.boost_accel,
                    if transition_followup_active { 0.03 } else { 0.10 },
                    self.shot_search_horizon(game_time), ball.location.z,
                    max_height, true,
                ) {
                    self.set_teamplay_aerial(me, ball, candidate, opponent_goal, aim, roles, urgent_save, false);
                    if should_commit_to_shot(
                        me, cars, ball, roles, urgent_save, self.teamplay_score,
                        self.teamplay_contact_time, &self.teammate_comms, game_time,
                    ) {
                        self.aerial_candidate = Some(candidate);
                        return Plan {
                            decision: Decision::AerialChallenge,
                            target: candidate.target_location,
                            target_speed: MAX_SPEED,
                            shot: None,
                            jump: None,
                        };
                    }
                    self.yield_teamplay_opportunity();
                }
            }

            // Aerials compete with ground and single-jump contacts only after
            // the same team-role gate has elected this car to challenge.
            // The CLib reachability check enforces the real boost/speed budget.
            let max_aerial_height = if living_teammates.len() == 1 && opponents.len() == 1 {
                1200.0
            } else {
                f32::INFINITY
            };
            let min_aerial_height = if max_aerial_height > 1200.0
                && me.location.z >= 2044.0 - me.hitbox.height * 1.1
            {
                551.0
            } else if self.unlimited_boost || me.airborne {
                150.0
            } else {
                450.0
            };
            if !urgent_save && self.jump_shot.is_none() && self.ground_flip.is_none() {
                if let Some(candidate) = find_aerial_candidate(
                    me,
                    prediction,
                    aim,
                    self.shot_targets,
                    cars,
                    match_state.gravity,
                    self.boost_accel,
                    game_time,
                    self.kickoff_contest_until,
                    min_aerial_height,
                    max_aerial_height,
                    self.shot_search_horizon(game_time),
                    self.unlimited_boost,
                ).filter(|candidate| {
                    selected_score.is_none_or(|best_score| candidate.score < best_score)
                }) {
                    self.set_teamplay_aerial(
                        me, ball, candidate, opponent_goal, aim, roles, urgent_save, true,
                    );
                    if should_commit_to_shot(
                        me, cars, ball, roles, urgent_save, self.teamplay_score,
                        self.teamplay_contact_time, &self.teammate_comms, game_time,
                    ) {
                        self.aerial_candidate = Some(candidate);
                        return Plan {
                            decision: Decision::AerialChallenge,
                            target: candidate.target_location,
                            target_speed: MAX_SPEED,
                            shot: None,
                            jump: None,
                        };
                    }
                    self.yield_teamplay_opportunity();
                }
            }

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
        let aim = shot_aim_point(ball.location, side, me.airborne);
        let fast_intercept_zone = ball_depth > 0.0 && ball_depth <= 1280.0;
        let mut candidate = if fast_intercept_zone {
            find_fast_intercept(
                me, prediction, shot_targets, match_state.gravity, self.boost_accel,
                0.10, self.shot_search_horizon(game_time), self.unlimited_boost,
            )
        } else {
            find_best_shot(
                me, cars, ball.location, prediction, shot_targets,
                match_state.gravity, self.boost_accel, game_time,
                self.kickoff_contest_until, urgent_save, true, self.unlimited_boost, self.shot_search_horizon(game_time),
            ).or_else(|| {
                if ball_depth < me.location.y * side {
                    find_best_any_shot(
                        me, cars, prediction, shot_targets, match_state.gravity,
                        self.boost_accel, game_time, self.kickoff_contest_until,
                        self.unlimited_boost, self.shot_search_horizon(game_time),
                    )
                } else { None }
            })
        };
        if candidate.is_none() && !fast_intercept_zone {
            candidate = find_best_pass_shot(
                me, cars, ball.location, prediction, match_state.gravity,
                self.boost_accel, game_time, self.kickoff_contest_until,
                urgent_save, true, self.unlimited_boost, self.shot_search_horizon(game_time),
            );
        }
        if candidate.is_none()
            && ball_depth > 640.0
            && opponent_is_vulnerable(me, ball, cars, own_goal)
        {
            candidate = find_best_any_shot(
                me, cars, prediction, shot_targets, match_state.gravity,
                self.boost_accel, game_time, self.kickoff_contest_until,
                self.unlimited_boost, self.shot_search_horizon(game_time),
            );
        }
        if let Some(candidate) = candidate {
            self.set_teamplay_shot(me, ball, candidate, opponent_goal, roles, urgent_save);
        }
        if candidate.is_none() && !urgent_save && self.ground_flip.is_none()
            && self.defense_flip.is_none() && self.defense_recovery_target.is_none()
        {
            let kickoff = game_time < self.kickoff_contest_until;
            let rejected_jump = self.jump_shot.is_some_and(|shot| {
                opponent_reaches_before(
                    &opponents, shot.ball, shot.intercept_time - game_time,
                    ShotType::Jump, kickoff,
                )
            });
            let rejected_double = self.double_jump_shot.is_some_and(|shot| {
                opponent_reaches_before(
                    &opponents, shot.ball, shot.intercept_time - game_time,
                    ShotType::DoubleJump, kickoff,
                )
            });
            if rejected_jump || rejected_double {
                self.jump_shot = None;
                self.double_jump_shot = None;
                return Plan {
                    decision: if is_goalie { Decision::Goalkeeper } else { Decision::Transition },
                    target: Vector::new(0.0, side * 5050.0, 0.0),
                    target_speed: 1500.0,
                    shot: None,
                    jump: None,
                };
            }
        }
        let selected_score = candidate.map(|shot| shot.score);
        let selected = candidate.map(|candidate| candidate_to_intercept(me, candidate, shot_targets));
        let can_jump = candidate.is_some_and(|candidate| candidate.can_jump);
        let use_jump = selected.filter(|shot| shot.shot_type != ShotType::Ground || can_jump);
        let intercept = selected.filter(|shot| shot.shot_type == ShotType::Ground && !can_jump);
        if self.ground_flip.is_none() && self.defense_flip.is_none()
            && self.defense_recovery_target.is_none()
            && self.aerial_shot.as_ref().is_none_or(|shot| !shot.aerialing) {
            if let Some(wall) = find_wall_drive(
                &me, &prediction.slices, me.team, opponent_goal, 5.0,
            ) {
                self.wall_drive_candidate = Some(wall);
                self.teamplay_status = "WALL_PASS".to_owned();
                self.teamplay_score = self.teamplay_score.max(78.0);
                self.teamplay_contact_time = wall.intercept_time.clamp(0.0, 10.0);
                self.teamplay_cover_released = true;
                self.teamplay_receive = receive_plan(me, wall.ball_slice.location, opponent_goal);
                return Plan {
                    decision: Decision::WallDrive,
                    target: wall.ball_slice.location,
                    target_speed: MAX_SPEED,
                    shot: None,
                    jump: None,
                };
            }
        }

        let max_aerial_height = if living_teammates.len() == 1 && opponents.len() == 1 {
            1200.0
        } else {
            f32::INFINITY
        };
        let min_aerial_height = if max_aerial_height > 1200.0
            && me.location.z >= 2044.0 - me.hitbox.height * 1.1
        {
            551.0
        } else if self.unlimited_boost || me.airborne {
            150.0
        } else {
            450.0
        };
        if !urgent_save && self.ground_flip.is_none() && self.jump_shot.is_none() {
            if fast_intercept_zone && (me.airborne || game_time - self.last_landing_time > 0.5) {
                if let Some(candidate) = find_fast_aerial_candidate(
                    me, prediction, aim, match_state.gravity, self.boost_accel,
                    0.10, self.shot_search_horizon(game_time), ball.location.z,
                    max_aerial_height, true,
                ) {
                    self.set_teamplay_aerial(me, ball, candidate, opponent_goal, aim, roles, urgent_save, false);
                    if should_commit_to_shot(
                        me, cars, ball, roles, urgent_save, self.teamplay_score,
                        self.teamplay_contact_time, &self.teammate_comms, game_time,
                    ) {
                        self.aerial_candidate = Some(candidate);
                        return Plan {
                            decision: Decision::AerialChallenge,
                            target: candidate.target_location,
                            target_speed: MAX_SPEED,
                            shot: None,
                            jump: None,
                        };
                    }
                    self.yield_teamplay_opportunity();
                }
            }
            if let Some(candidate) = find_aerial_candidate(
                me,
                prediction,
                aim,
                self.shot_targets,
                cars,
                match_state.gravity,
                self.boost_accel,
                game_time,
                self.kickoff_contest_until,
                min_aerial_height,
                max_aerial_height,
                self.shot_search_horizon(game_time),
                self.unlimited_boost,
            ).filter(|candidate| {
                selected_score.is_none_or(|best_score| candidate.score < best_score)
            }) {
                self.set_teamplay_aerial(me, ball, candidate, opponent_goal, aim, roles, urgent_save, true);
                if should_commit_to_shot(
                    me, cars, ball, roles, urgent_save, self.teamplay_score,
                    self.teamplay_contact_time, &self.teammate_comms, game_time,
                ) {
                    self.aerial_candidate = Some(candidate);
                    return Plan {
                        decision: Decision::AerialChallenge,
                        target: candidate.target_location,
                        target_speed: MAX_SPEED,
                        shot: None,
                        jump: None,
                    };
                }
                self.yield_teamplay_opportunity();
            }
        }
        if transition_followup_active && selected.is_none() {
            return Plan {
                decision: Decision::Transition,
                target: Vector::new(
                    if me.location.x == 0.0 { 0.0 } else { me.location.x.signum() * 900.0 },
                    side * 5050.0,
                    0.0,
                ),
                target_speed: 1700.0,
                shot: None,
                jump: None,
            };
        }
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



    fn ball_rolling_toward_me(
        &mut self,
        me: CarState,
        cars: &[CarState],
        ball: BallState,
        prediction: &BallTrajectory,
        game_time: f32,
    ) -> bool {
        let side = team_side(me.team);
        let teammates: Vec<CarState> = cars.iter().copied()
            .filter(|car| car.team == me.team && !car.demolished).collect();
        let own_goal = Vector::new(0.0, side * 5120.0, 350.0);
        let foe_goal = Vector::new(0.0, -side * 5120.0, 300.0);
        if teammates.len() > 1 {
            let roles = assign_roles_volt(
                &teammates, ball, side, own_goal, foe_goal, prediction, game_time,
            );
            if roles.is_some_and(|role| role.goalie == me.index) {
                self.last_goalie_role_time = game_time;
                return false;
            }
        }
        if game_time - self.last_goalie_role_time < 5.0 {
            return false;
        }
        if ball.location.z > 120.0 || me.location.flat_distance(ball.location) > 2500.0 {
            return false;
        }
        let ball_velocity = ball.velocity.flatten();
        if ball_velocity.magnitude() < 250.0 {
            return false;
        }
        let ball_to_car = (me.location - ball.location).flatten().normalize();
        ball_velocity.normalize().dot(ball_to_car) > 0.65
    }

    fn run_emergency_clear(
        &mut self,
        car: CarState,
        ball: BallState,
        game_time: f32,
        mut state: EmergencyClearState,
    ) -> ControllerState {
        if game_time - state.started_at > 2.0 {
            self.emergency_clear = None;
            return ControllerState::default();
        }
        let target = *state.target.get_or_insert_with(|| {
            Vector::new(
                if ball.location.x >= 0.0 { 3500.0 } else { -3500.0 },
                ball.location.y,
                0.0,
            )
        });
        let target = cap_in_field(&car, target);
        state.target = Some(target);
        self.emergency_clear = Some(state);

        let clear_direction = (target - ball.location).flatten().normalize();
        if clear_direction.magnitude() < 0.001 {
            return ControllerState::default();
        }
        let hit_target = ball.location - clear_direction * BALL_RADIUS;
        if car.airborne {
            self.defense_recovery_target = Some(hit_target);
            return GridDefense::recovery_controller(car, hit_target, self.match_state.gravity.z);
        }

        let local_target = car.local_location(hit_target);
        let local_ball = car.local_location(ball.location);
        let distance = car.location.flat_distance(hit_target);
        let angle = local_target.y.atan2(local_target.x).abs();
        let drive = default_drive(
            &car,
            DriveInput {
                target_speed: MAX_SPEED,
                local_target,
                upside_down: false,
                up: None,
                handbrake: false,
                boost_accel: self.boost_accel,
            },
        );
        let mut controller = controller_from_drive(drive, car);
        controller.throttle = 1.0;
        controller.boost = angle < 0.35 && distance > 500.0 && car.boost > 0.0;

        if distance < 700.0 && local_ball.y.abs() < 180.0 && local_ball.x > 0.0 {
            let flip = FlipState {
                started_at: game_time,
                direction: car.local(clear_direction).flatten().normalize(),
                cancel: false,
            };
            self.emergency_clear = None;
            self.ground_flip = Some(flip);
            return self.run_ground_flip(Some(car), game_time, flip);
        }
        controller
    }

    fn run_power_clear(
        &mut self,
        car: CarState,
        cars: &[CarState],
        ball: BallState,
        prediction: &BallTrajectory,
        game_time: f32,
        mut state: PowerClearState,
    ) -> ControllerState {
        let local_ball = car.local_location(ball.location);
        let distance = car.location.flat_distance(ball.location);
        if state.phase == 1 {
            if car.airborne || game_time - state.started_at > 1.3
                || (ball.location.z > 260.0 && distance > 350.0)
            {
                self.power_clear = None;
                return ControllerState::default();
            }
            let velocity_change = (ball.velocity - state.initial_ball_velocity).magnitude();
            let likely_contact = distance < 300.0
                && (ball.location.z > state.initial_ball_z + 25.0
                    || ball.velocity.z > 120.0
                    || velocity_change > 300.0);
            if likely_contact {
                state.phase = 2;
                state.contact_time = Some(game_time);
                self.power_clear = Some(state);
                return ControllerState::default();
            }

            let angle = local_ball.y.atan2(local_ball.x).abs();
            let drive = default_drive(
                &car,
                DriveInput {
                    target_speed: 1300.0,
                    local_target: local_ball,
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel: self.boost_accel,
                },
            );
            let mut controller = controller_from_drive(drive, car);
            controller.throttle = 1.0;
            controller.boost = angle < 0.18
                && distance > 100.0
                && car.boost > 0.0
                && car.velocity.magnitude() < 1390.0;
            self.power_clear = Some(state);
            return controller;
        }

        let elapsed = game_time - state.contact_time.unwrap_or(game_time);
        if ball.location.z > 140.0 || ball.velocity.z > 100.0 {
            let friend_count = cars.iter().filter(|other| {
                other.team == car.team && other.index != car.index && !other.demolished
            }).count();
            let foe_count = cars.iter().filter(|other| {
                other.team != car.team && !other.demolished
            }).count();
            let max_aerial_height = if friend_count == 0 && foe_count == 1 {
                1200.0
            } else {
                f32::INFINITY
            };
            let min_aerial_height = if max_aerial_height > 1200.0
                && car.location.z >= 2044.0 - car.hitbox.height * 1.1
            {
                551.0
            } else if self.unlimited_boost || car.airborne {
                150.0
            } else {
                450.0
            };
            let aim = relative_goal_aim_point(car, ball.location, self.shot_targets);
            if let Some(candidate) = find_aerial_candidate(
                car,
                prediction,
                aim,
                self.shot_targets,
                cars,
                self.match_state.gravity,
                self.boost_accel,
                game_time,
                self.kickoff_contest_until,
                min_aerial_height,
                max_aerial_height,
                6.0,
                self.unlimited_boost,
            ) {
                let mut aerial = AerialShot::new(
                    car, candidate, game_time, self.match_state.gravity, self.boost_accel,
                );
                let aerial_goal = relative_goal_aim_point(car, ball.location, self.shot_targets);
                aerial.refresh_prediction(
                    car, prediction, game_time, aerial_goal, self.match_state.gravity, ball.location.z,
                );
                if aerial.fast_aerial
                    && (aerial.ball.z <= 350.0 || ball.location.z <= 350.0)
                {
                    aerial.fast_aerial = false;
                    aerial.double_jump = false;
                }
                let controls = aerial.run(
                    car, game_time, self.match_state.gravity, self.boost_accel, 0.0,
                );
                self.power_clear = None;
                if !aerial.finished { self.aerial_shot = Some(aerial); }
                return controls;
            }
        }
        if elapsed > 1.45 {
            self.power_clear = None;
        } else {
            self.power_clear = Some(state);
        }
        ControllerState::default()
    }

    fn run_wall_approach(
        &mut self,
        car: CarState,
        current_ball: BallState,
        plan: WallDrivePlan,
        prediction: &BallTrajectory,
        game_time: f32,
    ) -> ControllerState {
        // The source wall_approach aborts when its prediction expires or
        // the ball no longer satisfies Noob Black's wall-shot conditions.
        // Do not keep driving at the stale ball position after that abort;
        // the normal planner gets the next packet to choose another routine.
        if !(0.12..=5.2).contains(&plan.intercept_time) {
            self.wall_drive_candidate = None;
            return ControllerState::default();
        }

        let side = team_side(car.team);
        let foe_goal = Vector::new(0.0, -side * 5120.0, 300.0);
        let Some(geometry) = wall_approach_geometry(
            &car, plan.ball_slice, car.team, foe_goal, plan.mode, plan.intercept_time,
        ) else {
            self.wall_drive_candidate = None;
            return ControllerState::default();
        };

        let mut output = default_drive(
            &car,
            DriveInput {
                target_speed: geometry.speed_required,
                local_target: geometry.local_target,
                upside_down: false,
                up: geometry.surface_up.map(|up| car.local(up)),
                handbrake: false,
                boost_accel: self.boost_accel,
            },
        );
        let yaw_error = if geometry.on_wall {
            output.target_angles.yaw.abs()
        } else {
            geometry.local_angle
        };
        let needs_boost = car.boost > 0.0
            && geometry.distance > 300.0
            && geometry.speed_required > car.velocity.magnitude() + 100.0
            && yaw_error < 0.85
            && output.handbrake != Some(true);
        output.boost = Some(needs_boost);
        if needs_boost {
            // Python wall_approach explicitly overrides defaultThrottle's
            // throttle to full when the wall route has a valid boost window.
            output.throttle = Some(1.0);
        }
        let mut controller = controller_from_drive(output, car);
        let time_remaining = plan.intercept_time;

        if geometry.on_wall && geometry.distance < 320.0 && time_remaining < 1.65 {
            if plan.mode == WallDriveMode::Aerial {
                let slice = crate::prediction::BallSlice {
                    time: time_remaining,
                    ..plan.ball_slice
                };
                let shot_target = relative_goal_aim_point(car, geometry.ball, self.shot_targets);
                let (target_location, shot_direction) = party_cannon_contact_geometry(
                    car, slice, shot_target, self.match_state.gravity,
                );
                let candidate = AerialCandidate {
                    slice,
                    shot_target,
                    target_location,
                    shot_direction,
                    double_jump: true,
                    fast: true,
                    estimated_boost: car.boost,
                    score: time_remaining,
                };
                let aerial = AerialShot::new(
                    car, candidate, game_time, self.match_state.gravity, self.boost_accel,
                );
                // Python wall_approach pushes Aerial and returns. The aerial
                // owns the next packet; do not execute its jump on the handoff
                // frame.
                self.aerial_shot = Some(aerial);
                self.wall_drive_candidate = None;
                return controller;
            }
            if plan.mode == WallDriveMode::DoubleJump {
                let shot = GroundIntercept {
                    time: time_remaining,
                    ball: geometry.ball,
                    ball_velocity: plan.ball_slice.velocity,
                    approach: geometry.ball - geometry.shot_direction * (92.75 + car.hitbox.length * 0.5),
                    shot_direction: geometry.shot_direction,
                    contact_quality: 1.0,
                    contact_point: geometry.ball - geometry.shot_direction * 92.75,
                    contact_face: "centre",
                    shot_type: ShotType::DoubleJump,
                    side_shot: None,
                    targets: self.shot_targets,
                };
                // Python pops wall_approach and pushes double_jump here.
                // Keep the intercept as its own routine until the double
                // jump launch phase commits, even if the ball leaves the wall.
                self.pending_double_jump = Some(PendingDoubleJump {
                    shot,
                    intercept_time: game_time + time_remaining,
                });
                self.wall_drive_candidate = None;
                return controller;
            }

            let state = JumpShotState {
                started_at: game_time,
                intercept_time: game_time + time_remaining,
                ball: geometry.ball,
                ball_velocity: plan.ball_slice.velocity,
                approach: geometry.approach,
                shot_direction: geometry.shot_direction,
                contact_face: "centre",
                contact_face_label: None,
                goal_aim_point: None,
                needed_jump_time: get_jump_time(
                    geometry.ball.z - car.location.z,
                    car.velocity.z,
                    self.match_state.gravity.z,
                ).time as f32,
                jumping: false,
                jump_time: None,
                quick: time_remaining <= 0.45,
                counter: 0,
                patience_applied: false,
                dodge_started_at: None,
                dodge_direction: Vector::new(1.0, 0.0, 0.0),
                    side_mode: false,
                    side_sign: 1.0,
                    side_lateral_distance: 0.0,
                    targets: self.shot_targets,
            };
            self.jump_shot = Some(state);
            self.wall_drive_candidate = None;
            // Match the Python routine stack: the wall approach frame only
            // pushes jump_shot; its drive/launch phase begins next packet.
            return controller;
        }

        if !car.airborne
            && time_remaining <= geometry.jump_time
            && geometry.local_angle <= std::f32::consts::FRAC_PI_2
        {
            controller.jump = true;
        }

        if car.airborne {
            let (_, pd) = default_pd(
                &car,
                car.local(geometry.ball - car.location),
                false,
                Some(car.local(Vector::new(0.0, 0.0, 1.0))),
            );
            controller.pitch = pd.pitch;
            controller.yaw = pd.yaw;
            controller.roll = pd.roll;
        }
        controller
    }

    fn drive_to(&mut self, car: CarState, plan: Plan, game_time: f32, ball: BallState, prediction: &BallTrajectory) -> ControllerState {
        if plan.decision == Decision::Boost {
            let Some(pad) = self.boost_target else {
                return ControllerState::default();
            };
            let car_to_target = pad.location - car.location;
            let distance_remaining = car_to_target.flatten().magnitude();
            let mut final_target = pad.location.flatten();
            if !pad.large && distance_remaining > 640.0 {
                let vector = ball.location;
                let side_value = vector.cross(Vector::new(0.0, 0.0, 1.0)).dot(car_to_target);
                let side_of_vector = if side_value > 0.0 {
                    1.0
                } else if side_value < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let perpendicular = car_to_target
                    .cross(Vector::new(0.0, 0.0, side_of_vector))
                    .normalize();
                let flat_to_target = car_to_target.flatten();
                let flat_vector = vector.flatten();
                let signed_angle = flat_to_target
                    .cross(flat_vector)
                    .z
                    .atan2(flat_to_target.dot(flat_vector));
                let adjustment = signed_angle * distance_remaining / std::f32::consts::PI;
                final_target = final_target + perpendicular * adjustment;
            }
            final_target = cap_in_field(&car, final_target);
            let output = default_drive(
                &car,
                DriveInput {
                    target_speed: MAX_SPEED,
                    local_target: car.local_location(final_target),
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel: self.boost_accel,
                },
            );
            return controller_from_drive(output, car);
        }
        if plan.decision == Decision::TransitionReceive
            && self.jump_shot.is_none()
            && self.double_jump_shot.is_none()
        {
            let direction = self.transition_direction.unwrap_or_else(|| {
                (plan.target - ball.location).flatten().normalize()
            });
            let face_target = plan.target + direction * 1000.0;
            let mut drive = default_drive(
                &car,
                DriveInput {
                    target_speed: 1700.0,
                    local_target: car.local_location(plan.target),
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel: self.boost_accel,
                },
            );
            let face_local = car.local_location(face_target);
            drive.controls.steer = (face_local.y.atan2(face_local.x) * 5.0).clamp(-1.0, 1.0);
            return controller_from_drive(drive, car);
        }
        if plan.decision == Decision::EmergencyChallenge
            && self.ground_flip.is_none()
            && self.defense_flip.is_none()
        {
            let state = self.emergency_clear.unwrap_or(EmergencyClearState {
                started_at: game_time,
                target: None,
            });
            self.emergency_clear = Some(state);
            return self.run_emergency_clear(car, ball, game_time, state);
        }
        if let Some(state) = self.double_jump_shot {
            return self.run_double_jump(car, state, game_time);
        }
        if let Some(state) = self.jump_shot {
            return self.run_jump_shot(car, ball, prediction, state, game_time);
        }

        if plan.decision == Decision::WallDrive {
            if let Some(wall) = self.wall_drive_candidate {
                return self.run_wall_approach(car, ball, wall, prediction, game_time);
            }
        }

        if plan.decision == Decision::AerialChallenge {
            if let Some(candidate) = self.aerial_candidate {
                let drive = default_drive(
                    &car,
                    DriveInput {
                        target_speed: MAX_SPEED,
                        local_target: car.local_location(candidate.target_location),
                        upside_down: false,
                        up: None,
                        handbrake: false,
                        boost_accel: self.boost_accel,
                    },
                );
                let mut aerial = AerialShot::new(
                    car, candidate, game_time, self.match_state.gravity, self.boost_accel,
                );
                let aerial_goal = relative_goal_aim_point(car, ball.location, self.shot_targets);
                aerial.refresh_prediction(
                    car, prediction, game_time, aerial_goal, self.match_state.gravity, ball.location.z,
                );
                if aerial.fast_aerial
                    && (aerial.ball.z <= 350.0 || ball.location.z <= 350.0)
                {
                    aerial.fast_aerial = false;
                    aerial.double_jump = false;
                }
                let controls = aerial.run(
                    car, game_time, self.match_state.gravity, self.boost_accel,
                    drive.target_angles.yaw,
                );
                if aerial.finished {
                    return controller_from_drive(drive, car);
                }
                if !aerial.aerialing {
                    self.aerial_shot = Some(aerial);
                    if aerial.side_wall_active {
                        return controls;
                    }
                    return controller_from_drive(drive, car);
                }
                self.aerial_shot = Some(aerial);
                return controls;
            }
        }

        if let Some(jump) = plan.jump {
            if jump.shot_type == ShotType::DoubleJump {
                return self.drive_double_jump(car, ball, jump, prediction, game_time);
            }

            // Python pushes jump_shot as soon as the parser finds a viable
            // candidate. The routine owns the entire approach phase; it is
            // not created only after the car has already reached its launch
            // window. This keeps one intercept, one patience decision and one
            // live contact target across the full approach.
            let side = jump.side_shot;
            let state = JumpShotState {
                started_at: game_time,
                intercept_time: game_time + jump.time,
                ball: jump.ball,
                ball_velocity: jump.ball_velocity,
                approach: jump.approach,
                shot_direction: side.map_or(jump.shot_direction, |s| s.shot_direction),
                contact_face: jump.contact_face,
                contact_face_label: None,
                goal_aim_point: None,
                needed_jump_time: get_jump_time(
                    jump.ball.z - car.location.z,
                    car.velocity.z,
                    self.match_state.gravity.z,
                ).time as f32,
                jumping: false,
                jump_time: None,
                quick: jump.time <= 0.45,
                counter: 0,
                patience_applied: false,
                dodge_started_at: None,
                dodge_direction: Vector::new(1.0, 0.0, 0.0),
                side_mode: side.is_some(),
                side_sign: side.map_or(1.0, |s| s.side_sign),
                side_lateral_distance: side.map_or(0.0, |s| s.lateral_distance),
                targets: jump.targets,
            };
            self.jump_shot = Some(state);
            return self.run_jump_shot(car, ball, prediction, state, game_time);
        }

        if let Some(shot) = plan.shot {
            if let Some(flip) = self.ground_flip {
                return self.run_ground_flip(Some(car), game_time, flip);
            }
            let t = shot.time;
            let time_remaining = t.clamp(0.000001, 6.0);
            // Python ground_shot caches its goal aim from the live ball.
            // Keep that height/aim choice separate from the predicted contact.
            let goal_aim = relative_goal_aim_point(car, ball.location, shot.targets);
            let shot_vector = (goal_aim - shot.ball).flatten().normalize();
            // Python's agent.best_shot_value is 92.75 for ground_shot.
            let offset_target = contact_approach_from_label(
                shot.ball, shot_vector, shot.contact_face, 92.75,
            ).unwrap_or(shot.ball - shot_vector * 92.75);

            let local_ball = car.local_location(ball.location);
            let close_contact = local_ball.y.abs() < 92.75 + car.hitbox.width * 0.5
                && local_ball.z.abs() < 92.75 + car.hitbox.height * 0.5;
            let mut final_target = if close_contact {
                ball.location - shot_vector * 92.75
            } else {
                let target = shot_approach_target(
                    car, shot.ball, offset_target.flatten(), shot_vector, t,
                );
                target
            };
            let distance_remaining = final_target.flat_distance(car.location)
                - car.hitbox.length / 3.0;
            let mut speed_required = if close_contact {
                MAX_SPEED
            } else {
                distance_remaining / time_remaining
            };
            let ball_speed_toward_shot = ball.velocity.dot(shot_vector);
            if !close_contact
                && speed_required < 1900.0
                && car.boost > 50.0
                && t > 2.0
                && distance_remaining > 2560.0
                && ball_speed_toward_shot > 1000.0
            {
                speed_required *= 0.75;
            }

            // Python ground_shot measures arrival distance before field
            // clamping, then drives with defaultDrive at the required speed.
            // controlled_drive adds non-source heading caps and can brake
            // even when the planned contact is straight ahead.
            final_target = cap_in_field(&car, final_target);
            let local_final_target = car.local_location(final_target);
            let local_offset_target = car.local_location(offset_target.flatten());
            let angle_to_target = local_final_target.y.atan2(local_final_target.x).abs();
            let output = default_drive(
                &car,
                DriveInput {
                    target_speed: speed_required,
                    local_target: local_final_target,
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel: self.boost_accel,
                },
            );
            let velocity = if output.car_speed.abs() < 0.001 { 1.0 } else { output.car_speed };
            let impulse = dodge_impulse(car.velocity, MAX_SPEED);
            let denominator = velocity + impulse;
            let time_to_target = if denominator.abs() > 0.001 {
                distance_remaining / denominator
            } else {
                f32::INFINITY
            };
            let future_velocity = car.velocity + self.match_state.gravity * t;
            let local_future_velocity = car.local(future_velocity.flatten());
            let aligned = local_offset_target.y.abs() < 92.0
                && local_future_velocity.x >= local_offset_target.x - car.hitbox.length / 3.0
                && local_offset_target.x > 0.0;
            let close_enough = velocity.abs() < 100.0
                && distance_remaining < car.hitbox.length / 2.0;

            if !car.airborne
                && t < 0.35 && t > 0.10
                && (velocity.abs() + impulse).min(MAX_SPEED)
                    <= if car.boost > 12.0 { speed_required.abs() } else { 1900.0 }
                && (close_enough || aligned)
            {
                let flip = FlipState {
                    started_at: game_time,
                    direction: local_offset_target.flatten().normalize(),
                    cancel: angle_to_target > 2.0,
                };
                self.ground_flip = Some(flip);
                return self.run_ground_flip(Some(car), game_time, flip);
            } else if car.airborne {
                let recovery_target = if t > 0.5 {
                    final_target
                } else {
                    car.location
                        + if car.velocity.magnitude() > 1.0 {
                            car.velocity.normalize() * 1000.0
                        } else {
                            Vector::ZERO
                        }
                };
                self.defense_recovery_target = Some(recovery_target);
                return GridDefense::recovery_controller(
                    car, recovery_target, self.match_state.gravity.z,
                );
            } else {
                let runtime_reachable = runtime_ground_reachability(
                    t,
                    self.boost_accel,
                    &car,
                    (final_target - car.location).normalize(),
                    distance_remaining,
                ).is_some_and(|_| {
                    ground_shot_is_viable(car.hitbox.height, car.airborne, offset_target.z)
                });
                if t <= 0.0
                    || (t > 0.35 && distance_remaining > 50.0 && !runtime_reachable)
                {
                    // Python calls defaultDrive before its viability check.
                    // Popping the routine does not undo the controller values
                    // already written by defaultDrive for this frame.
                    return controller_from_drive(output, car);
                }
            }
            if !self.unlimited_boost
                && car.boost < 48.0
                && angle_to_target < 0.03
                && velocity > 500.0
                && t > 1.0
                && time_to_target >= 1.5
            {
                if self.match_state.gravity.z < -450.0 && time_to_target < 3.0 && time_to_target > 1.0 {
                    let mut wave = crate::mechanics::WaveDash::new(Some(local_offset_target));
                    let result = wave.tick(car, game_time);
                    if !result.finished {
                        self.wave_dash = Some(wave);
                        return result.controller;
                    }
                } else {
                    let flip = FlipState {
                        started_at: game_time,
                        direction: local_offset_target.flatten().normalize(),
                        cancel: false,
                    };
                    self.ground_flip = Some(flip);
                    return self.run_ground_flip(Some(car), game_time, flip);
                }
            }

            return controller_from_drive(output, car);
        }

        self.ground_flip = None;
        let target = plan.target.flatten();
        let local_target = car.local_location(target);
        let distance = local_target.flatten().magnitude();
        let angle = local_target.y.atan2(local_target.x).abs();

        // generic_kickoff's final forward dodge, triggered only once the
        // kicker is within the original 550-unit launch window.
        if plan.decision == Decision::KickoffKicker
            && !car.airborne
            && distance < 550.0
        {
            let direction = car.local_location(Vector::new(
                0.0,
                -team_side(car.team) * 5120.0,
                300.0,
            )).flatten().normalize();
            let state = KickoffDodgeState {
                started_at: game_time,
                direction,
            };
            self.kickoff_dodge = Some(state);
            return self.run_kickoff_dodge(state, game_time);
        }

        // Port of goto's one-speedflip-per-navigation-run gate.
        let forward_speed = car.local_velocity().x;
        if distance < 600.0 {
            self.navigation_speedflip_used = false;
        }
        if !car.airborne
            && !self.navigation_speedflip_used
            && local_target.x > 0.0
            && angle < 0.14
            && distance > 3000.0
            && (600.0..1600.0).contains(&forward_speed)
            && car.boost >= 20.0
        {
            let right_handed = car.location.x * -team_side(car.team) > 0.0;
            let state = SpeedFlipState {
                step_start: game_time,
                step_index: 0,
                direction: if right_handed { 1.0 } else { -1.0 },
                use_boost: true,
            };
            self.navigation_speedflip_used = true;
            self.speed_flip = Some(state);
            return self.run_speed_flip(car, state, game_time);
        }

        // Exact half_flip_manager navigation gate from Volt's goto routine:
        // target is substantially behind, reverse velocity is established,
        // and enough distance remains to benefit from the reposition.
        if !car.airborne
            && angle > 2.5
            && car.local_velocity().x < -300.0
            && distance > 600.0
        {
            let state = HalfFlipState {
                started_at: game_time,
                release_frames: 0,
                flip_started_at: None,
            };
            self.half_flip = Some(state);
            return self.run_half_flip(car, state, game_time, self.match_state.gravity.z);
        }

        // Python goto.run drives at 2300 through defaultDrive. The
        // source throttle routine already applies its curvature-based turn
        // limit; extra distance/heading caps here change the source behavior.
        let output = default_drive(
            &car,
            DriveInput {
                target_speed: plan.target_speed,
                local_target,
                upside_down: false,
                up: None,
                handbrake: false,
                boost_accel: self.boost_accel,
            },
        );

        let controller = controller_from_drive(output, car);

        // The selected decision is intentionally retained in Plan for
        // deterministic testing and future routine-specific execution.
        let _ = plan.decision;
        controller
    }


    fn drive_double_jump(
        &mut self,
        car: CarState,
        current_ball: BallState,
        shot: GroundIntercept,
        prediction: &BallTrajectory,
        game_time: f32,
    ) -> ControllerState {
        // Port double_jump._select_patient_intercept: begin at the selected
        // candidate, wait for the first later reachable double-jump slice,
        // and delay a poor opponent-back-wall angle until it enters the
        // central shooting lane.
        let (t, ball) = patient_double_jump_intercept(
            car,
            prediction,
            shot.targets,
            shot.time,
            foe_goal(team_side(car.team)).y,
        )
        .map_or((shot.time, shot.ball), |(time, location)| (time, location));

        // Python caches _relative_goal_aim_point on the first routine
        // frame using the live ball's half of the field, not the future slice.
        let goal_aim = relative_goal_aim_point(car, current_ball.location, shot.targets);
        let shot_direction = (goal_aim - ball).flatten().normalize();
        // Keep the contact label selected by tools.select_ball_contact.
        // Rebuild its surface normal against the refreshed ball and shot
        // direction, using Python's best_shot_value standoff (min 120 uu).
        let offset_target = contact_approach_from_label(
            ball, shot_direction, shot.contact_face, 92.75,
        ).unwrap_or(ball - shot_direction * 92.75);

        let required_height = (offset_target.z - car.location.z).max(0.0);
        let Some(jump_time) = grid_jump_time(required_height, true) else {
            return ControllerState::default();
        };
        let tj = t - jump_time * 1.15;
        let mut final_target = offset_target.flatten();
        if tj > 0.0 {
            final_target = shot_approach_target(
                car, shot.ball, final_target, shot_direction, tj,
            );
        }
        final_target = cap_in_field(&car, final_target);
        let local_target = car.local_location(final_target);
        let local_offset = car.local_location(offset_target.flatten());
        let angle_to_target = local_target.y.atan2(local_target.x).abs();
        let distance_remaining = final_target.flat_distance(car.location)
            - car.hitbox.length / 3.0;
        // Python's reverse-drive decision uses the live ball, not the
        // selected future prediction slice.
        let local_ball = car.local_location(current_ball.location);
        let direction_sign = if angle_to_target >= 1.6
            && car.local_velocity().x <= 1000.0
            && local_ball.x < -250.0
        { -1.0 } else { 1.0 };

        let speed_required = distance_remaining / t.clamp(0.000001, 6.0);
        // Python double_jump.run calls defaultDrive with the already
        // calculated local target. Do not route this through a separate
        // speed/heading heuristic.
        let output = default_drive(
            &car,
            DriveInput {
                target_speed: speed_required * direction_sign,
                local_target,
                upside_down: false,
                up: None,
                handbrake: false,
                boost_accel: self.boost_accel,
            },
        );
        let mut velocity = output.car_speed;
        if velocity == 0.0 { velocity = 1.0; }
        let local_velocity = car.local_velocity();
        let denominator = velocity.abs() + dodge_impulse(car.velocity, MAX_SPEED);
        let time_to_target = if denominator > 0.001 {
            distance_remaining / denominator
        } else {
            f32::INFINITY
        };
        let aligned = local_offset.y.abs() < 92.0
            && direction_sign * local_velocity.x
                >= direction_sign * (local_offset.x - car.hitbox.length / 3.0)
            && direction_sign * local_offset.x > 0.0;
        let close_enough = velocity.abs() < 100.0
            && distance_remaining < car.hitbox.length / 2.0;

        if !car.airborne && t <= jump_time && (aligned || close_enough) {
            let state = DoubleJumpState {
                started_at: game_time,
                intercept_time: game_time + t,
                ball,
                offset_target,
                shot_direction,
                jump_time,
                jump_frames: 0,
                second_jump_sent: false,
                prev_jump: false,
                targets: shot.targets,
            };
            self.double_jump_shot = Some(state);
            return self.run_double_jump(car, state, game_time);
        }

        if car.airborne {
            self.defense_recovery_target = Some(final_target);
            return GridDefense::recovery_controller(car, final_target, self.match_state.gravity.z);
        }

        // Python's CLib double_jump_shot_is_viable first runs the generic
        // ground reachability simulation (turning, braking, boost and
        // distance), then validates the double-jump vertical window.
        let runtime_reachable = runtime_ground_reachability(
            t,
            self.boost_accel,
            &car,
            (final_target - car.location).normalize(),
            distance_remaining,
        )
        .is_some_and(|remaining| {
            double_jump_shot_is_viable(
                remaining,
                self.match_state.gravity.z,
                car.velocity.z,
                car.hitbox.height * 0.5,
                car.airborne,
                offset_target.z,
                true,
            )
        });
        if t <= jump_time || (tj > 0.0 && distance_remaining > 50.0 && !runtime_reachable) {
            // Python double_jump.run executes defaultDrive before its
            // reachability/timeout exit. Preserve that frame's drive controls.
            return controller_from_drive(output, car);
        }

        if !self.unlimited_boost
            && car.boost < 48.0
            && angle_to_target < 0.03
            && velocity > 600.0
            && time_to_target >= jump_time * 1.15 + 1.75
        {
            if self.match_state.gravity.z < -450.0 && time_to_target < 5.0 {
                let mut wave = crate::mechanics::WaveDash::new(Some(local_offset));
                let result = wave.tick(car, game_time);
                if !result.finished {
                    self.wave_dash = Some(wave);
                    return result.controller;
                }
            } else {
                let flip = FlipState {
                    started_at: game_time,
                    direction: local_offset.flatten().normalize(),
                    cancel: false,
                };
                self.ground_flip = Some(flip);
                return self.run_ground_flip(Some(car), game_time, flip);
            }
        }

        if !self.unlimited_boost
            && angle_to_target >= 2.0
            && velocity < 200.0
            && time_to_target >= jump_time * 1.15 + 2.0
            && distance_remaining > 1280.0
        {
            let state = HalfFlipState {
                started_at: game_time,
                release_frames: 0,
                flip_started_at: None,
            };
            self.half_flip = Some(state);
            return self.run_half_flip(car, state, game_time, self.match_state.gravity.z);
        }

        controller_from_drive(output, car)
    }

    /// Start the Python ball_recovery routine after a committed shot
    /// finishes or aborts. Unlike ordinary recovery, this may boost toward
    /// a predicted floor landing.
    fn start_ball_recovery(&mut self, car: CarState) -> ControllerState {
        self.defense_recovery_target = None;
        self.ball_recovery = car.airborne;
        GridDefense::ball_recovery_controller(car, self.match_state.gravity.z)
    }

    fn run_double_jump(
        &mut self,
        car: CarState,
        mut state: DoubleJumpState,
        game_time: f32,
    ) -> ControllerState {
        let time_remaining = state.intercept_time - game_time;
        let elapsed = (game_time - state.started_at).max(0.0);
        state.jump_frames = (elapsed * 120.0).round().max(0.0) as u32;
        if time_remaining <= -0.4 || (!car.airborne && state.jump_frames >= 30) {
            self.double_jump_shot = None;
            return self.start_ball_recovery(car);
        }

        let local_contact = car.local_location(state.offset_target);
        let (_, pd) = default_pd(&car, local_contact, false, None);
        let mut controller = ControllerState::default();
        controller.pitch = pd.pitch;
        controller.yaw = pd.yaw;
        controller.roll = pd.roll;

        if state.jump_frames < 24 {
            controller.jump = true;
            state.prev_jump = true;
        } else if !state.second_jump_sent && !car.doublejumped {
            if state.prev_jump {
                controller.jump = false;
                state.prev_jump = false;
            } else {
                controller.jump = true;
                state.second_jump_sent = true;
                state.prev_jump = true;
            }
        } else {
            controller.jump = false;
            state.prev_jump = false;
        }

        if time_remaining > 0.0 {
            let future_position = car.location
                + car.velocity * time_remaining
                + self.match_state.gravity * (0.5 * time_remaining * time_remaining);
            let delta = state.offset_target - future_position;
            let forward_need = delta.dot(car.forward()) / time_remaining.max(0.000001);
            let min_boost_time = 1.0 / 30.0;
            let throttle_accel = 66.0 + (2.0 / 3.0);
            if forward_need > self.boost_accel * min_boost_time && car.boost > 0.0 {
                controller.boost = true;
            } else if forward_need.abs() > 1.0 {
                controller.throttle = (forward_need / (throttle_accel * min_boost_time))
                    .clamp(-1.0, 1.0);
            }
        }

        let _ = state.shot_direction;
        self.double_jump_shot = Some(state);
        controller
    }

    fn run_jump_shot(
        &mut self,
        car: CarState,
        current_ball: BallState,
        prediction: &BallTrajectory,
        mut state: JumpShotState,
        game_time: f32,
    ) -> ControllerState {
        let mut jump_elapsed = state.jump_time.map_or(0.0, |started| (game_time - started).max(0.0));
        if state.jumping && state.jump_time.is_none() {
            // Python starts its jump timer on the first frame of the jumping
            // branch, so that frame receives the initial jump impulse.
            state.jump_time = Some(game_time);
            jump_elapsed = 0.0;
        }
        // Python's shot_patience.adjust runs inside the routine, including
        // the first frame after the shot is selected. Do not wait until the
        // car reaches the jump window before allowing the intercept to move.
        if !state.jumping && !state.patience_applied {
            let foe_goal_y = -team_side(car.team) * 5120.0;
            if let Some((new_time, later_slice)) = shot_patience_adjustment(
                car, &prediction.slices, game_time, state.intercept_time, foe_goal_y,
            ) {
                state.intercept_time = new_time;
                state.ball = later_slice.location;
                state.ball_velocity = later_slice.velocity;
                state.patience_applied = true;
            }
        }

        if state.goal_aim_point.is_none() {
            // Python caches _relative_goal_aim_point on the first routine
            // frame, using the live ball's half of the field. Do not recompute
            // it from the future intercept slice on every packet.
            state.goal_aim_point = Some(relative_goal_aim_point(
                car, current_ball.location, state.targets,
            ));
        }
        let goal_aim = state.goal_aim_point.unwrap_or_else(|| {
            relative_goal_aim_point(car, current_ball.location, state.targets)
        });

        let t = state.intercept_time - game_time;
        let time_remaining = t.clamp(0.000001, 6.0);

        // Python refreshes its predicted ball slice every packet until the
        // jump begins. A committed shot must follow the moving intercept,
        // not keep steering toward the stale slice from selection time.
        if !state.jumping {
            if prediction.slices.is_empty() {
                self.jump_shot = None;
                return ControllerState::default();
            }
            let slice_index = ((t * 120.0).round() as isize - 1)
                .clamp(0, prediction.slices.len() as isize - 1) as usize;
            let slice = prediction.slices[slice_index];
            state.ball = slice.location;
            state.ball_velocity = slice.velocity;
            state.needed_jump_time = get_jump_time(
                state.ball.z - car.location.z,
                car.velocity.z,
                self.match_state.gravity.z,
            ).time as f32;
        }

        // Recalculate the goal-directed shot vector and contact approach each
        // frame, as Python jump_shot.run does after it has committed to the
        // predicted ball slice.
        let predicted_slice = BallSlice {
            time: time_remaining,
            location: state.ball,
            velocity: state.ball_velocity,
        };
        // Python jump_shot calls Aerial._party_cannon_shot_direction
        // directly, then applies WallShot.adjust_jump after blending. Do not
        // use party_cannon_contact_geometry here: that helper already applies
        // the aerial wall correction and would double-correct this shot.
        let predicted_shot = party_cannon_shot_direction(
            car,
            state.ball,
            state.ball_velocity,
            time_remaining,
            goal_aim,
            self.match_state.gravity,
        );
        let goal_shot = (goal_aim - state.ball).flatten().normalize();
        let predicted_shot = predicted_shot.flatten().normalize();
        let blended_shot = if goal_shot.magnitude() > 0.001
            && predicted_shot.magnitude() > 0.001
        {
            (goal_shot * 0.65 + predicted_shot * 0.35).normalize()
        } else {
            goal_shot
        };
        let (adjusted_shot, wall_normal, _) = wall_shot_adjust_jump(state.ball, blended_shot);
        state.shot_direction = adjusted_shot;

        if state.side_mode {
            let lateral = Vector::new(
                -state.shot_direction.y,
                state.shot_direction.x,
                0.0,
            ).normalize();
            state.approach =
                state.ball - lateral * state.side_lateral_distance * state.side_sign;
        } else {
            // Python _party_cannon_hitbox_target: ball radius plus half the
            // actual hitbox length, with the current contact label rebuilt
            // against the refreshed shot vector.
            let contact_standoff = 92.75 + car.hitbox.length * 0.5;
            let nominal_target = state.ball - state.shot_direction * contact_standoff;
            // VirxERLU exposes hitbox offset in local forward/right/up axes;
            // convert it to world space exactly as Python Aerial does.
            let world_hitbox_offset = car.forward() * car.hitbox.offset.x
                + car.up().cross(car.forward()) * car.hitbox.offset.y
                + car.up() * car.hitbox.offset.z;
            let hitbox_target = nominal_target - world_hitbox_offset;
            let mut offset_target = contact_approach_from_label(
                state.ball,
                state.shot_direction,
                state.contact_face,
                contact_standoff,
            ).map_or(hitbox_target, |refreshed| {
                refreshed + hitbox_target - nominal_target
            });

            if let Some(normal) = wall_normal {
                let tangent = state.shot_direction - normal * state.shot_direction.dot(normal);
                if tangent.magnitude() > 0.001 {
                    offset_target = offset_target - tangent.normalize() * 25.0;
                }
                if normal.z.abs() < 0.95 {
                    let jump_time = get_jump_time(
                        state.ball.z - car.location.z,
                        car.velocity.z,
                        self.match_state.gravity.z,
                    ).time as f32;
                    offset_target.z = state.ball.z
                        + (offset_target.z - state.ball.z).clamp(-120.0, 120.0);
                    offset_target = offset_target
                        - self.match_state.gravity * (0.5 * jump_time * jump_time);
                }
            }

            if wall_normal.is_none() {
                if let Some((selected_target, label)) = party_cannon_jump_front_contact(
                    car,
                    state.ball,
                    state.shot_direction,
                    time_remaining,
                    self.match_state.gravity,
                    state.jumping,
                    state.contact_face_label,
                ) {
                    offset_target = selected_target;
                    state.contact_face_label = Some(label);
                }
            }
            state.approach = offset_target;
        }

        let local_ball = car.local_location(state.ball).flatten();
        let local_target = car.local_location(state.approach).flatten();
        let contact_distance = car.location.distance(state.approach);
        let contact_limit = (92.75 + car.hitbox.length) * 1.02;

        // Python's complete pre-launch phase: shape the approach, calculate
        // the required arrival speed, and keep driving until the jump window.
        // This must not be replaced by the airborne jump controller.
        if !state.jumping {
            let t_j = t - state.needed_jump_time * 1.15;
            let mut final_target = state.approach.flatten();
            if t_j > 0.0 && (state.targets.left - state.targets.right).magnitude() > 1.0 && !state.side_mode {
                final_target = shot_approach_target(
                    car,
                    state.ball,
                    final_target,
                    state.shot_direction,
                    t_j,
                );
            }

            let mut distance_remaining = car.location.flat_distance(final_target);
            final_target = cap_in_field(&car, final_target);
            let local_final_target = car.local_location(final_target);
            let angle_to_target = local_final_target.y.atan2(local_final_target.x).abs();
            let vf = car.velocity + self.match_state.gravity * t;
            if car.airborne {
                distance_remaining = car.local_location(state.approach.flatten()).x;
            }
            distance_remaining -= car.hitbox.length / 3.0;
            let mut speed_required = distance_remaining / time_remaining;
            let ball_speed_toward_shot = current_ball.velocity.dot(state.shot_direction);

            if speed_required < 1900.0
                && car.boost > 50.0
                && t > 2.0
                && distance_remaining > 2560.0
                && ball_speed_toward_shot > 1000.0
            {
                speed_required *= 0.75;
            }

            let current_forward_speed = car.local_velocity().x;
            let power_dodge_window = t < 0.35
                && t > 0.1
                && car.local_location(current_ball.location).x > 0.0
                && current_forward_speed > 700.0;
            let mut drive_speed = speed_required;
            if power_dodge_window {
                drive_speed = drive_speed.max(current_forward_speed.min(1200.0));
            }

            // Keep a low-rate trace of the actual Python-equivalent
            // arrival-speed controller. This distinguishes a planner timing
            // error (large T -> low required speed) from a throttle/steering
            // error (high requested speed but negative throttle).
            if ((game_time * 4.0).round() as i32)
                != (((game_time - 1.0 / 120.0) * 4.0).round() as i32)
            {
                println!(
                    "[JUMP APPROACH TRACE] T={:.3} Tj={:.3} dist={:.1} required={:.1} drive={:.1} forward={:.1} angle={:.3} local_target=({:.1},{:.1}) boost={:.1} launch_window={}",
                    t,
                    t_j,
                    distance_remaining,
                    speed_required,
                    drive_speed,
                    car.local_velocity().x,
                    angle_to_target,
                    local_final_target.x,
                    local_final_target.y,
                    car.boost,
                    t <= state.needed_jump_time * 1.08
                );
            }

            let drive = default_drive(
                &car,
                DriveInput {
                    target_speed: drive_speed,
                    local_target: local_final_target,
                    upside_down: false,
                    up: None,
                    handbrake: false,
                    boost_accel: self.boost_accel,
                },
            );
            let velocity = if drive.car_speed.abs() < 0.001 {
                1.0
            } else {
                drive.car_speed
            };
            let local_offset_target = car.local_location(state.approach.flatten());
            let local_vf = car.local(vf.flatten());
            let time_to_contact =
                distance_remaining / (velocity + dodge_impulse(car.velocity, MAX_SPEED));

            let side_ready = state.side_mode
                && distance_remaining < 180.0
                && t <= state.needed_jump_time * 1.08;
            let close_enough = velocity.abs() < 100.0
                && distance_remaining < car.hitbox.length * 0.5;
            let aligned = local_offset_target.y.abs() < 92.0
                && local_vf.x >= local_offset_target.x - car.hitbox.length / 3.0
                && local_offset_target.x > 0.0;
            let normal_ready = (close_enough || aligned)
                && t <= state.needed_jump_time * 1.08;

            if side_ready || normal_ready {
                state.quick = t <= 0.45;
                state.jumping = true;
                state.jump_time = None;
                self.jump_shot = Some(state);
                return controller_from_drive(drive, car);
            }

            // Python calls VirxERLU-CLib jump_shot_is_viable with the
            // live car, boost, heading, approach direction and distance.
            // That wrapper first runs generic_is_viable, then checks the
            // jump/dodge vertical window. The vertical-only helper by itself
            // is not an equivalent reachability test.
            let runtime_reachable = runtime_ground_reachability(
                t,
                self.boost_accel,
                &car,
                (final_target - car.location).normalize(),
                distance_remaining,
            )
            .is_some_and(|remaining| {
                jump_shot_is_viable(
                    remaining,
                    self.match_state.gravity.z,
                    car.velocity.z,
                    car.hitbox.height,
                    car.airborne,
                    state.approach.z,
                    true,
                )
            });

            if t <= state.needed_jump_time
                || (t_j > 0.0 && distance_remaining > 50.0 && !runtime_reachable)
            {
                self.jump_shot = None;
                if car.airborne {
                    let recovery_target = if car.velocity.magnitude() > 1.0 {
                        car.location + car.velocity.normalize() * 1000.0
                    } else {
                        car.location + car.forward() * 1000.0
                    };
                    self.defense_recovery_target = Some(recovery_target);
                    return GridDefense::recovery_controller(
                        car, recovery_target, self.match_state.gravity.z,
                    );
                }
                // Python defaultDrive has already populated the controller
                // before jump_shot_is_viable rejects and pops this routine.
                return controller_from_drive(drive, car);
            }

            if !self.unlimited_boost
                && car.boost < 48.0
                && angle_to_target < 0.03
                && velocity > 600.0
                && time_to_contact >= state.needed_jump_time * 1.15 + 1.5
            {
                if self.match_state.gravity.z < -450.0 && time_to_contact < 5.0 {
                    let mut wave = crate::mechanics::WaveDash::new(Some(local_offset_target));
                    let result = wave.tick(car, game_time);
                    if !result.finished {
                        self.wave_dash = Some(wave);
                        return result.controller;
                    }
                } else {
                    let flip = FlipState {
                        started_at: game_time,
                        direction: local_offset_target.flatten().normalize(),
                        cancel: false,
                    };
                    self.ground_flip = Some(flip);
                    return self.run_ground_flip(Some(car), game_time, flip);
                }
            }

            self.jump_shot = Some(state);
            return controller_from_drive(drive, car);
        }

        // Python's jump_shot aborts only after the intercept has expired by
        // 0.8s, or after it has landed without finding a dodge window.
        if t <= -0.8 || (!car.airborne && state.counter >= 3) {
            self.jump_shot = None;
            self.defense_recovery_target = None;
            if car.airborne {
                let target = if car.velocity.magnitude() > 1.0 {
                    car.location + car.velocity.normalize() * 1000.0
                } else {
                    car.location + car.forward() * 1000.0
                };
                self.defense_recovery_target = Some(target);
                return GridDefense::recovery_controller(
                    car, target, self.match_state.gravity.z,
                );
            }
            return ControllerState::default();
        }

        let mut controller = ControllerState::default();

        // Python jump_shot's flight prediction: jump impulse is added on the
        // launch frame only; hf is captured before the held-jump acceleration.
        let gravity = self.match_state.gravity;
        let mut vf = car.velocity + gravity * t;
        let mut xf = car.location + car.velocity * t + gravity * (0.5 * t * t);
        if jump_elapsed <= 1.0e-6 {
            vf = vf + car.up() * JUMP_SPEED;
            xf = xf + car.up() * (JUMP_SPEED * t);
        }
        let hf = vf.z;
        let tau = (JUMP_MAX_DURATION - jump_elapsed).max(0.0);
        vf = vf + car.up() * (JUMP_ACCEL * tau);
        xf = xf + car.up() * (JUMP_ACCEL * tau * (t - 0.5 * tau));

        // Orient and add forward-axis acceleration only during the first
        // three jump-counter phases, exactly like the Python routine.
        if state.counter < 3 && state.dodge_started_at.is_none() {
            let local_shot_direction = car
                .local(state.shot_direction.flatten())
                .flatten()
                .normalize();
            let (_, pd) = default_pd(&car, local_shot_direction, false, None);
            controller.pitch = pd.pitch;
            controller.yaw = pd.yaw;
            controller.roll = pd.roll;

            let hold_s = state.ball.z > 130.0
                && state.ball.z > car.location.z + 65.0
                && !state.quick;
            if hold_s {
                controller.pitch = 1.0;
                controller.roll = if pd.yaw.abs() > 0.08 {
                    -0.20 * pd.yaw.signum()
                } else {
                    0.0
                };
            }

            let delta_x = state.approach - xf;
            let d_direction = delta_x.normalize();
            if car.forward().dot(d_direction).abs() > 0.5 && t > 1.0e-6 {
                let mut delta_v = delta_x.dot(car.forward()) / t;
                if car.boost > 0.0 && delta_v >= self.boost_accel * MIN_AIR_BOOST_TIME {
                    controller.boost = true;
                    delta_v -= self.boost_accel * MIN_AIR_BOOST_TIME;
                }
                // Python jump_shot uses the fixed global throttle_accel
                // constant (66 2/3 uu/s²) for this airborne correction.
                // throttle_acceleration() is the nonlinear ground-drive curve
                // and must not be substituted here.
                let throttle_accel = 66.0 + (2.0 / 3.0);
                if delta_v.abs() > throttle_accel * MIN_AIR_BOOST_TIME {
                    controller.throttle = (delta_v
                        / (throttle_accel * MIN_AIR_BOOST_TIME)).clamp(-1.0, 1.0);
                }
            }
        }

        // Counter 3 is the exact Python dodge window. Do not delay it behind
        // an invented 0.18s timer or require extra RLBot dodge flags.
        if state.counter == 3 {
            if contact_distance < contact_limit
                && (local_target.x > 0.0 || state.side_mode)
            {
                let dodge_direction = if local_ball.magnitude() > 1.0 {
                    local_ball.normalize()
                } else {
                    Vector::new(1.0, 0.0, 0.0)
                };
                state.dodge_direction = dodge_direction;
                state.dodge_started_at = Some(game_time);
                state.counter = 4;
                self.last_flip_time = game_time;
                controller.jump = true;
                controller.pitch = -dodge_direction.x;
                controller.yaw = dodge_direction.y;
            } else if local_target.x <= 0.0
                || (contact_distance > contact_limit + 150.0 && t < 0.12)
            {
                state.counter = 5;
            }
        } else if state.counter == 4 {
            // The dodge has been issued; let the original jump hold and
            // intercept-expiry logic complete the routine.
        } else if state.counter == 5 {
            self.jump_shot = None;
            return self.start_ball_recovery(car);
        }

        if state.counter == 5 {
            self.jump_shot = None;
            return self.start_ball_recovery(car);
        }

        // Python's minimum 0.06s jump hold and its original hf/height
        // comparison are retained rather than replaced with a vertical-speed
        // controller.
        if jump_elapsed <= JUMP_MAX_DURATION
            && (jump_elapsed <= 0.06 || hf <= state.approach.z)
        {
            controller.jump = true;
        } else if state.counter < 3 {
            state.counter += 1;
        }

        let _ = vf;
        self.jump_shot = Some(state);
        controller
    }

    fn run_ground_flip(&mut self, car: Option<CarState>, game_time: f32, flip: FlipState) -> ControllerState {
        let elapsed = game_time - flip.started_at;
        let mut controller = ControllerState::default();

        if elapsed < 0.075 {
            controller.jump = true;
        } else if elapsed < 0.10 {
            controller.jump = false;
        } else if elapsed < if flip.cancel { 0.4 } else { 0.9 } {
            if self.last_flip_time < flip.started_at {
                self.last_flip_time = game_time;
            }
            controller.jump = true;
            controller.pitch = -flip.direction.x;
            controller.yaw = flip.direction.y;
        } else {
            self.ground_flip = None;
            if game_time - self.last_flip_time >= 0.5 {
                if let Some(car) = car.filter(|car| car.airborne) {
                    let target = if car.velocity.magnitude() > 1.0 {
                        car.location + car.velocity.normalize() * 1000.0
                    } else {
                        car.location + car.forward() * 1000.0
                    };
                    self.defense_recovery_target = Some(target);
                    return GridDefense::recovery_controller(
                        car,
                        target,
                        self.match_state.gravity.z,
                    );
                }
            }
        }

        controller
    }
}


fn receive_plan(car: CarState, contact: Vector, aim: Vector) -> Option<(Vector, Vector, Vector)> {
    let mut lane = (aim - contact).flatten();
    if lane.magnitude() < 1.0 {
        lane = car.forward().flatten();
    }
    if lane.magnitude() < 1.0 {
        return None;
    }
    lane = lane.normalize();
    // Python's send_teamplay_comm scales the receive lead from the
    // predicted contact-to-aim-point distance.
    let lane_distance = (contact.flat_distance(aim) * 0.28).clamp(350.0, 1050.0);
    Some((contact + lane * lane_distance, lane, contact))
}


fn opponent_is_vulnerable(
    me: CarState,
    ball: BallState,
    cars: &[CarState],
    own_goal: Vector,
) -> bool {
    if me.location.distance(ball.location) > 1800.0 {
        return false;
    }
    cars.iter()
        .filter(|foe| foe.team != me.team && !foe.demolished)
        .any(|foe| {
            if foe.location.distance(ball.location) > 900.0 {
                return false;
            }
            let toward_our_goal = (own_goal - foe.location).flatten().normalize();
            let foe_forward = foe.forward().flatten().normalize();
            let goal_alignment = foe_forward.dot(toward_our_goal);
            let ball_behind_foe = foe.local_location(ball.location).x < -150.0;
            goal_alignment < -0.25 || ball_behind_foe
        })
}

fn should_commit_to_shot(
    me: CarState,
    cars: &[CarState],
    ball: BallState,
    roles: Option<TeamRoles>,
    urgent_save: bool,
    my_score: f32,
    my_contact_time: f32,
    messages: &HashMap<usize, TeamplayMessage>,
    game_time: f32,
) -> bool {
    if urgent_save { return true; }
    let friends: Vec<CarState> = cars.iter().copied()
        .filter(|car| car.team == me.team && car.index != me.index && !car.demolished)
        .collect();
    // Python's find_teamplay_opportunity makes the elected cover player
    // yield to any teammate with a comparable real shot/touch, before the
    // more general best-score/arrival-time comparison below.
    if friends.len() >= 2
        && roles.is_some_and(|role| role.goalie == me.index && role.challenger != me.index)
    {
        for friend in &friends {
            let Some(message) = messages.get(&friend.index) else { continue; };
            if game_time - message.received_at > 0.5 { continue; }
            if matches!(message.status.as_str(), "SHOT_FOUND" | "ANY_SHOT_FOUND" | "WALL_PASS")
                && message.score >= my_score - 8.0
            {
                return false;
            }
        }
    }
    let side = team_side(me.team);
    let ball_is_our_side = ball.location.y * side > 640.0;
    for friend in cars.iter().filter(|car| car.team == me.team && car.index != me.index && !car.demolished) {
        let Some(message) = messages.get(&friend.index) else { continue; };
        if game_time - message.received_at > 0.5 { continue; }
        if message.status == "WALL_PASS" { return false; }
        if message.status == "DEFEND" && ball_is_our_side
            && roles.is_some_and(|role| role.challenger == friend.index) {
            return false;
        }
        if !matches!(message.status.as_str(), "SHOT_FOUND" | "ANY_SHOT_FOUND") {
            continue;
        }
        if message.cover_released { return false; }
        let friend_depth = friend.location.y * side;
        let my_depth = me.location.y * side;
        let teammate_rotating_from_behind = friend_depth > my_depth + 250.0
            && ball.location.y * side < 640.0;
        if teammate_rotating_from_behind {
            let clearly_better = my_score >= message.score + 12.0
                && my_contact_time <= message.contact_time + 0.35;
            if !clearly_better { return false; }
            continue;
        }
        let teammate_has_better = message.score >= my_score + 8.0
            && message.contact_time <= my_contact_time + 0.80;
        let teammate_arrives_first = message.contact_time + 0.45 < my_contact_time
            && message.score >= my_score - 8.0;
        if teammate_has_better || teammate_arrives_first { return false; }
    }
    true
}

fn transition_bump_target(
    me: CarState,
    attacker: CarState,
    ball: BallState,
    cars: &[CarState],
    foe_goal: Vector,
) -> Option<CarState> {
    if attacker.location.flat_distance(ball.location) > 1700.0 { return None; }
    let goal_direction = (foe_goal - attacker.location).flatten().normalize();
    if goal_direction.magnitude() < 0.001 { return None; }
    let mut defenders = Vec::new();
    for foe in cars.iter().copied().filter(|car| car.team != me.team && !car.demolished) {
        let offset = (foe.location - attacker.location).flatten();
        let progress = offset.dot(goal_direction);
        if !(150.0..=3500.0).contains(&progress) { continue; }
        let lateral = (offset - goal_direction * progress).magnitude();
        if lateral > 650.0 || foe.location.flat_distance(ball.location) > 2600.0
            || me.location.flat_distance(foe.location) > 4200.0 {
            continue;
        }
        defenders.push(foe);
    }
    if defenders.is_empty() || defenders.len() > 2 { return None; }
    defenders.into_iter().min_by(|a,b| {
        a.location.flat_distance(ball.location).total_cmp(&b.location.flat_distance(ball.location))
    })
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

/// Direct port of Volt's Noob Black role election in
/// instructions for chatgpt on volt::get_grid_team_roles().
///
/// It evaluates the ball's next floor/wall/ceiling bounce, predicted goal
/// crossing, side-wall clearance lane, and each car's dependency on that
/// lane. Do not simplify this into nearest-car selection: the dependency
/// result is what makes the rotating challenger/goalie/transition roles work.
fn assign_roles_volt(
    cars: &[CarState],
    ball: BallState,
    side: f32,
    own_goal: Vector,
    foe_goal: Vector,
    prediction: &BallTrajectory,
    game_time: f32,
) -> Option<TeamRoles> {
    let mut living: Vec<CarState> = cars
        .iter()
        .copied()
        .filter(|car| !car.demolished)
        .collect();
    living.sort_by_key(|car| car.index);

    if living.is_empty() {
        return None;
    }
    if living.len() == 1 {
        return Some(TeamRoles {
            challenger: living[0].index,
            goalie: living[0].index,
            transition: None,
        });
    }

    let mut save_deadline = -1.0_f32;
    for slice in &prediction.slices {
        let time_to_goal = slice.time;
        if time_to_goal <= 0.0 {
            continue;
        }
        if time_to_goal > 5.0 {
            break;
        }
        if slice.location.y.abs() >= 5235.0 {
            save_deadline = time_to_goal;
            break;
        }
    }

    let sidewall_clear = (ball.location.y - own_goal.y).abs() <= 1000.0
        && ball.location.x.abs() >= 900.0
        && save_deadline < 0.0;

    let normal_bounce_time = next_bounce_time(ball.location, ball.velocity, 92.75);
    let reverse_bounce_time = next_bounce_time(ball.location, -ball.velocity, 92.75);
    let normal_bounce = role_air_position(ball.location, ball.velocity, normal_bounce_time);
    let reverse_bounce = role_air_position(ball.location, -ball.velocity, reverse_bounce_time);

    let mut backwall_point = None;
    if !sidewall_clear && save_deadline < 0.0 {
        let normal_hits_backwall =
            (normal_bounce.y - own_goal.y).abs() <= 100.0 && normal_bounce.z > 0.0;
        let reverse_hits_backwall =
            (reverse_bounce.y - own_goal.y).abs() <= 100.0 && reverse_bounce.z > 0.0;
        if normal_hits_backwall || reverse_hits_backwall {
            backwall_point = Some(if (normal_bounce.y - foe_goal.y).abs()
                < (reverse_bounce.y - foe_goal.y).abs()
            {
                normal_bounce
            } else {
                reverse_bounce
            });
        }
    }

    let mut ball_dependency = (ball.location.y - foe_goal.y).abs();
    if let Some(backwall) = backwall_point {
        ball_dependency = (backwall.y - foe_goal.y).abs();
    } else {
        let double_jump_height_time: f32 =
            -(564.993_16_f32 + 292.0_f32) / (-325.055_66_f32 * 2.0_f32);
        let double_jump_height = -325.055_66 * double_jump_height_time.powi(2)
            + (564.993_16 + 292.0) * double_jump_height_time
            - 11.659_18
            - 292.0 * 25.0 / 120.0;
        let bounce_time = next_bounce_time(
            ball.location,
            ball.velocity,
            double_jump_height,
        );
        let vertical_bounce_time = next_bounce_time(
            ball.location,
            Vector::new(0.0, 0.0, ball.velocity.z),
            double_jump_height,
        );
        if (bounce_time - vertical_bounce_time).abs() < 1.0e-6 {
            let low_bounce =
                role_air_position(ball.location, ball.velocity, bounce_time);
            if low_bounce.y * foe_goal.y > ball.location.y * foe_goal.y
                && low_bounce.z < 600.0
            {
                ball_dependency = (low_bounce.y - foe_goal.y).abs();
            }
        }
    }

    let ball_flat_velocity = ball.velocity.flatten();
    let ball_flat_speed = ball_flat_velocity.magnitude();
    let ball_side = if ball.location.x >= 0.0 { 1.0 } else { -1.0 };

    let mut attacker: Option<CarState> = None;
    let mut attacker_score = f32::INFINITY;
    let mut attacker_dependency = false;
    let mut goalie: Option<CarState> = None;
    let mut goalie_score = f32::INFINITY;

    for car in &living {
        let location = car.location;
        let velocity = car.velocity;
        let forward = car.forward();

        let furthest_score = location.distance(own_goal).min(900.0)
            + (location.y - own_goal.y).abs();
        if furthest_score < goalie_score {
            goalie_score = furthest_score;
            goalie = Some(*car);
        }

        let post_x = if location.x >= 0.0 { 900.0 } else { -900.0 };
        let own_goal_post = Vector::new(post_x, side * 5050.0, 0.0);
        let facing_badly = forward.y * side > 0.0;
        let falling_without_jump = location.z <= 200.0
            && car.doublejumped
            && velocity.z < 10.0
            && velocity.y * side > 0.0;
        let on_goal_post = location.flatten().distance(own_goal_post) <= 100.0;
        let facing_own_goal_from_line = location.y * side >= 5050.0
            && location.x.abs() <= 900.0
            && forward.y * side > 0.0;
        // Python's side-wall and back-wall branches intentionally use only
        // the goal-line-facing penalty. The general branch additionally
        // penalizes bad facing, falling after a double jump, and sitting on
        // the post; applying those extra penalties to every branch changes
        // which teammate is elected to challenge.
        let special_wall_clear = sidewall_clear || backwall_point.is_some();
        let closest_penalty = if special_wall_clear {
            facing_own_goal_from_line
        } else {
            facing_badly || falling_without_jump || on_goal_post || facing_own_goal_from_line
        };
        let closest_score = location.distance(ball.location)
            + if closest_penalty { 5000.0 } else { 0.0 };

        let dependency = if sidewall_clear {
            ball.location.x * ball_side >= location.x * ball_side
        } else if let Some(backwall) = backwall_point {
            (location.y - foe_goal.y).abs()
                + (location.x - backwall.x).abs() / 5.0
                > (backwall.y - foe_goal.y).abs()
        } else {
            let lane_distance = location.flatten() - ball.location.flatten();
            let cross_track = (lane_distance.x * ball_flat_velocity.y
                - lane_distance.y * ball_flat_velocity.x)
                .abs();
            let along_track = (lane_distance.x * ball_flat_velocity.x
                + lane_distance.y * ball_flat_velocity.y)
                .abs();
            let path_relevant = save_deadline < 0.0
                || (cross_track * ball_flat_speed / 2300.0 - 134.85
                    < along_track);
            (location.y - foe_goal.y).abs()
                + (location.x - ball.location.x).abs()
                    / 5.0
                    * if save_deadline >= 0.0 { 1.0 } else { 0.0 }
                > ball_dependency
                && path_relevant
        };

        if (closest_score <= attacker_score && (dependency as u8) >= (attacker_dependency as u8))
            || (dependency && !attacker_dependency)
        {
            attacker = Some(*car);
            attacker_score = closest_score;
            attacker_dependency = dependency;
        }
    }

    let mut attacker = attacker?;
    let goalie = goalie?;

    // Exact defensive goalie-angle exception from Volt: only in our half,
    // and only when the deepest player's nose has the best 3D angle to ball.
    if goalie.index != attacker.index && ball.location.y * side > 0.0 {
        let goalie_angle = car_ball_facing_angle(goalie, ball.location);
        let best_other_angle = living
            .iter()
            .filter(|car| car.index != goalie.index)
            .map(|car| car_ball_facing_angle(*car, ball.location))
            .fold(f32::INFINITY, f32::min);
        if goalie_angle + 0.001 < best_other_angle {
            attacker = goalie;
        }
    }

    let transition = living
        .iter()
        .find(|car| car.index != attacker.index && car.index != goalie.index)
        .map(|car| car.index);

    let _ = game_time; // packet-relative slices already encode time_to_goal.
    Some(TeamRoles {
        challenger: attacker.index,
        goalie: goalie.index,
        transition,
    })
}

/// Direct port of Volt's should_prioritize_ball() mode-specific challenge
/// gate. The active-shot exception and post-kickoff contest window are
/// preserved so a committed mechanic is not abandoned by a later ETA check.
fn should_prioritize_ball(
    me: CarState,
    cars: &[CarState],
    ball: BallState,
    side: f32,
    game_time: f32,
    kickoff_contest_until: f32,
    defensive_emergency: bool,
    active_shot: bool,
    roles: Option<TeamRoles>,
) -> bool {
    if active_shot {
        return true;
    }

    let distance = me.location.distance(ball.location);
    let local_ball = me.local_location(ball.location);
    let angle = local_ball.y.atan2(local_ball.x).abs();
    let ball_depth = ball.location.y * side;
    let ball_toward_our_goal = ball.velocity.y * side > 400.0;
    let ball_in_enemy_half = ball_depth < -500.0;
    let ball_near_enemy_goal = ball_depth < -3000.0;

    let friends: Vec<CarState> = cars
        .iter()
        .copied()
        .filter(|car| car.team == me.team && car.index != me.index)
        .collect();
    let foes: Vec<CarState> = cars
        .iter()
        .copied()
        .filter(|car| car.team != me.team && !car.demolished)
        .collect();
    let mode = friends.len() + 1;
    if defensive_emergency {
        return true;
    }
    if ball_depth > 4000.0 && distance < 3200.0 {
        return true;
    }
    if ball_depth > 3400.0 && ball_toward_our_goal && distance < 2800.0 {
        return true;
    }

    if mode == 1 {
        if distance < 650.0 && local_ball.x > -250.0 {
            return true;
        }
        if local_ball.x < -200.0 && distance > 1100.0 {
            return false;
        }
        if distance < 1500.0 {
            return angle < 1.65;
        }
        return angle < 1.15;
    }

    let mut teammate_has_ball = false;
    if mode != 3 {
        for friend in &friends {
            let friend_distance = friend.location.distance(ball.location);
            let friend_local = friend.local_location(ball.location);
            let friend_angle = friend_local.y.atan2(friend_local.x).abs();
            if friend_distance + 500.0 < distance
                && friend_angle < 0.85
                && friend_local.x > 0.0
            {
                teammate_has_ball = true;
            }
        }
    }

    let in_post_kickoff_window = game_time <= kickoff_contest_until;
    if mode == 2 {
        if teammate_has_ball {
            return false;
        }
        if ball_depth > 640.0 && ball_toward_our_goal && distance < 2000.0 {
            return true;
        }
        if distance < 550.0 && local_ball.x > -150.0 {
            return true;
        }
        if local_ball.x < -250.0 && distance > 1000.0 {
            return false;
        }
        if !foes.is_empty() && !in_post_kickoff_window {
            let enemy_distance = foes
                .iter()
                .map(|foe| foe.location.distance(ball.location))
                .fold(f32::INFINITY, f32::min);
            if enemy_distance + 350.0 < distance {
                return distance < 650.0 && angle < 0.7;
            }
        }
        if distance < 1400.0 {
            return angle < 1.5;
        }
        return angle < 1.2;
    }

    if mode == 3 {
        let active_challenger = roles.map(|role| role.challenger);
        if active_challenger != Some(me.index) {
            return false;
        }
        if teammate_has_ball {
            return false;
        }
        if ball_depth > 640.0 && ball_toward_our_goal && distance < 2300.0 {
            return true;
        }
        if distance < 700.0 && local_ball.x > -250.0 {
            return true;
        }
        if ball_in_enemy_half {
            if ball_near_enemy_goal && distance < 2600.0 {
                return angle < 2.0;
            }
            if distance < 1900.0 {
                return angle < 1.75;
            }
            if !foes.is_empty() && !in_post_kickoff_window {
                let enemy_distance = foes
                    .iter()
                    .map(|foe| foe.location.distance(ball.location))
                    .fold(f32::INFINITY, f32::min);
                if enemy_distance + 750.0 < distance {
                    return false;
                }
            }
            return angle < 1.45;
        }
        if distance < 1500.0 {
            return angle < 1.5;
        }
        if !foes.is_empty() && !in_post_kickoff_window {
            let enemy_distance = foes
                .iter()
                .map(|foe| foe.location.distance(ball.location))
                .fold(f32::INFINITY, f32::min);
            if enemy_distance + 450.0 < distance {
                return distance < 650.0 && angle < 0.7;
            }
        }
        if distance < 2400.0 && angle < 2.35 {
            return true;
        }
        if distance < 3100.0 && angle < 1.65 && me.boost >= 16.0 {
            return true;
        }
        return angle < 1.25;
    }

    false
}

fn next_bounce_time(position: Vector, velocity: Vector, height: f32) -> f32 {
    fn bounce_root(position_z: f32, velocity_z: f32, target: f32, root_sign: f32) -> Option<f32> {
        let a = -325.0_f32;
        let b = velocity_z;
        let c = position_z;
        let discriminant = (target - c) / a + b * b / (4.0 * a * a);
        if discriminant < 0.0 {
            return None;
        }
        Some(-b / (2.0 * a) + discriminant.sqrt() * root_sign)
    }

    let wall_x_time = if velocity.x.abs() > 0.001 {
        let sx = velocity.x.signum();
        (4096.0 - height - position.x * sx) * sx / velocity.x
    } else {
        f32::INFINITY
    };
    let wall_y_time = if velocity.y.abs() > 0.001 {
        let sy = velocity.y.signum();
        (5120.0 - height - position.y * sy) * sy / velocity.y
    } else {
        f32::INFINITY
    };
    let floor_time = bounce_root(position.z, velocity.z, height, 1.0).unwrap_or(0.0);
    let ceiling_time =
        bounce_root(position.z, velocity.z, 2044.0 - height, -1.0).unwrap_or(f32::INFINITY);

    floor_time.min(wall_x_time).min(wall_y_time).min(ceiling_time)
}

fn role_air_position(position: Vector, velocity: Vector, time: f32) -> Vector {
    position + velocity * time + Vector::new(0.0, 0.0, -325.0) * time.powi(2)
}

fn car_ball_facing_angle(car: CarState, ball: Vector) -> f32 {
    let to_ball = ball - car.location;
    let forward = car.forward();
    if to_ball.magnitude() < 1.0 || forward.magnitude() < 0.001 {
        return std::f32::consts::PI;
    }
    forward.normalize().dot(to_ball.normalize()).clamp(-1.0, 1.0).acos()
}

fn boost_on_route_to_ball(
    car: CarState,
    ball: BallState,
    pads: &[BoostPadState],
    foes: &[CarState],
    side: f32,
) -> Option<BoostPadState> {
    if foes.iter().any(|foe| foe.location.distance(ball.location) < 700.0) {
        return None;
    }

    let dx = ball.location.x - car.location.x;
    let dy = ball.location.y - car.location.y;
    let length_squared = dx * dx + dy * dy;
    let ball_depth = ball.location.y * side;
    let own_corner = ball_depth > 640.0 && ball.location.x.abs() > 1200.0 && car.boost < 70.0;

    let mut best: Option<BoostPadState> = None;
    let mut best_score = f32::INFINITY;
    for pad in pads.iter().copied().filter(|pad| pad.active) {
        let car_distance = car.location.flat_distance(pad.location);
        let pad_to_ball = pad.location.flat_distance(ball.location);
        let enemy_near_pad = foes
            .iter()
            .any(|foe| foe.location.flat_distance(pad.location) < 1100.0);
        let own_corner_pad = pad.large
            && pad.location.y * side > 2500.0
            && pad.location.x.abs() > 2200.0;

        if own_corner && own_corner_pad && !enemy_near_pad {
            let max_distance = if car.boost < 35.0 {
                2800.0
            } else if car.boost < 55.0 {
                2100.0
            } else {
                1500.0
            };
            if car_distance <= max_distance && pad_to_ball < 3200.0 {
                let score = car_distance + 0.35 * pad_to_ball;
                if score < best_score {
                    best_score = score;
                    best = Some(pad);
                }
            }
        }

        if length_squared > 1.0 {
            let bx = pad.location.x - car.location.x;
            let by = pad.location.y - car.location.y;
            let progress = (bx * dx + by * dy) / length_squared;
            if !(0.05 < progress && progress < 0.92) {
                continue;
            }
            let nearest_x = car.location.x + progress * dx;
            let nearest_y = car.location.y + progress * dy;
            let lateral_distance =
                ((pad.location.x - nearest_x).powi(2) + (pad.location.y - nearest_y).powi(2)).sqrt();
            if lateral_distance > 350.0 || car_distance > 1400.0 {
                continue;
            }
            let score = car_distance + lateral_distance * 1.5;
            if score < best_score {
                best_score = score;
                best = Some(pad);
            }
        }
    }
    best
}

fn kickoff_boost_target(
    car: CarState,
    side: f32,
    pads: &[BoostPadState],
) -> Option<Vector> {
    let mut candidates: Vec<BoostPadState> = pads
        .iter()
        .copied()
        .filter(|pad| pad.active && pad.large && pad.location.y * side > 2500.0)
        .collect();
    if candidates.is_empty() {
        candidates = pads
            .iter()
            .copied()
            .filter(|pad| pad.active && pad.large)
            .collect();
    }
    candidates
        .into_iter()
        .min_by(|a, b| {
            let score = |pad: BoostPadState| {
                (
                    pad.location.distance(car.location)
                        + 0.35 * (pad.location.x - car.location.x).abs(),
                    pad.location.x * car.location.x < 0.0,
                )
            };
            let sa = score(*a);
            let sb = score(*b);
            sa.0.total_cmp(&sb.0).then_with(|| sa.1.cmp(&sb.1))
        })
        .map(|pad| pad.location)
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


/// Port of tools.select_ball_contact: select centre/left/right/high/low
/// contact using the car's current approach and the intended shot direction.
fn select_ball_contact(car: CarState, ball: Vector, targets: ShotTargets) -> (Vector, Vector, &'static str, f32) {
    let target_center=(targets.left+targets.right)*0.5;
    let mut shot_direction=(target_center-ball).flatten().normalize();
    if shot_direction.magnitude()<0.001 {
        shot_direction=Vector::new(0.0,team_side(car.team),0.0);
    }
    let lateral=Vector::new(-shot_direction.y,shot_direction.x,0.0).normalize();
    let car_from_ball=(car.location-ball).normalize();
    let candidates=[
        ("centre",-shot_direction),
        ("left",(-shot_direction+lateral*0.55).normalize()),
        ("right",(-shot_direction-lateral*0.55).normalize()),
        ("high",(-shot_direction+Vector::new(0.0,0.0,0.30)).normalize()),
        ("low",(-shot_direction-Vector::new(0.0,0.0,0.30)).normalize()),
    ];
    let (label,normal,quality)=candidates.into_iter().map(|(label,normal)|{
        let approach_alignment=car_from_ball.dot(normal);
        let shot_alignment=(-normal).dot(shot_direction);
        (label,normal,approach_alignment*0.75+shot_alignment*0.25)
    }).max_by(|a,b|a.2.total_cmp(&b.2)).unwrap_or(("centre",-shot_direction,0.0));
    (ball+normal*BALL_RADIUS,ball+normal*120.0,label,quality)
}

fn select_side_shot(car:CarState,candidate:ShotCandidate,targets:ShotTargets)->Option<SideShotPlan> {
    if !candidate.can_jump || !matches!(candidate.shot_type,ShotType::Ground|ShotType::Jump) {return None;}
    let ball=candidate.ball;
    let velocity=candidate.ball_velocity;
    let goal_center=(targets.left+targets.right)*0.5;
    let shot_direction=(goal_center-ball).flatten().normalize();
    if shot_direction.magnitude()<0.001 {return None;}
    let left_of_goal=Vector::new(-shot_direction.y,shot_direction.x,0.0).normalize();
    let half_goal_width=(targets.left.x-targets.right.x).abs()*0.5;
    let time_to_goal_line=if velocity.y.abs()>150.0 {(goal_center.y-ball.y)/velocity.y}else{-1.0};
    let projected_goal_x=if time_to_goal_line>0.0 && time_to_goal_line<=2.5 {
        Some(ball.x+velocity.x*time_to_goal_line)
    } else {None};
    let rolling_toward_goal=ball.z<=420.0
        && velocity.flatten().magnitude()>=300.0
        && time_to_goal_line>0.0 && time_to_goal_line<=2.5;
    let projected_x=projected_goal_x?;
    if !rolling_toward_goal {return None;}
    let projected_lateral=(Vector::new(projected_x,goal_center.y,0.0)-goal_center).dot(left_of_goal);
    let approach_side=if projected_lateral>half_goal_width-92.75 {
        -1.0
    } else if projected_lateral< -half_goal_width+92.75 {
        1.0
    } else {return None;};
    let left_of_shot=Vector::new(-shot_direction.y,shot_direction.x,0.0).normalize();
    let lateral_distance=(car.location.flat_distance(ball)*0.22).clamp(350.0,650.0);
    let side_setup=ball-left_of_shot*lateral_distance*approach_side;
    let side_setup_distance=car.location.flat_distance(side_setup);
    let side_setup_speed=(car.velocity.flatten().magnitude()+500.0).clamp(1000.0,1700.0);
    let local=car.local_location(side_setup);
    let turn_angle=local.y.atan2(local.x).abs();
    let setup_time=side_setup_distance/side_setup_speed+turn_angle*0.30;
    if candidate.time>setup_time+0.35 && side_setup.x.abs()<3900.0 && side_setup.y.abs()<4900.0 {
        Some(SideShotPlan {
            side_sign:approach_side,
            lateral_distance:car.location.flat_distance(ball)*0.5,
            shot_direction,
        })
    } else {None}
}

fn candidate_to_intercept(car:CarState,candidate:ShotCandidate,targets:ShotTargets)->GroundIntercept {
    let parser_has_targets =
        (candidate.parser.targets.left - candidate.parser.targets.right).magnitude() > 1.0;
    let untargeted = !parser_has_targets && !candidate.targeted;
    let selected_targets = if parser_has_targets {
        candidate.parser.targets
    } else if untargeted {
        ShotTargets::default()
    } else {
        targets
    };

    let contact_standoff=BALL_RADIUS+car.hitbox.length*0.5;
    let (mut contact_point,mut approach,mut contact_face,contact_quality,shot_direction,side_shot) =
        if untargeted {
            // fast_intercept_finder constructs routines without a target
            // tuple. Python therefore uses the opponent-goal ray and a
            // centred front contact, not select_ball_contact's side-face
            // candidates or a forced goal-target flank route.
            let goal_direction=(foe_goal(team_side(car.team))-candidate.ball)
                .flatten().normalize();
            let direction=if goal_direction.magnitude()>0.001 {
                goal_direction
            } else {
                Vector::new(0.0,-team_side(car.team),0.0)
            };
            (
                candidate.ball-direction*BALL_RADIUS,
                candidate.ball-direction*contact_standoff,
                "centre",
                1.0,
                direction,
                None,
            )
        } else {
            let (point,_,face,quality)=select_ball_contact(car,candidate.ball,selected_targets);
            let normal=(point-candidate.ball).normalize();
            let approach=candidate.ball+normal*contact_standoff;
            let side=select_side_shot(car,candidate,selected_targets);
            let direction=side.map_or(candidate.shot_direction,|s|s.shot_direction);
            let (approach,point,face)=if let Some(side)=side {
                let lateral=Vector::new(-side.shot_direction.y,side.shot_direction.x,0.0).normalize();
                (
                    candidate.ball-lateral*side.lateral_distance*side.side_sign,
                    candidate.ball-lateral*BALL_RADIUS*side.side_sign,
                    if side.side_sign<0.0 {"left"} else {"right"},
                )
            } else {
                (approach,point,face)
            };
            (point,approach,face,quality,direction,side)
        };
    GroundIntercept {
        time:candidate.time, ball:candidate.ball, ball_velocity:candidate.ball_velocity,
        approach, shot_direction, contact_quality,
        contact_point, contact_face, shot_type:candidate.shot_type, side_shot, targets:selected_targets,
    }
}

/// Volt's relative goal aim: use a higher central target while the ball
/// remains in our half, then return to the normal ground-shot height.
/// Python double_jump._select_patient_intercept, including its goal-mouth
/// angle delay. Prediction times are relative to the current game tick.
fn patient_double_jump_intercept(
    car: CarState,
    prediction: &BallTrajectory,
    targets: ShotTargets,
    initial_time: f32,
    goal_y: f32,
) -> Option<(f32, Vector)> {
    let slices = &prediction.slices;
    if slices.is_empty() {
        return None;
    }

    let start_time = initial_time.max(1.0 / 120.0);
    let end_time = 5.0_f32.min(slices.last()?.time);

    let clamped_approach = |ball: Vector| {
        let toward_ball = (ball - car.location).normalize();
        let left = (targets.left - ball).normalize();
        let right = (targets.right - ball).normalize();
        if toward_ball.magnitude() < 1.0 {
            left
        } else {
            toward_ball.clamp_2d(left, right)
        }
    };

    let reachable = |ball: Vector, direction: Vector| {
        let contact_z = ball.z - direction.z * 92.75;
        let required_height = (contact_z - car.location.z).max(0.0);
        grid_jump_time(required_height, true).is_some()
    };

    let first = slices.iter().position(|slice| slice.time >= start_time)?;
    let mut selected = None;
    for (index, slice) in slices.iter().enumerate().skip(first) {
        if slice.time > end_time {
            break;
        }
        let direction = clamped_approach(slice.location);
        if reachable(slice.location, direction) {
            selected = Some(index);
            break;
        }
    }
    let mut index = selected?;
    let selected_slice = slices[index];
    let selected_ball = selected_slice.location;
    let selected_velocity = selected_slice.velocity;
    let x_sign = if selected_ball.x < 0.0 {
        -1.0
    } else if selected_ball.x > 0.0 {
        1.0
    } else {
        0.0
    };
    let needs_alignment_delay = selected_ball.x.abs() < 3000.0
        && selected_ball.x.abs() - 890.0 + 92.75 * 2.0_f32.sqrt()
            > (selected_ball.y - goal_y).abs()
        && selected_velocity.x * x_sign < 0.0;

    if needs_alignment_delay {
        for later in (index + 1)..slices.len() {
            let slice = slices[later];
            if slice.time > end_time {
                break;
            }
            let direction = clamped_approach(slice.location);
            let aligned = slice.location.x.abs() - 890.0
                + 92.75 * 2.0_f32.sqrt()
                <= (slice.location.y - goal_y).abs();
            if aligned && reachable(slice.location, direction) {
                index = later;
                break;
            }
        }
    }

    let result = slices[index];
    Some((result.time, result.location))
}

fn relative_goal_aim_point(car:CarState,ball:Vector,targets:ShotTargets)->Vector {
    let mut center=if (targets.left-targets.right).magnitude()>1.0 {
        (targets.left+targets.right)*0.5
    } else {
        foe_goal(team_side(car.team))
    };
    center.z=300.0;
    if ball.y*team_side(car.team)<=0.0 {
        center.z=470.0;
        return center;
    }
    if !car.airborne {center.z+=10.0;} else {center.z-=90.0;}
    center
}


fn party_cannon_jump_front_contact(
    car: CarState,
    ball: Vector,
    shot_direction: Vector,
    time_remaining: f32,
    gravity: Vector,
    jumping: bool,
    previous_label: Option<&'static str>,
) -> Option<(Vector, &'static str)> {
    let shot_forward = shot_direction.flatten().normalize();
    let car_forward = car.forward().normalize();
    let mut car_up = car.up().normalize();
    let car_right = car_forward.cross(car_up).normalize();
    if shot_forward.magnitude() <= 0.001
        || car_forward.magnitude() <= 0.001
        || car_up.magnitude() <= 0.001
        || car_right.magnitude() <= 0.001
    {
        return None;
    }
    car_up = car_right.cross(car_forward).normalize();

    let orientation_horizon = if jumping {
        time_remaining.max(0.0).min(0.18)
    } else {
        0.0
    };
    // This Party Cannon contact prediction reads raw RLBot physics
    // angular_velocity, which is world-space. CarState stores the VirxERLU
    // localized value for defaultPD, so transform it back before Rodrigues'
    // world-space orientation extrapolation.
    let omega = car.forward() * car.angular_velocity.x
        + car.right() * car.angular_velocity.y
        + car.up() * car.angular_velocity.z;
    let angular_speed = omega.magnitude();
    let mut impact_forward = car_forward;
    let mut impact_up = car_up;
    if angular_speed > 0.05 && orientation_horizon > 0.0 {
        let axis = omega / angular_speed;
        let angle = (angular_speed * orientation_horizon).min(0.65);
        let cosine = angle.cos();
        let sine = angle.sin();
        impact_forward = (
            car_forward * cosine
                + axis.cross(car_forward) * sine
                + axis * (axis.dot(car_forward) * (1.0 - cosine))
        ).normalize();
        impact_up = (
            car_up * cosine
                + axis.cross(car_up) * sine
                + axis * (axis.dot(car_up) * (1.0 - cosine))
        ).normalize();
    }

    let impact_right = impact_up.cross(impact_forward).normalize();
    impact_up = impact_forward.cross(impact_right).normalize();
    let half_length = car.hitbox.length * 0.5;
    let half_width = car.hitbox.width * 0.5;
    let half_height = car.hitbox.height * 0.5;
    let world_hitbox_offset = impact_forward * car.hitbox.offset.x
        + impact_right * car.hitbox.offset.y
        + impact_up * car.hitbox.offset.z;
    let candidates: [(&str, f32, f32); 7] = [
        ("centre", 0.0, 0.0),
        ("top_left", -half_width * 0.82, half_height * 0.82),
        ("top_right", half_width * 0.82, half_height * 0.82),
        ("left", -half_width * 0.88, 0.0),
        ("right", half_width * 0.88, 0.0),
        ("top", 0.0, half_height * 0.88),
        ("bottom", 0.0, -half_height * 0.70),
    ];

    let impact_horizon = time_remaining.max(0.0).min(0.30);
    let predicted_car = car.location
        + car.velocity * impact_horizon
        + gravity * (0.5 * impact_horizon * impact_horizon);
    let mut scored: Vec<(f32, &'static str, Vector)> = Vec::with_capacity(candidates.len());
    for (label, lateral_offset, vertical_offset) in candidates {
        let front_point_offset = impact_forward * half_length
            + impact_right * lateral_offset
            + impact_up * vertical_offset
            + world_hitbox_offset;
        let predicted_contact_point = predicted_car + front_point_offset;
        let incoming_normal = (ball - predicted_contact_point).normalize();
        if incoming_normal.magnitude() < 0.001 {
            continue;
        }
        let shot_alignment = incoming_normal.dot(shot_forward);
        let front_alignment = incoming_normal.dot(impact_forward);
        let required_car_centre =
            ball - shot_forward * 92.75 - front_point_offset;
        let reach_cost = required_car_centre.distance(predicted_car);
        let score = shot_alignment
            + front_alignment.clamp(-1.0, 1.0) * 0.12
            - reach_cost * 0.00020;
        scored.push((score, label, required_car_centre));
    }
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut selected = scored[0];
    if let Some(previous) = previous_label {
        if let Some(previous_candidate) = scored.iter().copied().find(|entry| entry.1 == previous) {
            if previous_candidate.0 >= selected.0 - 0.06 {
                selected = previous_candidate;
            }
        }
    }
    Some((selected.2, selected.1))
}

fn contact_approach_from_label(ball:Vector,shot:Vector,label:&str,distance:f32)->Option<Vector> {
    let shot=shot.flatten().normalize();
    if shot.magnitude()<0.001 {return None;}
    let lateral=Vector::new(-shot.y,shot.x,0.0).normalize();
    let normal=match label {
        "centre"=>-shot,
        "left"=>(-shot+lateral*0.55).normalize(),
        "right"=>(-shot-lateral*0.55).normalize(),
        "high"=>(-shot+Vector::new(0.0,0.0,0.30)).normalize(),
        "low"=>(-shot-Vector::new(0.0,0.0,0.30)).normalize(),
        _=>return None,
    };
    Some(ball+normal*distance.max(120.0))
}

fn shot_aim_point(ball: Vector, side: f32, airborne: bool) -> Vector {
    let goal = foe_goal(side);
    let height = if ball.y * side <= 0.0 { 470.0 } else if airborne { 210.0 } else { 310.0 };
    Vector::new(goal.x, goal.y, height)
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
    gravity_z: f32,
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

        let jump_time = get_jump_time(height, car.velocity.z, gravity_z).time as f32;
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
                shot_type: ShotType::Jump,
                side_shot: None,
                targets: ShotTargets { left: Vector::ZERO, right: Vector::ZERO },
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
            shot_type: ShotType::Ground,
            side_shot: None,
            targets: ShotTargets { left: Vector::ZERO, right: Vector::ZERO },
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
    fn elected_three_player_cover_yields_to_a_comparable_teammate_shot() {
        let me = car(0, 0, Vector::new(0.0, -2500.0, 17.0));
        let attacker = car(1, 0, Vector::new(0.0, -1000.0, 17.0));
        let transition = car(2, 0, Vector::new(1000.0, -1800.0, 17.0));
        let ball = BallState {
            location: Vector::new(0.0, -500.0, 100.0),
            velocity: Vector::ZERO,
        };
        let role = TeamRoles { challenger: 1, goalie: 0, transition: Some(2) };
        let mut messages = HashMap::new();
        messages.insert(1, TeamplayMessage {
            status: "SHOT_FOUND".to_owned(),
            shot_type: "SHOT_FOUND".to_owned(),
            score: 80.0,
            contact_time: 1.2,
            received_at: 0.9,
            cover_owner: Some(0),
            cover_released: false,
            receive_x: None,
            receive_y: None,
            receive_dir_x: None,
            receive_dir_y: None,
            contact_x: None,
            contact_y: None,
        });
        assert!(!should_commit_to_shot(
            me, &[me, attacker, transition], ball, Some(role), false,
            75.0, 1.5, &messages, 1.0,
        ));
    }

    #[test]
    fn vulnerable_opponent_fallback_requires_close_ball_and_bad_facing() {
        let me = car(0, 0, Vector::new(0.0, -1000.0, 17.0));
        let ball = BallState {
            location: Vector::new(0.0, -500.0, 150.0),
            velocity: Vector::ZERO,
        };
        let mut foe = car(1, 1, Vector::new(0.0, -300.0, 17.0));
        foe.orientation = Matrix3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0);
        assert!(opponent_is_vulnerable(
            me, ball, &[me, foe], Vector::new(0.0, -5120.0, 300.0),
        ));

        let far_me = car(0, 0, Vector::new(0.0, -3000.0, 17.0));
        assert!(!opponent_is_vulnerable(
            far_me, ball, &[far_me, foe], Vector::new(0.0, -5120.0, 300.0),
        ));
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
            -650.0,
        ).unwrap();

        assert_eq!(shot.time, 1.50);
        assert!(shot.ball.z > 190.0);
        assert_eq!(shot.approach.z, shot.ball.z);
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
            -650.0,
        ).unwrap();
        assert!(shot.contact_quality.is_finite());
        assert!(shot.contact_point.distance(shot.ball) <= BALL_RADIUS + 0.01);
        assert!(!shot.contact_face.is_empty());
    }



    #[test]
    fn side_shot_requires_a_projected_goal_line_miss_and_enough_setup_time() {
        let mut me = car(0, 0, Vector::new(0.0, 2500.0, 17.0));
        me.velocity = Vector::new(0.0, 1200.0, 0.0);
        me.orientation = Matrix3::new(0.0, 0.0, 0.0);
        let candidate = ShotCandidate {
            time: 2.0,
            ball: Vector::new(1000.0, 3500.0, 100.0),
            ball_velocity: Vector::new(200.0, 800.0, 0.0),
            approach: Vector::ZERO,
            shot_direction: Vector::new(0.0, 1.0, 0.0),
            shot_type: ShotType::Ground,
            targeted: true,
            can_jump: true,
            fast: false,
            score: 0.0,
            parser: Default::default(),
        };
        let targets = ShotTargets {
            left: Vector::new(-892.0, 5120.0, 300.0),
            right: Vector::new(892.0, 5120.0, 300.0),
        };

        let side = select_side_shot(me, candidate, targets).unwrap();
        assert_eq!(side.side_sign, 1.0);
        assert!((side.lateral_distance - me.location.flat_distance(candidate.ball) * 0.5).abs() < 0.01);

        let centred = ShotCandidate {
            ball: Vector::new(0.0, 3500.0, 100.0),
            ball_velocity: Vector::new(0.0, 800.0, 0.0),
            ..candidate
        };
        assert!(select_side_shot(me, centred, targets).is_none());
    }

    #[test]
    fn grid_double_jump_holds_first_press_releases_then_sends_second_press() {
        let mut me = car(0, 0, Vector::ZERO);
        me.airborne = true;
        me.doublejumped = false;
        let mut brain = BotBrain::new();
        let state = DoubleJumpState {
            started_at: 0.0,
            intercept_time: 1.0,
            ball: Vector::new(0.0, 300.0, 300.0),
            offset_target: Vector::new(0.0, 250.0, 250.0),
            shot_direction: Vector::new(0.0, 1.0, 0.0),
            jump_time: 0.5,
            jump_frames: 0,
            second_jump_sent: false,
            prev_jump: false,
                targets: ShotTargets { left: Vector::ZERO, right: Vector::ZERO },
        };

        let first = brain.run_double_jump(me, state, 0.10);
        assert!(first.jump);

        let second_state = brain.double_jump_shot.unwrap();
        let release = brain.run_double_jump(me, second_state, 0.21);
        assert!(!release.jump);

        let third_state = brain.double_jump_shot.unwrap();
        let second_press = brain.run_double_jump(me, third_state, 0.22);
        assert!(second_press.jump);
        assert!(brain.double_jump_shot.unwrap().second_jump_sent);
    }

    #[test]
    fn jump_shot_tracks_prediction_and_accelerates_before_launch() {
        let mut brain = BotBrain::new();
        let me = car(0, 0, Vector::new(0.0, 0.0, 17.0));
        let state = JumpShotState {
            started_at: 1.0,
            intercept_time: 2.5,
            ball: Vector::new(400.0, 0.0, 93.0),
            ball_velocity: Vector::ZERO,
            approach: Vector::new(248.0, 0.0, 93.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            contact_face: "centre",
            contact_face_label: None,
            goal_aim_point: None,
            needed_jump_time: 0.2,
            jumping: false,
            jump_time: None,
            quick: false,
            counter: 0,
            patience_applied: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
            side_mode: false,
            side_sign: 1.0,
            side_lateral_distance: 0.0,
            targets: ShotTargets {
                left: Vector::new(5000.0, -500.0, 300.0),
                right: Vector::new(5000.0, 500.0, 300.0),
            },
        };
        let future = BallSlice {
            time: 1.5,
            location: Vector::new(650.0, 0.0, 93.0),
            velocity: Vector::ZERO,
        };
        let prediction = BallTrajectory { slices: vec![future; 200] };
        let controller = brain.run_jump_shot(
            me,
            BallState { location: future.location, velocity: future.velocity },
            &prediction,
            state,
            1.0,
        );

        assert!(controller.throttle > 0.0, "the car must accelerate toward the predicted contact");
        assert!(!controller.jump, "the routine must drive before its launch window");
        let committed = brain.jump_shot.expect("jump shot remains committed");
        assert!(!committed.jumping);
        assert!((committed.ball.x - 650.0).abs() < 0.01);
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
            ball: Vector::new(300.0, 0.0, 0.0),
            ball_velocity: Vector::ZERO,
            approach: Vector::new(200.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            contact_face: "centre",
            contact_face_label: None,
                goal_aim_point: Some(Vector::new(5120.0, 512.0, 300.0)),
            needed_jump_time: 0.2,
            jumping: true,
            jump_time: Some(1.0),
            quick: true,
            counter: 3,
            patience_applied: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
                    side_mode: false,
                    side_sign: 1.0,
                    side_lateral_distance: 0.0,
                    targets: ShotTargets { left: Vector::ZERO, right: Vector::ZERO },
        };

        let controller = brain.run_jump_shot(me, BallState { location: state.ball, velocity: state.ball_velocity }, &BallTrajectory { slices: Vec::new() }, state, 1.10);
        assert!(controller.jump);
        let committed = brain.jump_shot.unwrap();
        assert!(committed.dodge_started_at.is_some());
        assert!((controller.pitch + committed.dodge_direction.x).abs() < 1e-6);
        assert!((controller.yaw - committed.dodge_direction.y).abs() < 1e-6);
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
            ball_velocity: Vector::ZERO,
            approach: Vector::new(1000.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            contact_face: "centre",
            contact_face_label: None,
                goal_aim_point: Some(Vector::new(5120.0, 512.0, 300.0)),
            needed_jump_time: 0.2,
            jumping: true,
            jump_time: Some(1.0),
            quick: false,
            counter: 0,
            patience_applied: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
                    side_mode: false,
                    side_sign: 1.0,
                    side_lateral_distance: 0.0,
                    targets: ShotTargets { left: Vector::ZERO, right: Vector::ZERO },
        };

        let controller = brain.run_jump_shot(me, BallState { location: state.ball, velocity: state.ball_velocity }, &BallTrajectory { slices: Vec::new() }, state, 1.10);
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
            ball_velocity: Vector::ZERO,
            approach: Vector::new(200.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 0.0, 0.0),
            contact_face: "centre",
            contact_face_label: None,
                goal_aim_point: Some(Vector::new(5120.0, 512.0, 300.0)),
            needed_jump_time: 0.2,
            jumping: true,
            jump_time: Some(1.0),
            quick: false,
            counter: 0,
            patience_applied: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
                    side_mode: false,
                    side_sign: 1.0,
                    side_lateral_distance: 0.0,
                    targets: ShotTargets { left: Vector::ZERO, right: Vector::ZERO },
        };

        let controller = brain.run_jump_shot(me, BallState { location: state.ball, velocity: state.ball_velocity }, &BallTrajectory { slices: Vec::new() }, state, 1.90);
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
            ball_velocity: Vector::ZERO,
            approach: Vector::new(200.0, 0.0, 0.0),
            shot_direction: Vector::new(1.0, 1.0, 0.0),
            contact_face: "centre",
            contact_face_label: None,
                goal_aim_point: Some(Vector::new(5120.0, 512.0, 300.0)),
            needed_jump_time: 0.2,
            jumping: true,
            jump_time: Some(1.0),
            quick: false,
            counter: 0,
            patience_applied: false,
            dodge_started_at: None,
            dodge_direction: Vector::new(1.0, 0.0, 0.0),
                    side_mode: false,
                    side_sign: 1.0,
                    side_lateral_distance: 0.0,
                    targets: ShotTargets { left: Vector::ZERO, right: Vector::ZERO },
        };

        let normal = brain.run_jump_shot(me, BallState { location: state.ball, velocity: state.ball_velocity }, &BallTrajectory { slices: Vec::new() }, state, 1.01);
        assert!(normal.jump);
        assert_eq!(normal.pitch, 1.0);
        assert!(normal.yaw.abs() > 0.0);
        assert!(normal.roll < 0.0);

        let quick = JumpShotState { quick: true, ..state };
        let fast = brain.run_jump_shot(me, BallState { location: quick.ball, velocity: quick.ball_velocity }, &BallTrajectory { slices: Vec::new() }, quick, 1.01);
        assert_ne!(fast.pitch, 1.0);
        assert_eq!(fast.roll, 0.0);
    }

    #[test]
    fn teamplay_receive_lead_uses_contact_to_goal_distance() {
        let me = car(0, 0, Vector::ZERO);
        let contact = Vector::new(1000.0, 0.0, 93.0);
        let aim = Vector::new(1000.0, 5120.0, 300.0);
        let (receive, direction, stored_contact) =
            receive_plan(me, contact, aim).expect("valid receive lane");
        assert_eq!(stored_contact, contact);
        assert!((direction.y - 1.0).abs() < 1e-6);
        assert!((receive.y - 1050.0).abs() < 1e-4);
    }

    #[test]
    fn route_boost_selects_a_pad_on_the_car_ball_lane() {
        let mut me = car(0, 0, Vector::ZERO);
        me.boost = 20.0;
        let ball = BallState {
            location: Vector::new(3000.0, 0.0, 93.0),
            velocity: Vector::ZERO,
        };
        let pad = BoostPadState {
            index: 0,
            location: Vector::new(1000.0, 100.0, 73.0),
            active: true,
            large: false,
        };
        let selected = boost_on_route_to_ball(me, ball, &[pad], &[], 1.0)
            .expect("active route pad should be selected");
        assert_eq!(selected.location, pad.location);
    }

    #[test]
    fn route_boost_is_blocked_by_an_enemy_near_the_ball() {
        let mut me = car(0, 0, Vector::ZERO);
        me.boost = 20.0;
        let ball = BallState {
            location: Vector::new(3000.0, 0.0, 93.0),
            velocity: Vector::ZERO,
        };
        let pad = BoostPadState {
            index: 0,
            location: Vector::new(1000.0, 100.0, 73.0),
            active: true,
            large: false,
        };
        let foe = car(1, 1, Vector::new(2800.0, 0.0, 17.0));
        assert!(boost_on_route_to_ball(me, ball, &[pad], &[foe], 1.0).is_none());
    }

    #[test]
    fn jump_shot_hitbox_target_uses_python_ball_radius_and_hitbox_offset() {
        let mut me = car(0, 0, Vector::ZERO);
        me.hitbox.offset = Vector::new(10.0, 5.0, 3.0);
        let ball = Vector::new(500.0, 0.0, 0.0);
        let direction = Vector::new(1.0, 0.0, 0.0);
        let target = ball
            - direction * (92.75 + me.hitbox.length * 0.5)
            - me.forward() * 10.0
            - me.up().cross(me.forward()) * 5.0
            - me.up() * 3.0;
        let standoff = 92.75 + me.hitbox.length * 0.5;
        let nominal = ball - direction * standoff;
        let offset = me.forward() * me.hitbox.offset.x
            + me.up().cross(me.forward()) * me.hitbox.offset.y
            + me.up() * me.hitbox.offset.z;
        assert!((target - (nominal - offset)).magnitude() < 1e-4);
    }

    #[test]
    fn shot_patience_waits_for_goal_alignment_when_the_car_can_reach_it() {
        let mut me = car(0, 0, Vector::new(1000.0, 4500.0, 17.0));
        me.velocity = Vector::new(1500.0, 0.0, 0.0);
        let mut slices: Vec<BallSlice> = (0..100)
            .map(|i| BallSlice {
                time: i as f32 / 120.0,
                location: Vector::ZERO,
                velocity: Vector::ZERO,
            })
            .collect();
        slices[59] = BallSlice {
            time: 59.0 / 120.0,
            location: Vector::new(1000.0, 5000.0, 200.0),
            velocity: Vector::new(-100.0, 0.0, 0.0),
        };
        slices[80] = BallSlice {
            time: 80.0 / 120.0,
            location: Vector::new(1000.0, 4000.0, 100.0),
            velocity: Vector::ZERO,
        };

        let adjusted = shot_patience_adjustment(me, &slices, 0.0, 0.5, 5120.0);
        let (new_time, selected) = adjusted.expect("later aligned slice should be reachable");
        assert!((new_time - 81.0 / 120.0).abs() < 1e-6);
        assert_eq!(selected.location, Vector::new(1000.0, 4000.0, 100.0));
    }

    #[test]
    fn ground_flip_releases_jump_before_directional_dodge() {
        let mut brain = BotBrain::new();
        let flip = FlipState {
            started_at: 1.0,
            direction: Vector::new(0.8, 0.6, 0.0),
            cancel: false,
        };

        assert!(brain.run_ground_flip(None, 1.05, flip).jump);
        assert!(!brain.run_ground_flip(None, 1.08, flip).jump);

        let dodge = brain.run_ground_flip(None, 1.12, flip);
        assert!(dodge.jump);
        assert!((dodge.pitch + 0.8).abs() < 1e-6);
        assert!((dodge.yaw - 0.6).abs() < 1e-6);
    }

    #[test]
    fn two_v_two_priority_yields_when_teammate_is_clearly_closer_and_facing_ball() {
        let me = car(0, 0, Vector::new(-1000.0, -1000.0, 0.0));
        let teammate = car(1, 0, Vector::new(-100.0, -100.0, 0.0));
        let opponent = car(2, 1, Vector::new(1500.0, 0.0, 0.0));
        let ball = BallState {
            location: Vector::ZERO,
            velocity: Vector::ZERO,
        };

        assert!(!should_prioritize_ball(
            me,
            &[me, teammate, opponent],
            ball,
            -1.0,
            5.0,
            -999.0,
            false,
            false,
            None,
        ));
    }

    #[test]
    fn two_v_two_priority_keeps_emergency_defence_over_teammate_awareness() {
        let me = car(0, 0, Vector::new(-1000.0, -1000.0, 0.0));
        let teammate = car(1, 0, Vector::new(-100.0, -100.0, 0.0));
        let opponent = car(2, 1, Vector::new(1500.0, 0.0, 0.0));
        let ball = BallState {
            location: Vector::new(0.0, -4500.0, 200.0),
            velocity: Vector::new(0.0, -500.0, 0.0),
        };

        assert!(should_prioritize_ball(
            me,
            &[me, teammate, opponent],
            ball,
            -1.0,
            5.0,
            -999.0,
            true,
            false,
            None,
        ));
    }

    #[test]
    fn kickoff_boost_selection_prefers_own_half_full_pads() {
        let me = car(0, 0, Vector::new(0.0, -3000.0, 0.0));
        let pads = [
            BoostPadState {
                index: 0,
                location: Vector::new(3500.0, -4000.0, 0.0),
                large: true,
                active: true,
            },
            BoostPadState {
                index: 1,
                location: Vector::new(-3500.0, 4000.0, 0.0),
                large: true,
                active: true,
            },
            BoostPadState {
                index: 2,
                location: Vector::new(0.0, -2000.0, 0.0),
                large: false,
                active: true,
            },
        ];

        assert_eq!(
            kickoff_boost_target(me, -1.0, &pads),
            Some(pads[0].location)
        );
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
            ShotTargets::default(),
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
        ShotTargets::default(),
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
            ShotTargets::default(),
        );

        assert_ne!(plan.decision, Decision::KickoffKicker);
    }

    #[test]
    fn kickoff_dodge_uses_python_jump_release_dodge_windows() {
        let mut brain = BotBrain::new();
        let state = KickoffDodgeState {
            started_at: 0.0,
            direction: Vector::new(1.0, 0.0, 0.0),
        };
        assert!(brain.run_kickoff_dodge(state, 0.02).jump);
        let release = brain.run_kickoff_dodge(state, 0.06);
        assert!(!release.jump);
        let dodge = brain.run_kickoff_dodge(state, 0.10);
        assert!(dodge.jump);
        assert_eq!(dodge.pitch, -1.0);
        let done = brain.run_kickoff_dodge(state, 0.16);
        assert!(!done.jump);
        assert!(brain.kickoff_complete);
        assert!(brain.kickoff_dodge.is_none());
    }

    #[test]
    fn speedflip_applies_the_python_diagonal_dodge_and_cancel_sequence() {
        let mut brain = BotBrain::new();
        let mut me = car(0, 0, Vector::ZERO);
        me.boost = 50.0;
        let state = SpeedFlipState {
            step_start: 0.0,
            step_index: 3,
            direction: 1.0,
            use_boost: true,
        };
        let dodge = brain.run_speed_flip(me, state, 0.07);
        assert!(dodge.jump);
        assert_eq!(dodge.pitch, -1.0);
        assert_eq!(dodge.yaw, 1.0);
        assert!(dodge.boost);

        let cancel_state = brain.speed_flip.unwrap();
        let cancel = brain.run_speed_flip(me, cancel_state, 0.09);
        assert_eq!(cancel.pitch, 1.0);
        assert!(cancel.boost);
    }

    #[test]
    fn half_flip_releases_jump_then_cancels_backflip() {
        let mut brain = BotBrain::new();
        let me = car(0, 0, Vector::ZERO);
        let state = HalfFlipState {
            started_at: 0.0,
            release_frames: 0,
            flip_started_at: None,
        };
        let initial = brain.run_half_flip(me, state, 0.02, -650.0);
        assert!(initial.jump);
        assert_eq!(initial.pitch, 1.0);

        let release_state = brain.half_flip.unwrap();
        let release = brain.run_half_flip(me, release_state, 0.08, -650.0);
        assert!(!release.jump);

        let mut cancel_state = brain.half_flip.unwrap();
        cancel_state.release_frames = 3;
        let cancel = brain.run_half_flip(me, cancel_state, 0.25, -650.0);
        assert!(cancel.jump);
        assert_eq!(cancel.pitch, -1.0);
        assert!(brain.half_flip.unwrap().flip_started_at.is_some());
    }

}
