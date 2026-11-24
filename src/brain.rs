//! Main decision engine.
//!
//! This is a new modular decision layer: the role planner chooses who should
//! challenge, the Volt shot planner evaluates reachable contacts, the strike
//! executor runs the timed mechanics, and off-ball roles navigate separately.
//! Packet decoding and RLBot v5 I/O remain outside this module.

use std::collections::HashMap;

use crate::{
    aerial::{find_aerial_candidate, AerialCandidate, AerialShot},
    control::{default_drive, DriveInput},
    defense::GridDefense,
    drive::{drive_to, recover_in_air, DriveRequest},
    math::Vector,
    planning::{
        find_wall_drive, wall_approach_geometry, WallDriveMode, WallDrivePlan,
    },
    prediction::{BallSlice, BallTrajectory},
    routines::{KickoffExecutor, StrikeExecutor},
    shot_parser::{ParsedShot, ShotTargets},
    shot_physics::{get_jump_time, ShotType},
    shot_planner::{find_best_any_shot, find_best_pass_shot, find_best_shot, find_fast_intercept, ShotCandidate},
    state::{AgentFrame, TeamplayMessage},
    strategy::{
        build_plan, choose_boost_pad, choose_support_pad, cover_target, kickoff_cheat_target,
        kickoff_kicker, opponent_near_ball, support_target, Role,
    },
};

fn shot_type_name(kind: ShotType) -> &'static str {
    match kind {
        ShotType::Ground => "GROUND",
        ShotType::Jump => "JUMP",
        ShotType::DoubleJump => "DOUBLE_JUMP",
        ShotType::Aerial => "AERIAL",
    }
}

fn team_side(team: u32) -> f32 {
    if team == 0 { -1.0 } else { 1.0 }
}

fn transition_bump_lead_target(me: crate::state::CarState, foe: crate::state::CarState) -> Option<Vector> {
    let distance = me.location.flat_distance(foe.location);
    if !(260.0..=4200.0).contains(&distance) {
        return None;
    }
    let car_speed = me.velocity.flatten().magnitude();
    let lead_time = (distance / (car_speed + 650.0).max(1100.0)).clamp(0.12, 0.55);
    Some(foe.location + foe.velocity.flatten() * lead_time)
}

fn transition_bump_target(frame: &AgentFrame) -> Option<Vector> {
    let side = team_side(frame.me.team);
    let goal_direction = -side;

    frame.cars.iter()
        .filter(|car| car.team != frame.me.team && !car.demolished)
        .filter_map(|foe| {
            let distance_to_ball = foe.location.flat_distance(frame.ball.location);
            if distance_to_ball > 2300.0 {
                return None;
            }

            let ball_to_goal = (Vector::new(0.0, goal_direction * 5120.0, 0.0)
                - frame.ball.location).flatten().normalize();
            let ball_to_foe = (foe.location - frame.ball.location).flatten().normalize();
            // Only bump a defender who actually occupies the attack lane.
            let lane_alignment = ball_to_foe.dot(ball_to_goal);
            if lane_alignment < 0.25 || foe.velocity.flatten().dot(ball_to_goal) < -500.0 {
                return None;
            }

            let predicted = transition_bump_lead_target(frame.me, *foe)?;
            Some((distance_to_ball - lane_alignment * 350.0, predicted))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, target)| target)
}

fn teammate_has_priority_shot(
    messages: &HashMap<usize, TeamplayMessage>,
    now: f32,
    our_candidate: Option<ShotCandidate>,
) -> bool {
    let our_time = our_candidate.map(|shot| shot.time).unwrap_or(f32::INFINITY);
    messages.values().any(|message| {
        let age = now - message.received_at;
        if !(0.0..=0.75).contains(&age) {
            return false;
        }
        // contact_time is relative to the sender's last packet. Account for
        // the age of the received message before comparing two challengers.
        let contact_time = message.contact_time - age;
        if contact_time < -0.15 {
            return false;
        }
        let contact_time = contact_time.max(0.0);
        match message.status.as_str() {
            // A wall approach is also a concrete committed challenge, not a
            // speculative ANY_SHOT_FOUND message.
            "SHOT_FOUND" | "WALL_PASS" => contact_time <= our_time + 0.20,
            "ANY_SHOT_FOUND" => our_candidate.is_none()
                || contact_time + 0.08 < our_time,
            _ => false,
        }
    })
}

fn aerial_message(
    aerial: &AerialShot,
    now: f32,
    score: f32,
    cover_owner: Option<usize>,
) -> TeamplayMessage {
    TeamplayMessage {
        status: "SHOT_FOUND".to_owned(),
        shot_type: "AERIAL".to_owned(),
        score: (100.0 / (1.0 + score.max(0.0))).clamp(0.0, 100.0),
        contact_time: (aerial.intercept_time - now).clamp(0.0, 10.0),
        received_at: now,
        cover_owner,
        cover_released: false,
        receive_x: None,
        receive_y: None,
        receive_dir_x: None,
        receive_dir_y: None,
        contact_x: Some(aerial.ball.x),
        contact_y: Some(aerial.ball.y),
    }
}

fn wall_message(
    wall: WallDrivePlan,
    now: f32,
    cover_owner: Option<usize>,
) -> TeamplayMessage {
    TeamplayMessage {
        status: "WALL_PASS".to_owned(),
        shot_type: match wall.mode {
            WallDriveMode::SingleJump => "JUMP",
            WallDriveMode::DoubleJump => "DOUBLE_JUMP",
            WallDriveMode::Aerial => "AERIAL",
        }.to_owned(),
        score: 78.0,
        contact_time: wall.intercept_time.clamp(0.0, 10.0),
        received_at: now,
        cover_owner,
        cover_released: true,
        receive_x: None,
        receive_y: None,
        receive_dir_x: None,
        receive_dir_y: None,
        contact_x: Some(wall.ball_slice.location.x),
        contact_y: Some(wall.ball_slice.location.y),
    }
}

fn controller_from_drive(
    output: crate::control::DriveOutput,
    car: crate::state::CarState,
) -> rlbot::flat::ControllerState {
    let mut controller = rlbot::flat::ControllerState::default();
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

fn shot_message(
    status: &str,
    candidate: Option<ShotCandidate>,
    now: f32,
    cover_owner: Option<usize>,
    cover_released: bool,
) -> TeamplayMessage {
    let (shot_type, score, contact_time, contact_x, contact_y) =
        if let Some(shot) = candidate {
            (
                shot_type_name(shot.shot_type).to_owned(),
                (100.0 / (1.0 + shot.score.max(0.0))).clamp(0.0, 100.0),
                shot.time.clamp(0.0, 10.0),
                Some(shot.ball.x),
                Some(shot.ball.y),
            )
        } else {
            (status.to_owned(), 0.0, 10.0, None, None)
        };

    TeamplayMessage {
        status: status.to_owned(),
        shot_type,
        score,
        contact_time,
        received_at: now,
        cover_owner,
        cover_released,
        receive_x: None,
        receive_y: None,
        receive_dir_x: None,
        receive_dir_y: None,
        contact_x,
        contact_y,
    }
}

#[derive(Debug, Clone)]
pub struct BotBrain {
    teammate_comms: HashMap<usize, TeamplayMessage>,
    strike: StrikeExecutor,
    aerial: Option<AerialShot>,
    aerial_score: f32,
    defense: GridDefense,
    defense_flip: Option<(f32, Vector)>,
    last_flip_time: f32,
    kickoff: KickoffExecutor,
    last_message: TeamplayMessage,
    kickoff_contest_until: f32,
    was_kickoff: bool,
    last_support_target: Option<Vector>,
}

impl Default for BotBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl BotBrain {
    pub fn new() -> Self {
        Self {
            teammate_comms: HashMap::new(),
            strike: StrikeExecutor::new(),
            aerial: None,
            aerial_score: 0.0,
            defense: GridDefense::default(),
            defense_flip: None,
            last_flip_time: -999.0,
            kickoff: KickoffExecutor::default(),
            last_message: shot_message("NO_SHOT", None, 0.0, None, false),
            kickoff_contest_until: -999.0,
            was_kickoff: false,
            last_support_target: None,
        }
    }

    pub fn set_teammate_comms(&mut self, messages: &HashMap<usize, TeamplayMessage>) {
        self.teammate_comms.clone_from(messages);
    }

    pub fn teamplay_message(&self, now: f32) -> TeamplayMessage {
        let mut message = self.last_message.clone();
        message.received_at = now;
        message
    }

    fn run_cover_role(&mut self, frame: &AgentFrame) -> rlbot::flat::ControllerState {
        let car = frame.me;
        let now = frame.match_state.time;

        if let Some((started_at, direction)) = self.defense_flip {
            let elapsed = now - started_at;
            let mut controller = rlbot::flat::ControllerState::default();
            if elapsed < 0.075 {
                controller.jump = true;
                return controller;
            }
            if elapsed < 0.10 {
                controller.jump = false;
                return controller;
            }
            if elapsed < 0.9 {
                if self.last_flip_time < started_at {
                    self.last_flip_time = now;
                }
                controller.jump = true;
                controller.pitch = -direction.x;
                controller.yaw = direction.y;
                return controller;
            }
            self.defense_flip = None;
        }

        let side = team_side(car.team);
        let opponent_goal = Vector::new(0.0, -side * 5120.0, 300.0);
        let output = self.defense.run(
            car,
            &frame.cars,
            frame.ball,
            &frame.boost_pads,
            opponent_goal,
            true,
            false,
        );

        if output.start_flip {
            self.defense_flip = Some((now, Vector::new(1.0, 0.0, 0.0)));
            let mut controller = rlbot::flat::ControllerState::default();
            controller.jump = true;
            return controller;
        }

        if car.airborne
            && now - self.last_flip_time > 78.0 / 120.0
            && car.velocity.z < -10.0
        {
            if let Some(target) = output.recovery_target {
                return GridDefense::recovery_controller(
                    car,
                    target,
                    frame.match_state.gravity.z,
                );
            }
        }

        output.controller
    }

    fn run_wall_approach(
        &mut self,
        car: crate::state::CarState,
        plan: WallDrivePlan,
        targets: ShotTargets,
        frame: &AgentFrame,
        boost_accel: f32,
    ) -> rlbot::flat::ControllerState {
        let side = team_side(car.team);
        let foe_goal = Vector::new(0.0, -side * 5120.0, 300.0);
        let Some(geometry) = wall_approach_geometry(
            &car,
            plan.ball_slice,
            car.team,
            foe_goal,
            plan.mode,
            plan.intercept_time,
        ) else {
            return rlbot::flat::ControllerState::default();
        };

        let mut output = default_drive(
            &car,
            DriveInput {
                target_speed: geometry.speed_required,
                local_target: geometry.local_target,
                upside_down: false,
                up: geometry.surface_up.map(|up| car.local(up)),
                handbrake: false,
                boost_accel,
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
            output.throttle = Some(1.0);
        }

        let controller = controller_from_drive(output, car);
        let time_remaining = plan.intercept_time;
        if geometry.on_wall && geometry.distance < 320.0 && time_remaining < 1.65 {
            if plan.mode == WallDriveMode::Aerial {
                let slice = BallSlice {
                    time: time_remaining,
                    ..plan.ball_slice
                };
                let shot_target = (targets.left + targets.right) * 0.5;
                let (target_location, shot_direction) =
                    crate::aerial::party_cannon_contact_geometry(
                        car,
                        slice,
                        shot_target,
                        frame.match_state.gravity,
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
                self.aerial_score = candidate.score;
                self.aerial = Some(AerialShot::new(
                    car,
                    candidate,
                    frame.match_state.time,
                    frame.match_state.gravity,
                    boost_accel,
                ));
                return controller;
            }

            let shot_type = if plan.mode == WallDriveMode::DoubleJump {
                ShotType::DoubleJump
            } else {
                ShotType::Jump
            };
            let candidate = ShotCandidate {
                time: time_remaining,
                ball: geometry.ball,
                ball_velocity: plan.ball_slice.velocity,
                approach: geometry.approach,
                shot_direction: geometry.shot_direction,
                shot_type,
                targeted: true,
                can_jump: true,
                fast: plan.mode == WallDriveMode::DoubleJump,
                score: time_remaining,
                parser: ParsedShot {
                    found: true,
                    fast: plan.mode == WallDriveMode::DoubleJump,
                    shot_type: Some(shot_type),
                    targets,
                    ball_offset: geometry.ball,
                    shot_vector: geometry.shot_direction,
                    approach_target: geometry.approach,
                    time: time_remaining,
                },
            };
            let press_time = get_jump_time(
                geometry.ball.z - car.location.z,
                car.velocity.z,
                frame.match_state.gravity.z,
            ).press as f32;
            self.strike.start_from_wall(candidate, frame.match_state.time, press_time);
            return controller;
        }

        let mut controller = controller;
        if !car.airborne
            && time_remaining <= geometry.jump_time
            && geometry.local_angle <= std::f32::consts::FRAC_PI_2
        {
            controller.jump = true;
        }
        if car.airborne {
            let (_, pd) = crate::control::default_pd(
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

    pub fn tick(
        &mut self,
        frame: &AgentFrame,
        prediction: &BallTrajectory,
        targets: ShotTargets,
        boost_accel: f32,
        unlimited_boost: bool,
    ) -> (rlbot::flat::ControllerState, TeamplayMessage) {
        let now = frame.match_state.time;
        let me = frame.me;
        let ball = frame.ball;

        if !frame.match_state.round_active || me.demolished {
            self.strike.reset();
            self.aerial = None;
            return (
                rlbot::flat::ControllerState::default(),
                shot_message("NO_SHOT", None, now, None, false),
            );
        }

        if frame.match_state.kickoff {
            if !self.was_kickoff {
                self.kickoff.reset();
                self.strike.reset();
                self.aerial = None;
                self.kickoff_contest_until = now + 10.0;
            }
            self.was_kickoff = true;

            let kicker = kickoff_kicker(&frame.cars, ball.location, me.team)
                .unwrap_or(me.index);
            let controller = if kicker == me.index {
                self.kickoff.tick_kicker(me, ball, me.team, boost_accel, now)
            } else {
                let cheat = kickoff_cheat_target(frame);
                drive_to(
                    &me,
                    DriveRequest {
                        target: cheat,
                        target_speed: 1850.0,
                        vector: None,
                        boost_accel,
                        allow_handbrake: false,
                        preserve_momentum: false,
                    },
                )
            };
            let status = if kicker == me.index { "NO_SHOT" } else { "DEFEND" };
            let message = shot_message(status, None, now, None, false);
            self.last_message = message.clone();
            return (controller, message);
        } else if self.was_kickoff {
            self.was_kickoff = false;
            self.kickoff.reset();
        }

        if let Some(mut aerial) = self.aerial.take() {
            let goal = (targets.left + targets.right) * 0.5;
            aerial.refresh_prediction(
                me,
                prediction,
                now,
                goal,
                frame.match_state.gravity,
                ball.location.z,
            );
            let controller = aerial.run(
                me,
                now,
                frame.match_state.gravity,
                boost_accel,
                0.0,
            );
            if !aerial.finished {
                let message = aerial_message(
                    &aerial,
                    now,
                    self.aerial_score,
                    None,
                );
                self.aerial = Some(aerial);
                self.last_message = message.clone();
                return (controller, message);
            }
        }

        let plan = build_plan(frame, prediction);
        let urgent = plan.urgent_save;
        let challenger = plan.role == Role::Challenge || urgent;
        // Do not cancel a committed shot merely because teammate ETA changes
        // flip the elected role for a frame. An explicit teammate SHOT_FOUND
        // claim is handled below; launched routines are never yielded.
        let horizon = if urgent { 3.0 } else { 5.5 };

        let mut candidate = if challenger {
            find_best_shot(
                me,
                &frame.cars,
                ball.location,
                prediction,
                targets,
                frame.match_state.gravity,
                boost_accel,
                now,
                self.kickoff_contest_until,
                urgent,
                true,
                unlimited_boost,
                horizon,
            )
        } else {
            None
        };

        if candidate.is_none() && challenger && !urgent {
            candidate = find_best_any_shot(
                me,
                &frame.cars,
                prediction,
                targets,
                frame.match_state.gravity,
                boost_accel,
                now,
                self.kickoff_contest_until,
                unlimited_boost,
                horizon.min(4.5),
            );
        }

        if candidate.is_none() && challenger && !urgent {
            // Python find_pass_shot is a last resort after goal-directed and
            // untargeted touches fail. The chosen pass lane is stored in the
            // parser targets and preserved by the strike executor.
            candidate = find_best_pass_shot(
                me,
                &frame.cars,
                ball.location,
                prediction,
                frame.match_state.gravity,
                boost_accel,
                now,
                self.kickoff_contest_until,
                false,
                true,
                unlimited_boost,
                horizon.min(4.0),
            );
        }

        if candidate.is_none() && challenger {
            // Final fallback: choose the earliest parser-valid touch rather
            // than driving past a reachable ball while waiting for a better
            // ranked contact. Emergency saves keep the shorter search window.
            candidate = find_fast_intercept(
                me,
                prediction,
                targets,
                frame.match_state.gravity,
                boost_accel,
                0.0,
                if urgent { 3.0 } else { 4.0 },
                unlimited_boost,
            );
        }

        // Aerial validation remains a separate boost-budgeted scan, as in
        // Python Volt. The committed AerialShot then owns its live prediction
        // refresh, jump/second-jump timing, pitchback and boost controller.
        candidate = candidate.filter(|shot| shot.shot_type != ShotType::Aerial);

        let priority_candidate = candidate.or_else(|| self.strike.active_candidate());
        let may_yield_committed_approach = challenger || self.strike.is_approaching();
        let teammate_claimed = !urgent
            && may_yield_committed_approach
            && teammate_has_priority_shot(&self.teammate_comms, now, priority_candidate);
        if teammate_claimed {
            candidate = None;
            self.strike.cancel_approach();
        }

        // Noob Black wall-route reachability is evaluated independently
        // from ordinary ground contacts. It is only selected when it is
        // competitive with the ordinary intercept and a teammate has not
        // already claimed a better concrete shot.
        if challenger && !urgent && !teammate_claimed && !self.strike.active() && self.aerial.is_none() {
            let side = team_side(me.team);
            let foe_goal = Vector::new(0.0, -side * 5120.0, 300.0);
            if let Some(wall) = find_wall_drive(&me, &prediction.slices, me.team, foe_goal, 5.0) {
                let competitive = candidate.is_none_or(|shot| wall.intercept_time <= shot.time + 0.18);
                if competitive && (0.12..=5.2).contains(&wall.intercept_time) {
                    let controller = self.run_wall_approach(
                        me,
                        wall,
                        targets,
                        frame,
                        boost_accel,
                    );
                    let message = wall_message(wall, now, plan.cover_owner);
                    self.last_message = message.clone();
                    return (controller, message);
                }
            }
        }

        // Prefer a viable source-validated aerial when its score beats the
        // current ground/jump candidate. The ceiling/height limits mirror
        // Volt's solo-match restriction and ceiling launch safety. Urgent
        // saves use earliest-contact scoring instead of attacking quality.
        if challenger && !teammate_claimed && !self.strike.active() && self.aerial.is_none() {
            let friendly_count = frame.cars.iter()
                .filter(|car| car.team == me.team && !car.demolished)
                .count();
            let enemy_count = frame.cars.iter()
                .filter(|car| car.team != me.team && !car.demolished)
                .count();
            let max_height = if friendly_count == 1 && enemy_count == 1 {
                1200.0
            } else {
                f32::INFINITY
            };
            let min_height = if max_height > 1200.0
                && me.location.z >= 2044.0 - me.hitbox.height * 1.1
            {
                551.0
            } else if unlimited_boost || me.airborne {
                150.0
            } else {
                450.0
            };
            let goal = (targets.left + targets.right) * 0.5;
            let aerial_candidate = find_aerial_candidate(
                me,
                prediction,
                goal,
                targets,
                &frame.cars,
                frame.match_state.gravity,
                boost_accel,
                now,
                ball.location.z,
                self.kickoff_contest_until,
                min_height,
                max_height,
                horizon.min(6.0),
                urgent,
                unlimited_boost,
            ).filter(|aerial| {
                candidate.is_none_or(|shot| aerial.score < shot.score)
            });

            if let Some(aerial_candidate) = aerial_candidate {
                let mut aerial = AerialShot::new(
                    me,
                    aerial_candidate,
                    now,
                    frame.match_state.gravity,
                    boost_accel,
                );
                aerial.refresh_prediction(
                    me,
                    prediction,
                    now,
                    goal,
                    frame.match_state.gravity,
                    ball.location.z,
                );
                let controller = aerial.run(
                    me,
                    now,
                    frame.match_state.gravity,
                    boost_accel,
                    0.0,
                );
                self.aerial_score = aerial_candidate.score;
                if !aerial.finished {
                    let message = aerial_message(
                        &aerial,
                        now,
                        self.aerial_score,
                        plan.cover_owner,
                    );
                    self.aerial = Some(aerial);
                    self.last_message = message.clone();
                    return (controller, message);
                }
            }
        }

        if let Some(shot) = candidate {
            let status = if shot.targeted { "SHOT_FOUND" } else { "ANY_SHOT_FOUND" };
            let controller = self.strike.tick(
                me,
                ball,
                Some(shot),
                prediction,
                targets,
                frame.match_state.gravity,
                boost_accel,
                now,
            );
            let message = shot_message(
                status,
                Some(shot),
                now,
                plan.cover_owner,
                false,
            );
            self.last_message = message.clone();
            return (controller, message);
        }

        // Keep a committed strike alive after launch. This also covers a
        // wall jump while the car is still physically attached to the wall.
        if self.strike.active() {
            let controller = self.strike.tick(
                me,
                ball,
                None,
                prediction,
                targets,
                frame.match_state.gravity,
                boost_accel,
                now,
            );
            let message = shot_message(
                "SHOT_FOUND",
                self.strike.active_candidate(),
                now,
                plan.cover_owner,
                false,
            );
            self.last_message = message.clone();
            return (controller, message);
        }

        self.strike.reset();

        let enemy_near_ball = opponent_near_ball(frame, 1700.0);
        let mut status = "NO_SHOT";
        let cover_released = false;
        let mut movement_vector = None;
        let mut movement_target = match plan.role {
            Role::Cover => {
                status = "DEFEND";
                cover_target(frame)
            }
            Role::Support | Role::KickoffCheat | Role::KickoffBoost => {
                let target = support_target(frame, self.last_support_target);
                self.last_support_target = Some(target);
                movement_vector = Some(ball.location);
                target
            }
            Role::Challenge => {
                // If no parser candidate exists, drive to the ball with a
                // high closing-speed target rather than orbiting for another
                // distant candidate.
                ball.location
            }
        };

        if plan.role == Role::Support && !enemy_near_ball {
            if let Some(bump_target) = transition_bump_target(frame) {
                movement_target = bump_target;
                movement_vector = None;
            }
        }

        // Low-boost collection is opportunistic. A cover player only leaves
        // the defensive lane for a nearby pad, and never during a goal threat.
        if !urgent && plan.role == Role::Support {
            if let Some(pad) = choose_support_pad(frame, movement_target) {
                movement_target = pad;
                status = "LOW_BOOST";
            }
        } else if !urgent && plan.role != Role::Cover && me.boost < 24.0 {
            if let Some(pad) = choose_boost_pad(frame, movement_target, enemy_near_ball) {
                if me.location.flat_distance(pad) < 900.0 {
                    movement_target = pad;
                    status = "LOW_BOOST";
                }
            }
        }

        let controller = if plan.role == Role::Cover {
            self.run_cover_role(frame)
        } else if me.airborne {
            recover_in_air(&me, boost_accel, Some(movement_target))
        } else {
            drive_to(
                &me,
                DriveRequest {
                    target: movement_target,
                    target_speed: if plan.role == Role::Challenge { 2300.0 } else { 1900.0 },
                    vector: movement_vector,
                    boost_accel,
                    allow_handbrake: true,
                    preserve_momentum: plan.role == Role::Challenge,
                },
            )
        };

        let message = shot_message(
            if teammate_claimed { "NO_SHOT" } else { status },
            None,
            now,
            plan.cover_owner,
            cover_released,
        );
        self.last_message = message.clone();
        (controller, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn team_message(status: &str, contact_time: f32, received_at: f32) -> TeamplayMessage {
        TeamplayMessage {
            status: status.to_owned(),
            shot_type: "JUMP".to_owned(),
            score: 50.0,
            contact_time,
            received_at,
            cover_owner: None,
            cover_released: false,
            receive_x: None,
            receive_y: None,
            receive_dir_x: None,
            receive_dir_y: None,
            contact_x: None,
            contact_y: None,
        }
    }

    #[test]
    fn transition_bump_uses_speed_dependent_lead_and_python_distance_limits() {
        let me = crate::state::CarState {
            location: Vector::ZERO,
            velocity: Vector::new(0.0, 2000.0, 0.0),
            ..crate::state::CarState::default()
        };
        let foe = crate::state::CarState {
            location: Vector::new(1000.0, 0.0, 17.0),
            velocity: Vector::new(0.0, 100.0, 80.0),
            ..crate::state::CarState::default()
        };
        let target = transition_bump_lead_target(me, foe).unwrap();
        let lead_time = (1000.0_f32 / 2650.0).clamp(0.12, 0.55);
        assert!((target.y - 100.0 * lead_time).abs() < 0.001);
        assert_eq!(target.z, 17.0);

        let too_close = crate::state::CarState {
            location: Vector::new(200.0, 0.0, 17.0),
            ..foe
        };
        assert!(transition_bump_lead_target(me, too_close).is_none());

        let too_far = crate::state::CarState {
            location: Vector::new(4201.0, 0.0, 17.0),
            ..foe
        };
        assert!(transition_bump_lead_target(me, too_far).is_none());
    }

    #[test]
    fn shot_type_names_are_stable_for_team_comms() {
        assert_eq!(shot_type_name(ShotType::Ground), "GROUND");
        assert_eq!(shot_type_name(ShotType::Jump), "JUMP");
        assert_eq!(shot_type_name(ShotType::DoubleJump), "DOUBLE_JUMP");
        assert_eq!(shot_type_name(ShotType::Aerial), "AERIAL");
    }

    #[test]
    fn wall_pass_is_a_concrete_teammate_priority_claim() {
        let messages = HashMap::from([
            (1, team_message("WALL_PASS", 1.0, 10.0)),
        ]);
        assert!(teammate_has_priority_shot(&messages, 10.10, None));
    }

    #[test]
    fn teammate_priority_ages_contact_time_and_ignores_expired_claims() {
        let active = HashMap::from([
            (1, team_message("SHOT_FOUND", 1.10, 10.0)),
        ]);
        assert!(teammate_has_priority_shot(&active, 10.20, None));

        let expired = HashMap::from([
            (1, team_message("SHOT_FOUND", 0.10, 10.0)),
        ]);
        assert!(!teammate_has_priority_shot(&expired, 10.30, None));
    }
}
