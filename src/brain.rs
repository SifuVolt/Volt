//! Main decision engine.
//!
//! This is a new modular decision layer: the role planner chooses who should
//! challenge, the Volt shot planner evaluates reachable contacts, the strike
//! executor runs the timed mechanics, and off-ball roles navigate separately.
//! Packet decoding and RLBot v5 I/O remain outside this module.

use std::collections::HashMap;

use crate::{
    drive::{drive_to, recover_in_air, DriveRequest},
    math::Vector,
    prediction::BallTrajectory,
    routines::{KickoffExecutor, StrikeExecutor},
    shot_parser::ShotTargets,
    shot_physics::ShotType,
    shot_planner::{
        find_best_any_shot, find_best_shot, find_fast_intercept,
        ShotCandidate,
    },
    state::{AgentFrame, TeamplayMessage},
    strategy::{
        build_plan, choose_boost_pad, cover_target, kickoff_cheat_target,
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

            let predicted = foe.location + foe.velocity * 0.18;
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
        match message.status.as_str() {
            // A concrete SHOT_FOUND always beats a speculative ANY_SHOT_FOUND
            // when the contacts are within the handoff window.
            "SHOT_FOUND" => message.contact_time <= our_time + 0.20,
            "ANY_SHOT_FOUND" => our_candidate.is_none()
                || message.contact_time + 0.08 < our_time,
            _ => false,
        }
    })
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
    kickoff: KickoffExecutor,
    last_message: TeamplayMessage,
    kickoff_contest_until: f32,
    was_kickoff: bool,
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
            kickoff: KickoffExecutor::default(),
            last_message: shot_message("NO_SHOT", None, 0.0, None, false),
            kickoff_contest_until: -999.0,
            was_kickoff: false,
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
            return (
                rlbot::flat::ControllerState::default(),
                shot_message("NO_SHOT", None, now, None, false),
            );
        }

        if frame.match_state.kickoff {
            if !self.was_kickoff {
                self.kickoff.reset();
                self.strike.reset();
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

        let plan = build_plan(frame, prediction);
        let urgent = plan.urgent_save;
        let challenger = plan.role == Role::Challenge || urgent;
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

        let teammate_claimed = !urgent
            && challenger
            && teammate_has_priority_shot(&self.teammate_comms, now, candidate);
        if teammate_claimed {
            candidate = None;
        }

        if let Some(shot) = candidate {
            let status = if shot.targeted { "SHOT_FOUND" } else { "ANY_SHOT_FOUND" };
            let controller = self.strike.tick(
                me,
                ball,
                Some(shot),
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

        // Keep a committed airborne strike alive if the ground-only planner
        // cannot issue a fresh candidate this frame.
        if self.strike.active() && me.airborne {
            let controller = self.strike.tick(
                me,
                ball,
                None,
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
        let mut movement_target = match plan.role {
            Role::Cover => {
                status = "DEFEND";
                cover_target(frame)
            }
            Role::Support | Role::KickoffCheat | Role::KickoffBoost => {
                support_target(frame, plan.challenger)
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
            }
        }

        // Low-boost collection is opportunistic. A cover player only leaves
        // the defensive lane for a nearby pad, and never during a goal threat.
        if !urgent && me.boost < 24.0 {
            if let Some(pad) = choose_boost_pad(frame, movement_target, enemy_near_ball) {
                if me.location.flat_distance(pad) < 900.0 || plan.role == Role::Support {
                    movement_target = pad;
                    status = "LOW_BOOST";
                }
            }
        }

        let controller = if me.airborne {
            recover_in_air(&me, boost_accel, Some(movement_target))
        } else {
            drive_to(
                &me,
                DriveRequest {
                    target: movement_target,
                    target_speed: if plan.role == Role::Challenge { 2300.0 } else { 1900.0 },
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

    #[test]
    fn shot_type_names_are_stable_for_team_comms() {
        assert_eq!(shot_type_name(ShotType::Ground), "GROUND");
        assert_eq!(shot_type_name(ShotType::Jump), "JUMP");
        assert_eq!(shot_type_name(ShotType::DoubleJump), "DOUBLE_JUMP");
        assert_eq!(shot_type_name(ShotType::Aerial), "AERIAL");
    }
}
