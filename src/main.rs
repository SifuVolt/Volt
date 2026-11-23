pub mod aerial;
pub mod physics;
pub mod prediction;
pub mod planning;
pub mod shot_physics;
pub mod shot_parser;
pub mod shot_planner;
pub mod mechanics;
pub mod kickoff;
pub mod teamplay;
pub mod bot;
pub mod brain;
mod control;
mod defense;
mod math;
mod state;

use crate::bot::BotBrain;
use crate::prediction::{AbsoluteBallSlice, BallTrajectory};
use crate::state::{AgentFrame, AgentSettings, BoostAmountMode, BotPersistentState, BoostPadState, GoalState, MatchState};

use std::sync::Arc;

use rlbot::{
    RLBotConnection,
    agents::{BotAgent, run_bot_agents},
    flat::{
        BallPrediction,
        ControllableInfo,
        ControllerState,
        FieldInfo,
        GamePacket,
        MatchConfiguration,
        MatchComm,
        PlayerInput,
    },
    util::{AgentEnvironment, PacketQueue},
};

struct VoltRustAgent {
    index: u32,
    team: u32,
    match_state: MatchState,
    settings: AgentSettings,
    friend_goal: GoalState,
    foe_goal: GoalState,
    foe_goal_shot: (crate::math::Vector, crate::math::Vector),
    anti_target: (crate::math::Vector, crate::math::Vector),
    persistent: BotPersistentState,
    ball_prediction: Vec<AbsoluteBallSlice>,
    boost_pads: Vec<BoostPadState>,
    brain: BotBrain,
    last_comms_signature: Vec<u8>,
    last_comms_send_time: f32,
    last_hold_log_time: f32,
}

impl BotAgent for VoltRustAgent {
    fn new(
        team: u32,
        controllable_info: ControllableInfo,
        match_config: Arc<MatchConfiguration>,
        field_info: Arc<FieldInfo>,
        _packet_queue: &mut PacketQueue,
    ) -> Self {
        println!(
            "Volt-Rust initialized for player {}",
            controllable_info.index
        );

        let settings = AgentSettings::from_match_configuration(&match_config);
        let friend_goal = GoalState::new(team as usize, 820.0, 350.0);
        let foe_goal = GoalState::new(1usize.saturating_sub(team as usize), 760.0, 300.0);

        let foe_goal_width_direction =
            (foe_goal.right_post - foe_goal.left_post).normalize();
        let foe_goal_shot = (
            foe_goal.left_post + foe_goal_width_direction * 39.0,
            foe_goal.right_post - foe_goal_width_direction * 39.0,
        );

        let friend_goal_width_direction =
            (friend_goal.right_post - friend_goal.left_post).normalize();
        let anti_target = (
            friend_goal.right_post + friend_goal_width_direction * 100.0,
            friend_goal.left_post - friend_goal_width_direction * 100.0,
        );

        let mut brain = BotBrain::new_with_boost_accel(settings.boost_accel);
        brain.set_unlimited_boost(settings.boost_amount == BoostAmountMode::Unlimited);

        Self {
            index: controllable_info.index,
            team,
            match_state: MatchState::default(),
            settings,
            friend_goal,
            foe_goal,
            foe_goal_shot,
            anti_target,
            persistent: BotPersistentState::new(),
            ball_prediction: Vec::new(),
            boost_pads: field_info.boost_pads.iter().enumerate().map(|(index, pad)| {
                BoostPadState::from_field_info(index, pad)
            }).collect(),
            brain,
            last_comms_signature: Vec::new(),
            last_comms_send_time: -999.0,
            last_hold_log_time: -999.0,
        }
    }

    fn on_match_comm(
        &mut self,
        match_comm: &MatchComm,
        _packet_queue: &mut PacketQueue,
    ) {
        if match_comm.team != self.team
            || !match_comm.team_only
            || match_comm.index == self.index
        {
            return;
        }
        if let Some((sender, message)) = crate::teamplay::decode(
            match_comm.index as usize,
            match_comm.team,
            self.team,
            self.index as usize,
            &match_comm.content,
            self.match_state.time,
        ) {
            self.persistent.teammate_comms.insert(sender, message);
        }
    }

    fn on_ball_prediction(
        &mut self,
        ball_prediction: &BallPrediction,
        _packet_queue: &mut PacketQueue,
    ) {
        self.ball_prediction = ball_prediction
            .slices
            .iter()
            .map(|slice| AbsoluteBallSlice {
                game_seconds: slice.game_seconds,
                location: crate::math::Vector::new(
                    slice.physics.location.x,
                    slice.physics.location.y,
                    slice.physics.location.z,
                ),
                velocity: crate::math::Vector::new(
                    slice.physics.velocity.x,
                    slice.physics.velocity.y,
                    slice.physics.velocity.z,
                ),
            })
            .collect();
    }

    fn tick(
        &mut self,
        game_packet: &GamePacket,
        packet_queue: &mut PacketQueue,
    ) {
        let game_time = game_packet.match_info.seconds_elapsed;
        let prediction = BallTrajectory::from_absolute_slices(
            self.ball_prediction.iter().copied(),
            game_time,
        );

        // Decode the packet once into the same stable per-tick objects used
        // throughout VirxERLU: self, teammates/opponents, ball, match and pads.
        let Some(frame) = AgentFrame::from_packet(
            game_packet,
            self.index as usize,
            self.team as usize,
            &self.boost_pads,
        ) else {
            // Never silently confuse packet decoding failure with a bot
            // choosing to stop. Log the exact missing-packet context while
            // still sending neutral controls for safety.
            if game_time - self.last_hold_log_time >= 0.5 {
                println!(
                    "[AGENT HOLD] reason=frame_decode_none player={} players={} balls={} teams={} t={:.3}",
                    self.index,
                    game_packet.players.len(),
                    game_packet.balls.len(),
                    game_packet.teams.len(),
                    game_time,
                );
                self.last_hold_log_time = game_time;
            }
            packet_queue.push(PlayerInput {
                player_index: self.index,
                controller_state: ControllerState::default(),
            });
            return;
        };
        self.match_state = frame.match_state;
        self.persistent.time = frame.match_state.time;
        self.persistent.odd_tick += 1;
        if self.persistent.odd_tick > 3 {
            self.persistent.odd_tick = 0;
        }
        if !self.persistent.kickoff_flag
            && frame.match_state.round_active
            && frame.match_state.kickoff
        {
            self.persistent.kickoff_done = false;
            self.persistent.kickoff_routine_started = false;
        }
        self.persistent.kickoff_flag =
            frame.match_state.round_active && frame.match_state.kickoff;
        self.boost_pads = frame.boost_pads.clone();

        // VirxERLU does not run gameplay routines outside Active/Kickoff.
        if !frame.match_state.round_active {
            if game_time - self.last_hold_log_time >= 0.5 {
                println!(
                    "[AGENT HOLD] reason=round_not_active player={} t={:.3} kickoff={}",
                    self.index,
                    game_time,
                    frame.match_state.kickoff,
                );
                self.last_hold_log_time = game_time;
            }
            packet_queue.push(PlayerInput {
                player_index: self.index,
                controller_state: ControllerState::default(),
            });
            return;
        }

        self.persistent.teammate_comms.retain(|_, message| {
            (game_time - message.received_at) <= 1.0
        });
        self.brain.set_teammate_comms(&self.persistent.teammate_comms);

        let controller: ControllerState = self.brain.tick(
            frame.me,
            &frame.cars,
            frame.ball,
            &prediction,
            game_time,
            frame.match_state,
            &frame.boost_pads,
            crate::shot_parser::ShotTargets {
                left: self.foe_goal_shot.0,
                right: self.foe_goal_shot.1,
            },
        );

        let teamplay_message = self.brain.teamplay_message(game_time);
        if let Some(content) = crate::teamplay::encode(
            self.index as usize,
            self.team,
            &teamplay_message,
        ) {
            let changed = content != self.last_comms_signature;
            let refresh_due = game_time - self.last_comms_send_time >= 0.25;
            let rate_limit_clear = game_time - self.last_comms_send_time >= 0.10;
            if rate_limit_clear && (changed || refresh_due) {
                packet_queue.push(MatchComm {
                    index: self.index,
                    team: self.team,
                    team_only: true,
                    display: None,
                    content: content.clone(),
                });
                self.last_comms_signature = content;
                self.last_comms_send_time = game_time;
            }
        }

        packet_queue.push(PlayerInput {
            player_index: self.index,
            controller_state: controller,
        });
    }
}

fn main() {
    let AgentEnvironment {
        server_addr,
        agent_id,
    } = AgentEnvironment::from_env();

    let agent_id = agent_id.unwrap_or_else(|| "sifu/volt-rust".into());

    println!("Volt-Rust connecting...");

    let connection = RLBotConnection::new(&server_addr)
        .expect("Failed to connect to RLBot");

    println!("Volt-Rust connected!");

    run_bot_agents::<VoltRustAgent>(
        agent_id.clone(),
        true,
        true,
        connection,
    )
    .expect("Volt-Rust agent crashed");

    println!("Volt-Rust exited.");
}
