pub mod brain;
pub mod drive;
pub mod mechanics;
pub mod physics;
pub mod prediction;
pub mod routines;
pub mod shot_physics;
pub mod shot_parser;
pub mod shot_planner;
pub mod strategy;
pub mod teamplay;

mod control;
mod math;
mod state;

use crate::brain::BotBrain;
use crate::prediction::{AbsoluteBallSlice, BallTrajectory};
use crate::state::{AgentFrame, AgentSettings, BoostPadState, GoalState, MatchState, TeamplayMessage};
use crate::shot_parser::ShotTargets;

use std::collections::HashMap;
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
    foe_goal_targets: ShotTargets,
    ball_prediction: Vec<AbsoluteBallSlice>,
    boost_pads: Vec<BoostPadState>,
    teammate_comms: HashMap<usize, TeamplayMessage>,
    brain: BotBrain,
    last_comms_signature: Vec<u8>,
    last_comms_send_time: f32,
}

impl BotAgent for VoltRustAgent {
    fn new(
        team: u32,
        controllable_info: ControllableInfo,
        match_config: Arc<MatchConfiguration>,
        field_info: Arc<FieldInfo>,
        _packet_queue: &mut PacketQueue,
    ) -> Self {
        let settings = AgentSettings::from_match_configuration(match_config.as_ref());
        let opponent_team = 1usize.saturating_sub(team as usize);
        let foe_goal = GoalState::new(opponent_team, 820.0, 350.0);
        let foe_goal_targets = ShotTargets {
            left: foe_goal.left_post,
            right: foe_goal.right_post,
        };
        let boost_pads = field_info.boost_pads.iter().enumerate()
            .map(|(index, pad)| BoostPadState::from_field_info(index, pad))
            .collect();

        Self {
            index: controllable_info.index,
            team,
            match_state: MatchState::default(),
            settings,
            foe_goal_targets,
            ball_prediction: Vec::new(),
            boost_pads,
            teammate_comms: HashMap::new(),
            brain: BotBrain::new(),
            last_comms_signature: Vec::new(),
            last_comms_send_time: -999.0,
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
            self.teammate_comms.insert(sender, message);
        }
    }

    fn on_ball_prediction(
        &mut self,
        ball_prediction: &BallPrediction,
        _packet_queue: &mut PacketQueue,
    ) {
        self.ball_prediction = ball_prediction.slices.iter()
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

        let Some(frame) = AgentFrame::from_packet(
            game_packet,
            self.index as usize,
            self.team as usize,
            &self.boost_pads,
        ) else {
            // Never leave last packet's controls latched if a packet is
            // temporarily incomplete.
            packet_queue.push(PlayerInput {
                player_index: self.index,
                controller_state: ControllerState::default(),
            });
            return;
        };

        self.match_state = frame.match_state;
        self.boost_pads.clone_from(&frame.boost_pads);

        if !frame.match_state.round_active {
            packet_queue.push(PlayerInput {
                player_index: self.index,
                controller_state: ControllerState::default(),
            });
            return;
        }

        self.teammate_comms.retain(|_, message| {
            (game_time - message.received_at) <= 1.0
        });
        self.brain.set_teammate_comms(&self.teammate_comms);

        let (controller, message) = self.brain.tick(
            &frame,
            &prediction,
            self.foe_goal_targets,
            self.settings.boost_accel,
            self.settings.unlimited_boost(),
        );

        if let Some(content) = crate::teamplay::encode(
            self.index as usize,
            self.team,
            &message,
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

    println!("Volt-Rust autonomous brain connecting...");
    let connection = RLBotConnection::new(&server_addr)
        .expect("Failed to connect to RLBot");
    println!("Volt-Rust connected.");

    run_bot_agents::<VoltRustAgent>(
        agent_id.clone(),
        true,
        true,
        connection,
    )
    .expect("Volt-Rust agent crashed");

    println!("Volt-Rust exited.");
}
