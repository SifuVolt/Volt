pub mod physics;
pub mod prediction;
pub mod planning;
pub mod shot_physics;
pub mod bot;
mod control;
mod math;
mod state;

use crate::bot::BotBrain;
use crate::prediction::{AbsoluteBallSlice, BallTrajectory};
use crate::state::{AgentFrame, BoostPadState, MatchState};

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
        PlayerInput,
    },
    util::{AgentEnvironment, PacketQueue},
};

struct VoltRustAgent {
    index: u32,
    team: u32,
    match_state: MatchState,
    ball_prediction: Vec<AbsoluteBallSlice>,
    boost_pads: Vec<BoostPadState>,
    brain: BotBrain,
}

impl BotAgent for VoltRustAgent {
    fn new(
        team: u32,
        controllable_info: ControllableInfo,
        _match_config: Arc<MatchConfiguration>,
        field_info: Arc<FieldInfo>,
        _packet_queue: &mut PacketQueue,
    ) -> Self {
        println!(
            "Volt-Rust initialized for player {}",
            controllable_info.index
        );

        Self {
            index: controllable_info.index,
            team,
            match_state: MatchState::default(),
            ball_prediction: Vec::new(),
            boost_pads: field_info.boost_pads.iter().enumerate().map(|(index, pad)| {
                BoostPadState::from_field_info(index, pad)
            }).collect(),
            brain: BotBrain::new(),
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
            return;
        };
        self.match_state = frame.match_state;
        self.boost_pads = frame.boost_pads;

        // VirxERLU does not run gameplay routines outside Active/Kickoff.
        if !frame.match_state.round_active {
            packet_queue.push(PlayerInput {
                player_index: self.index,
                controller_state: ControllerState::default(),
            });
            return;
        }

        let controller: ControllerState = self.brain.tick(
            frame.me,
            &frame.cars,
            frame.ball,
            &prediction,
            game_time,
            frame.match_state,
            &frame.boost_pads,
        );

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
