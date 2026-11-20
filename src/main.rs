pub mod physics;
mod math;
mod state;

use std::f32::consts::PI;
use std::sync::Arc;

use rlbot::{
    RLBotConnection,
    agents::{BotAgent, run_bot_agents},
    flat::{
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
}

impl BotAgent for VoltRustAgent {
    fn new(
        _team: u32,
        controllable_info: ControllableInfo,
        _match_config: Arc<MatchConfiguration>,
        _field_info: Arc<FieldInfo>,
        _packet_queue: &mut PacketQueue,
    ) -> Self {
        println!(
            "Volt-Rust initialized for player {}",
            controllable_info.index
        );

        Self {
            index: controllable_info.index,
        }
    }

    fn tick(
        &mut self,
        game_packet: &GamePacket,
        packet_queue: &mut PacketQueue,
    ) {
        let Some(ball) = state::BallState::from_packet(game_packet) else {
            return;
        };

        let Some(car_state) =
            state::CarState::from_packet(game_packet, self.index as usize)
        else {
            return;
        };

        // Keep the existing world-yaw steering behaviour while the new
        // state layer is introduced and validated.
        let car = &game_packet.players[self.index as usize].physics;
        let dx = ball.location.x - car_state.location.x;
        let dy = ball.location.y - car_state.location.y;

        let target_yaw = dy.atan2(dx);
        let angle = (target_yaw - car.rotation.yaw + PI)
            .rem_euclid(2.0 * PI) - PI;

        let mut controller = ControllerState::default();

        controller.throttle = 1.0;
        controller.steer = angle.clamp(-1.0, 1.0);

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
