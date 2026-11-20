use crate::math::{Matrix3, Vector};
use rlbot::flat::{AirState, GamePacket};

/// Hitbox dimensions and local-space offset, matching Volt's Python hitbox_object.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Hitbox {
    pub length: f32,
    pub width: f32,
    pub height: f32,
    pub offset: Vector,
}

/// Per-tick car data in Volt's coordinate conventions.
///
/// World-space position and velocity are kept separate from local angular
/// velocity. The orientation basis is forward/right/up, matching VirxERLU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarState {
    pub index: usize,
    pub team: u32,
    pub location: Vector,
    pub velocity: Vector,
    pub orientation: Matrix3,
    pub angular_velocity: Vector,
    pub demolished: bool,
    pub airborne: bool,
    pub supersonic: bool,
    pub jumped: bool,
    pub doublejumped: bool,
    pub air_state: AirState,
    pub has_jumped: bool,
    pub has_double_jumped: bool,
    pub has_dodged: bool,
    pub dodge_timeout: f32,
    pub dodge_elapsed: f32,
    pub dodge_dir: Vector,
    pub flip_reset_available: bool,
    pub boost: f32,
    pub hitbox: Hitbox,
}

impl CarState {
    /// Build the Volt-style car wrapper from the current RLBot v5 packet.
    pub fn from_packet(packet: &GamePacket, index: usize) -> Option<Self> {
        let player = packet.players.get(index)?;
        let physics = &player.physics;

        let location = Vector::new(
            physics.location.x,
            physics.location.y,
            physics.location.z,
        );
        let velocity = Vector::new(
            physics.velocity.x,
            physics.velocity.y,
            physics.velocity.z,
        );
        let orientation = Matrix3::new(
            physics.rotation.pitch,
            physics.rotation.yaw,
            physics.rotation.roll,
        );

        // RLBot provides angular velocity in world coordinates. VirxERLU
        // stores it in the car's local forward/right/up coordinate system.
        let angular_velocity = orientation.dot(Vector::new(
            physics.angular_velocity.x,
            physics.angular_velocity.y,
            physics.angular_velocity.z,
        ));

        let air_state = player.air_state;
        let has_jumped = player.has_jumped;
        let has_double_jumped = player.has_double_jumped;
        let has_dodged = player.has_dodged;

        Some(Self {
            index,
            team: player.team,
            location,
            velocity,
            orientation,
            angular_velocity,
            demolished: player.demolished_timeout != -1.0,
            airborne: air_state != AirState::OnGround,
            supersonic: player.is_supersonic,
            jumped: matches!(air_state, AirState::Jumping | AirState::DoubleJumping),
            doublejumped: air_state == AirState::DoubleJumping,
            air_state,
            has_jumped,
            has_double_jumped,
            has_dodged,
            dodge_timeout: player.dodge_timeout,
            dodge_elapsed: player.dodge_elapsed,
            dodge_dir: Vector::new(player.dodge_dir.x, player.dodge_dir.y, 0.0),
            flip_reset_available: air_state == AirState::InAir
                && !has_jumped
                && !has_double_jumped
                && !has_dodged,
            boost: player.boost,
            hitbox: Hitbox {
                length: player.hitbox.length,
                width: player.hitbox.width,
                height: player.hitbox.height,
                offset: Vector::new(
                    player.hitbox_offset.x,
                    player.hitbox_offset.y,
                    player.hitbox_offset.z,
                ),
            },
        })
    }

    /// Convert a world-space vector into local forward/right/up coordinates.
    pub fn local(&self, world_vector: Vector) -> Vector {
        self.orientation.dot(world_vector)
    }

    pub fn local_velocity(&self) -> Vector {
        self.local(self.velocity)
    }

    /// Convert a world-space point into coordinates relative to the car.
    pub fn local_location(&self, world_location: Vector) -> Vector {
        self.local(world_location - self.location)
    }

    pub fn forward(&self) -> Vector {
        self.orientation.forward
    }

    pub fn right(&self) -> Vector {
        self.orientation.right
    }

    pub fn up(&self) -> Vector {
        self.orientation.up
    }
}

/// Ball position and velocity, matching Volt's Python ball_object.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BallState {
    pub location: Vector,
    pub velocity: Vector,
}

impl BallState {
    pub fn from_packet(packet: &GamePacket) -> Option<Self> {
        let ball = packet.balls.first()?;
        let physics = &ball.physics;

        Some(Self {
            location: Vector::new(
                physics.location.x,
                physics.location.y,
                physics.location.z,
            ),
            velocity: Vector::new(
                physics.velocity.x,
                physics.velocity.y,
                physics.velocity.z,
            ),
        })
    }
}


/// Match-level state mirrored from VirxERLU's game_object.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MatchState {
    pub time: f32,
    pub time_remaining: f32,
    pub overtime: bool,
    pub round_active: bool,
    pub kickoff: bool,
    pub match_ended: bool,
    pub friend_score: i32,
    pub foe_score: i32,
    pub gravity: Vector,
}

impl MatchState {
    /// Build match state using the bot's team index, preserving Volt's
    /// friend/foe score convention and the live mutator gravity.
    pub fn from_packet(packet: &GamePacket, team: usize) -> Option<Self> {
        use rlbot::flat::MatchPhase;

        let friend_score = packet.teams.get(team)?.score;
        let foe_score = packet.teams.get(1usize.checked_sub(team)?)?.score;
        let game = &packet.match_info;
        let phase = game.match_phase;

        Some(Self {
            time: game.seconds_elapsed,
            time_remaining: game.game_time_remaining,
            overtime: game.is_overtime,
            round_active: matches!(phase, MatchPhase::Active | MatchPhase::Kickoff),
            kickoff: phase == MatchPhase::Kickoff,
            match_ended: phase == MatchPhase::Ended,
            friend_score,
            foe_score,
            gravity: Vector::new(0.0, 0.0, game.world_gravity_z),
        })
    }
}

/// A boost pad's static field geometry and packet-updated active state.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BoostPadState {
    pub index: usize,
    pub location: Vector,
    pub large: bool,
    pub active: bool,
}

impl BoostPadState {
    pub fn from_field_info(
        index: usize,
        pad: &rlbot::flat::BoostPad,
    ) -> Self {
        Self {
            index,
            location: Vector::new(pad.location.x, pad.location.y, pad.location.z),
            large: pad.is_full_boost,
            active: true,
        }
    }

    pub fn update(&mut self, packet: &GamePacket) {
        if let Some(state) = packet.boost_pads.get(self.index) {
            self.active = state.is_active;
        }
    }
}

/// Immutable per-tick snapshot produced from the RLBot packet. Strategy code
/// consumes this instead of repeatedly decoding FlatBuffer fields.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentFrame {
    pub me: CarState,
    pub cars: Vec<CarState>,
    pub ball: BallState,
    pub match_state: MatchState,
    pub boost_pads: Vec<BoostPadState>,
}

impl AgentFrame {
    pub fn from_packet(
        packet: &GamePacket,
        player_index: usize,
        team: usize,
        boost_pads: &[BoostPadState],
    ) -> Option<Self> {
        let me = CarState::from_packet(packet, player_index)?;
        let ball = BallState::from_packet(packet)?;
        let match_state = MatchState::from_packet(packet, team)?;
        let cars = (0..packet.players.len())
            .filter_map(|index| CarState::from_packet(packet, index))
            .collect();

        let mut pads = boost_pads.to_vec();
        for pad in &mut pads {
            pad.update(packet);
        }

        Some(Self { me, cars, ball, match_state, boost_pads: pads })
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    #[test]
    fn match_state_default_is_safe_for_uninitialized_agent() {
        let state = MatchState::default();
        assert_eq!(state.time, 0.0);
        assert!(!state.round_active);
        assert!(!state.kickoff);
        assert!(!state.match_ended);
    }

    #[test]
    fn boost_pad_defaults_to_active_until_first_packet_update() {
        let pad = BoostPadState::default();
        assert!(!pad.active);
        assert_eq!(pad.index, 0);
        assert_eq!(pad.location, Vector::ZERO);
    }
}
