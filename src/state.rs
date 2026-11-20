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
