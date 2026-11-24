# Volt-Rust

A clean modular Rust rebuild of Volt's Rocket League decision and execution
layers, using the RLBot v5 Rust interface.

## Design

The bot is split into small, explicit layers:

- brain.rs coordinates the per-tick decision, shot selection, team
  communication, and routine ownership.
- strategy.rs assigns one primary challenger, one cover player, and
  support/transition roles. Emergency saves override ordinary role priority.
- drive.rs combines Volt's existing defaultPD/defaultThrottle port with a
  speed-aware waypoint controller and an airborne recovery controller.
- aerial.rs keeps the Party Cannon-style aerial candidate validator, live
  prediction refresh, hitbox-aware contact correction, fast/normal aerial
  sequencing, backboard handling, and boost-aware flight controller.
- planning.rs and defense.rs retain Noob Black's wall-unfolding reachability
  scan and Volt's landing-plane/recovery calculations.
- routines.rs runs ground approaches, jump shots, double jumps, aerial
  alignment, directional dodge contact, and the timed speedflip kickoff.
- shot_parser.rs, shot_physics.rs, and shot_planner.rs retain the
  source-faithful VirxERLU/Volt contact reachability and candidate scoring
  calculations. They are kept separate from controller state machines.
- prediction/ and state.rs convert RLBot packets and the collision-aware
  ball prediction into stable per-tick values.
- teamplay.rs retains the volt_teamplay_v1 team-only message contract.
  Wall routes are selected only when reachable and competitive with the
  ordinary shot, and when no teammate has claimed a better concrete contact.

## Reference material

The decision/routing separation takes architectural inspiration from
[whatisaphone/self-driving-car](https://github.com/whatisaphone/self-driving-car).
That repository is an older Rust RLBot project. Its GPL source files are not
copied into this bot; the current RLBot v5 interface and this repository's
existing Rust packet adapter are retained.

The Botimus & Bumblebee strike folder in RLBotPack is Python, not Rust. It is
useful as a separate strike-behaviour reference, not as a Rust runtime base.

The mechanics and shot calculations are based on the user's
Grid-v1 partycannon-python-port branch:
https://github.com/SifuVolt/Grid-v1/tree/partycannon-python-port

The implementation keeps the 60 Hz shot-reachability parser, 120 Hz jump
constants, Party Cannon contact-direction model, hitbox-aware approach
geometry, speedflip sequence, and Volt team communication protocol in Rust.

## Rotation convention

The car basis is constructed with the following expression:

~~~rust
Mat3::from_euler(
    EulerRot::ZYX,
    physics.rotation.yaw,
    physics.rotation.pitch,
    physics.rotation.roll,
)
~~~

Matrix columns 0, 1, and 2 are forward, right, and up respectively. World-to-car
conversion is the dot product against those columns.

## Build and test

~~~sh
cargo test --locked
cargo build --release --locked
~~~

The repository GitHub Actions workflow runs both commands for pull requests
targeting main. A clean architecture and a passing build do not replace
in-game validation: shot timing, wall contacts, and team role thresholds still
need match testing and tuning.
