# Volt Rust Python-Parity Rebuild

## Source of truth

The behavioral specification is the Python Volt implementation on
`SifuVolt/Grid-v1:partycannon-python-port`, especially:

- `routines of volt`
- `tools for volt`
- `agent for volt`
- `utils from util of volt`
- the matching `util/` physics, drive, prediction, and contact helpers

Rust-specific ownership, borrowing, data layout, and module boundaries may differ.
The decision order, equations, thresholds, prediction indices, and controller
outputs must not be approximated or silently redesigned.

## Preserve and port as-is

These are explicitly reported as working in-game and are not part of the
rewrite unless parity tests show a concrete defect:

- side-wall drive and jumping off the wall
- fast aerial execution
- wavedash
- half flip
- double jump

The ordinary speedflip mechanic is retained, but the **speedflip kickoff
routine** had been missing from the Rust decision path and is restored in
`src/kickoff.rs`.

## Replace the broken gameplay path

The current Rust `BotBrain::choose_plan` / `drive_to` path is not the final
architecture. Replace it with independent, testable modules:

1. **Agent adapter** — decode RLBot packets once; own the persistent game
   objects and emit exactly one controller per tick. No strategy in this layer.
2. **Prediction** — preserve absolute game timestamps, convert them to packet
   relative time once, and expose explicit sample-at-time / sample-index
   functions. Never mix absolute and relative time.
3. **Shot/tool selector** — port Python `run_logic` and `tools.find_shot`
   decision order. Keep candidate discovery separate from candidate
   commitment. A candidate is not a routine until every source reachability
   check passes.
4. **Ground shot routine** — persistent state with refreshed intercept,
   cached goal aim, hitbox/contact approach, source speed request, boost
   decision, exact viability/flip window, and failure/recovery handoff.
5. **Jump shot routine** — persistent state with Python's prediction refresh,
   jump-time calculation, approach-speed control, S-tilt, jump hold/release,
   contact gate, directional dodge, and abort/recovery behavior.
6. **Steering and throttle** — pure functions porting `defaultPD`,
   `defaultThrottle`, and `defaultDrive`; callers must pass the same target
   and target speed the Python routine passes.
7. **Routine lifecycle** — explicit priority/ownership rules so a committed
   routine cannot be silently replaced, restarted, or left returning neutral
   controls.

The old planner and shot execution code should be removed only after the
replacement module has parity tests and the runtime is switched over. Do not
delete the working mechanics above or destroy the reference Python source.

## Confirmed source mismatches

### Kickoff

Python has a dedicated `speedflip_kickoff` routine. It accelerates toward the
ball, starts `speed_flip` only when total car speed is greater than 800 and
distance is greater than 1800, waits for landing, drives to 1000 distance,
then performs its one-frame jump / release / one-frame dodge and ends on touch
or a 2-second timeout. Rust previously had only a late generic kickoff dodge
inside 550 distance. The dedicated state machine is now in `src/kickoff.rs`.

### Ground shot

Python `ground_shot.run` is a real persistent routine. It refreshes the
predicted ball slice and contact approach, calculates

`speed_required = distance_remaining / clamp(T, 0.000001, 6)`

(with its source-specific 2300 close-contact case and speed-building
adjustment), runs `defaultDrive`, predicts arrival using dodge impulse, tests
the source flip window, calls `ground_shot_is_viable`, and pops to recovery
when the shot is no longer possible.

The current Rust selector returns a `GroundIntercept`, but the live path does
not have an equivalent persistent ground-shot executor. It passes the
approach target through generic `drive_to` with a plan speed of 2300. That is
not the Python ground-shot routine and must be replaced, not tuned.

### Steering

The Python `defaultPD` cubic PD and the `defaultThrottle` acceleration,
turn-radius, handbrake, and boost calculations already have close Rust
counterparts. Do not replace those equations with a new steering model.
First establish frame-by-frame parity for the inputs and outputs. In
particular, a correct steering function cannot compensate for a wrong
contact target, wrong time-to-intercept, or wrong requested speed.

### Runtime stalls

The adapter previously sent a neutral controller silently when
`AgentFrame::from_packet` returned `None`, and also sent neutral controls
when the match was not active. `src/main.rs` now logs these cases as
`[AGENT HOLD]` with packet counts and match time. These logs distinguish
packet/phase holds from a routine that has stopped producing useful controls.

## Rust ecosystem boundary

The `rlbot` crate is the RLBot v5 Rust interface: packet, agent, and
controller transport/types. It is not a Rocket League steering or shot-planning
library. Use it for the live agent boundary. RocketSim provides game physics
for offline simulation and can be evaluated as a test harness, but it does not
replace Volt's Python calculations or the live RLBot controller. See the
official crate documentation and RocketSim description:

- https://docs.rs/rlbot/0.7.1
- https://docs.rs/crate/rocketsim/0.2.8

## Required parity tests before runtime replacement

For recorded Python inputs, compare Rust against Python at the same frame:

- selected routine and rejection reason
- prediction sample index and absolute intercept time
- predicted ball position and velocity
- contact point and car-centre approach target
- target speed, throttle, steer, boost, and handbrake
- jump start time, S-tilt state, jump release frames, dodge frame
- routine push/pop/abort/recovery transitions

Tests must include low boost, high lateral velocity, a slowing/rolling ball,
sharp approach angles, a moving intercept, invalid prediction data, and a
candidate that becomes unreachable after selection.

A green `cargo test` suite is necessary but does not establish Python parity
or in-game success. Those are separate validation stages.
