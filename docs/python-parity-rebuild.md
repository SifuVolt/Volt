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
7. **Routine stack** — replace the collection of unrelated optional state
   fields with a typed Rust routine stack. Python Volt's `push`, `pop`,
   `clear`, same-class `update`, protected-routine rules, and
   `is_clear()/odd_tick` search cadence must have explicit Rust equivalents.
   A routine returns a controller plus a transition (`Keep`, `Pop`,
   `Push`, or `Replace`); it must never silently lose ownership.
8. **Routine lifecycle** — explicit priority/ownership rules so a committed
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

Python's selector deliberately maps a low-ball `GROUND_SHOT` candidate to
`jump_shot` when jumping is enabled; it uses the standalone `ground_shot`
routine only when jumping is disabled. The Rust planner similarly routes most
ground candidates through `JumpShotState`, but it hardcodes the allowed shot
types instead of receiving the caller's `can_ground/can_jump/can_double_jump/
can_aerial` settings.

The Rust live path now extracts the exact Python close-contact,
distance-over-time, and speed-building calculation into
`src/brain/ground_shot.rs`, with focused tests, and calls it from the live
ground-shot path. This is not yet the complete persistent Python routine:
intercept refresh, cached aim, patience state, stack ownership, and
recovery/flip transitions still need to be ported around this calculation.

### Selector and routine ownership

Python `tools.find_shot` accepts explicit `can_ground`, `can_jump`,
`can_double_jump`, and `can_aerial` flags, scans the samples returned by
`get_slices`, and scores candidates with `shot_candidate_score`. It also
keeps an active routine alive: the selector updates the same routine class,
and ordinary shots are protected from arbitrary replacement. The Rust
planner currently hardcodes much of the shot-type availability and its
`BotBrain` uses separate optional state fields rather than the Python
routine stack. The rewrite must port the call order and ownership semantics,
not just the parser equations.

### Steering

The Python `defaultPD` cubic PD and the `defaultThrottle` acceleration,
turn-radius, handbrake, and boost calculations already have close Rust
counterparts. Do not replace those equations with a new steering model.
First establish frame-by-frame parity for the inputs and outputs. In
particular, a correct steering function cannot compensate for a wrong
contact target, wrong time-to-intercept, or wrong requested speed.

### Runtime stalls

The adapter logs missing frame decoding and inactive match phases as
`[AGENT HOLD]`. The brain now also rate-limits `[BOT NEUTRAL]` diagnostics
when its final controller is completely neutral, reporting the active
routine category and game time. This is diagnostic only; it does not alter
controller output. It distinguishes deliberate `power_clear` waiting from
a jump/aerial routine producing neutral controls or the planner having no
active action.

## Rust ecosystem boundary

The `rlbot` crate is the RLBot v5 interface: packet, agent, and controller
transport/types. It is not a Rocket League steering or shot-planning library.

There is also an existing Rust project, **VirxERLU-RLib**, with 120 Hz ball
prediction analysis, acceleration simulation, Dubins-path ground planning,
post correction, and ground/jump/double-jump/aerial candidate analysis. It is
a valuable Rust implementation to study and cross-check. It is not a blind
drop-in replacement: the Python Volt branch currently calls
`VirxERLU-CLib/virxrlcu.c`, while RLib has its own Rust API and uses
`rl_ball_sym` for its game/prediction model. The parity port must compare the
specific algorithms and data conventions before adopting any RLib module.

The most reliable source for exact Python parity is therefore the actual
`virxrlcu.c` implementation behind Volt's Python calls. Port those functions
and test them against source fixtures. Use the RLBot crate for the live agent
boundary. RocketSim can be evaluated as an offline physics test harness, but
it does not replace Volt's Python calculations or the live RLBot controller.

References:
- https://docs.rs/rlbot/0.7.1
- https://github.com/VirxEC/VirxERLU-CLib
- https://github.com/VirxEC/VirxERLU-RLib
- https://docs.rs/crate/rocketsim/0.3.7

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
