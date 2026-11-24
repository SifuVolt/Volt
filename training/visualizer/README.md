# Volt-Rust live-match visualizer

This visualizer uses actual RLBot match telemetry from the Rust bot. It does not require RocketSim or collision meshes.

## 1. Enable telemetry

Open the same Command Prompt you use to launch RLBot and run:

```cmd
set VOLT_TRACE_DIR=C:\Users\yaees\OneDrive\Desktop\RLBOTS\Volt-Rust\training\traces
```

Then launch RLBot and play a match. Each Volt-Rust process writes its own CSV file:

- `volt_player_0_team_0.csv`
- `volt_player_1_team_0.csv`

The file is reset when that player process starts. Telemetry is disabled entirely when `VOLT_TRACE_DIR` is not set.

## 2. Install plotting dependency

From the Volt-Rust repository root:

```cmd
venv\Scripts\python.exe -m pip install -r training\visualizer\requirements.txt
```

## 3. Generate plots

After the match ends:

```cmd
venv\Scripts\python.exe training\visualizer\plot_volt_trace.py --input training\traces
```

Open the generated `training\traces\plots\` directory. For each bot, the visualizer creates:

- `*_xy.png`: car path and ball path
- `*_boost.png`: boost level over match time
- `*_distances.png`: distance to ball compared with nearest teammate/opponent
- `*_controls.png`: throttle, steer, pitch, yaw and roll output
- `*_summary.txt`: status counts, jump frames and boost-button frames

These plots let us distinguish a bot that never selects a shot from one that selects a shot but produces weak or incorrect controls. The CSV also records shot statuses, contact-time estimates and cover ownership.

This is live-match telemetry, not a physics simulator. It tells us what Volt decided and commanded; it does not predict counterfactual outcomes. The separate RocketSim smoke test remains only an installation/API check until a full controller adapter is implemented.
