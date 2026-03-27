# MKSC PPO Training with BizHawk

This project trains a PPO agent from screenshots from **Mario Kart: Super Circuit** 

Video of learned agent after approx. 15 million steps: <https://youtu.be/80bOWpP_WsE>

This project is using:

- `BizHawk` as the emulator
- `GymBizHawk` as the Python <-> BizHawk bridge
- `Gymnasium` as the environment API
- `Stable-Baselines3` and `rl_zoo3` for PPO training
- a local Tkinter GUI in `main.py` for editing hyperparameters, launching training, TensorBoard, and one-episode evaluation

The environment is registered as `MKSC-v0` by [mksc.py](./mksc.py), and the reward/termination logic lives in [mksc.lua](./mksc.lua).

## What is in this repo

- [main.py](./main.py): GUI for training, resume, TensorBoard, and evaluation
- [mksc.py](./mksc.py): `Gymnasium` environment registration
- [mksc.lua](./mksc.lua): BizHawk-side Lua environment logic
- [mksc.yml](./mksc.yml): PPO hyperparameters used by `rl_zoo3`
- [enjoy_one_episode.py](./enjoy_one_episode.py): run one full evaluation episode from a selected training run
- `.env`: local paths for BizHawk and the MKSC ROM

## Requirements

- Windows
- Python `3.12+`
- `uv` installed: <https://docs.astral.sh/uv/>
- BizHawk `2.11` available locally
- Mario Kart: Super Circuit ROM available locally

This repository expects the following local directories/files to exist:

- a BizHawk installation directory
- a MKSC ROM path

## 1. Clone the repository

If this repo includes `GymBizHawk` and `rl-baselines3-zoo` as folders, a normal clone is enough.

```powershell
git clone <your-repo-url>
cd MKSC-RL
```

## 2. Create the environment variables

Create a `.env` file in the project root with these variables:

```dotenv
BIZHAWK_DIR=D:\path\to\BizHawk-2.11-win-x64
MKSC_PATH=D:\path\to\Mario Kart - Super Circuit.gba
```

Meaning:

- `BIZHAWK_DIR`: BizHawk installation directory
- `MKSC_PATH`: full path to the MKSC ROM file

The environment is loaded automatically by [mksc.py](./mksc.py).
## 3. Apply the GymBizHawk patch

This project expects one small local patch on top of upstream `GymBizHawk`.

The patch file is:

```text
patches/gymbizhawk-keep-string-info.patch
```

Apply it after cloning `GymBizHawk`:

```powershell
git -C GymBizHawk apply ../patches/gymbizhawk-keep-string-info.patch
```

What it changes:

- preserves non-numeric values in the Lua `info` dictionary
- keeps fields like `done_reason` available on the Python side

## 4. Install Python dependencies

This project uses `uv` and the dependencies from [pyproject.toml](./pyproject.toml).

```powershell
uv sync
```

Notes:

- `tensorboard` is included
- `setuptools<81` is pinned because current TensorBoard still relies on `pkg_resources`

## 5. Optional sanity check

You can verify the main files compile:

```powershell
uv run python -m py_compile main.py
uv run python -m py_compile mksc.py
uv run python -m py_compile enjoy_one_episode.py
```

## 6. Launch the GUI

Start the training GUI:

```powershell
uv run python main.py
```

The GUI provides:

- editable PPO hyperparameters from `mksc.yml`
- save/load YAML
- training
- resume training from a saved `.zip`
- TensorBoard launcher
- run discovery from `logs/ppo/`
- one-episode evaluation via `enjoy_one_episode.py`

## 7. Configure hyperparameters

Hyperparameters are stored in [mksc.yml](./mksc.yml).

Important fields:

- `n_timesteps`: total training steps
- `n_steps`: PPO rollout size
- `batch_size`: PPO mini-batch size
- `n_epochs`: PPO epochs per rollout
- `learning_rate`
- `clip_range`
- `frame_stack`

For `learning_rate` and `clip_range`, the GUI accepts either:

- a constant numeric value, for example:

```yaml
learning_rate: 2.5e-4
clip_range: 0.1
```

- or a linear schedule string, for example:

```yaml
learning_rate: lin_2.5e-4
clip_range: lin_0.1
```

## 8. Start training

In the GUI:

1. adjust hyperparameters if needed
2. click `Save YAML`
3. click `Train`

Training is launched through `rl_zoo3.train` with:

- `--algo ppo`
- `--env MKSC-v0`
- `--gym-packages mksc`
- `--conf-file mksc.yml`

Artifacts are written to:

- `logs/ppo/` for models and run outputs
- `runs/` for TensorBoard logs

## 9. Resume training

To continue from a previous saved model:

1. click `Browse` next to `trained_agent`
2. select a saved `.zip` from `logs/ppo/...`
3. click `Resume`

Important behavior:

- the PPO weights are resumed
- optimizer state is resumed
- if you use schedule strings like `lin_2.5e-4`, the schedule restarts at the beginning of the resumed training run

If you want simpler resume behavior, use constant values such as:

```yaml
learning_rate: 2.5e-4
clip_range: 0.1
```

## 10. Monitor training with TensorBoard

From the GUI, click `TensorBoard`.

It starts on:

```text
http://127.0.0.1:6006
```

Typical messages such as:

- `pkg_resources is deprecated`
- `TensorFlow installation not found - running with reduced feature set`

are warnings, not failures.

## 11. Evaluate a trained model

### GUI

Use the `Runs` section:

1. click `Refresh`
2. select a run id
3. click `Enjoy 1 Episode`

This runs one full episode using [enjoy_one_episode.py](./enjoy_one_episode.py).

### Command line

You can run one evaluation episode manually:

```powershell
uv run python enjoy_one_episode.py --algo ppo --env MKSC-v0 --gym-packages mksc --folder logs --exp-id 2
```

This script:

- loads the selected model
- creates a test environment
- runs until episode termination
- prints a compact episode summary

## 12. How reset and termination work

Environment logic is in [mksc.lua](./mksc.lua).

Current behavior:

- observations are screenshots
- reward is computed in Lua
- termination happens in Lua
- after termination, Python resets the environment
- reset uses `savestate.loadslot(1)` in BizHawk, so savestate 1 has to be created manually prior to learning

Typical terminal reasons include:

- `race_finished`
- `stuck`
- `reverse`
- `slow_no_progress`

## 13. Known project assumptions

- This setup is Windows-oriented.
- BizHawk must be available locally; it is not installed by `uv`.
- The ROM path must be provided by `.env`.
- The GUI assumes a single-environment PPO workflow on CPU.

## 14. Typical workflow

```powershell
uv sync
uv run python main.py
```

Then:

1. verify `.env` paths are correct
2. adjust PPO hyperparameters if needed
3. click `Train`
4. use `TensorBoard` to monitor progress
5. use `Enjoy 1 Episode` to inspect learned behavior

## Troubleshooting

### `BIZHAWK_DIR` or `MKSC_PATH` assertion error

Your `.env` file is missing or incorrect. Check:

- `BIZHAWK_DIR`
- `MKSC_PATH`

### TensorBoard starts with warnings

This is expected. As long as the web UI opens, it is fine.

### Training starts but the agent does nothing useful

Check:

- BizHawk is receiving inputs
- the savestate in slot `1` is valid
- the ROM matches the memory addresses expected by [mksc.lua](./mksc.lua)

### Resumed training feels unstable

If using `lin_...` schedules, that is expected. Try constant values such as:

```yaml
learning_rate: 2.5e-4
clip_range: 0.1
```
