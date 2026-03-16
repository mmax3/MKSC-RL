import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import yaml

TRAIN_ALGO = "ppo"
DEFAULT_CONF_FILE = "mksc.yml"
DEFAULT_TB_LOGDIR = "runs"
DEFAULT_TB_PORT = "6006"

HYPERPARAM_FIELDS = {
    "policy": str,
    "n_envs": int,
    "n_timesteps": float,
    "n_steps": int,
    "batch_size": int,
    "n_epochs": int,
    "gamma": float,
    "gae_lambda": float,
    "learning_rate": str,
    "clip_range": str,
    "ent_coef": float,
    "vf_coef": float,
    "frame_stack": int,
}

HIDDEN_HYPERPARAM_FIELDS = {"policy", "n_envs"}
SCHEDULE_HYPERPARAM_FIELDS = {"learning_rate", "clip_range"}

DEFAULT_HYPERPARAMS = {
    "policy": "CnnPolicy",
    "n_envs": 1,
    "n_timesteps": 2e6,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": "lin_2.5e-4",
    "clip_range": "lin_0.1",
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "frame_stack": 4,
}

TRAIN_FIELDS = {
    "env_id": "MKSC-v0",
    "gym_packages": "mksc",
    "vec_env": "dummy",
    "device": "cpu",
    "eval_freq": "-1",
    "eval_episodes": "3",
    "save_freq": "100000",
    "log_folder": "logs",
    "trained_agent": "",
    "n_timesteps_override": "",
}

HIDDEN_TRAIN_FIELDS = {"env_id", "gym_packages", "device", "conf_file", "algo", "vec_env"}

HYPERPARAM_TOOLTIPS = {
    "n_timesteps": "Total training timesteps from YAML. Typical: 5e5 to 5e6. More steps = longer training.",
    "n_steps": "Rollout length per env before PPO update. Typical: 256 to 2048. Larger = smoother but slower updates.",
    "batch_size": "Mini-batch size for PPO SGD. Typical: 64 to 256. Must be <= n_steps * n_envs.",
    "n_epochs": "Gradient passes over each rollout. Typical: 3 to 10. Higher can overfit stale rollouts.",
    "gamma": "Discount factor. Typical: 0.98 to 0.999. Higher values favor long-term reward.",
    "gae_lambda": "Bias/variance tradeoff for GAE. Typical: 0.9 to 0.98. 0.95 is common.",
    "learning_rate": "Optimizer LR. Use a number for constant LR (e.g. 2.5e-4) or a schedule string (e.g. lin_2.5e-4). Typical: 1e-4 to 3e-4.",
    "clip_range": "PPO policy clip. Use a number for constant clip (e.g. 0.1) or a schedule string (e.g. lin_0.1). Typical: 0.1 to 0.3.",
    "ent_coef": "Entropy bonus. Typical: 0.0 to 0.02. Higher encourages exploration.",
    "vf_coef": "Value loss weight. Typical: 0.25 to 1.0.",
    "frame_stack": "How many frames are stacked as observation. Typical: 4.",
}

TRAIN_FIELD_TOOLTIPS = {
    "eval_freq": "Evaluation frequency (steps). -1 disables evaluation. Typical: -1, 10000, 25000.",
    "eval_episodes": "Episodes per evaluation when eval is enabled. Typical: 3 to 10.",
    "save_freq": "Checkpoint save frequency (steps). -1 disables checkpoints. Typical: 25000 to 100000.",
    "log_folder": "Root folder for RL Zoo outputs. Typical: logs",
    "trained_agent": "Path to a .zip model to resume training from. Leave empty for fresh training.",
    "n_timesteps_override": "Optional CLI override for total timesteps (-n). Empty uses YAML n_timesteps.",
}


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)
        self.widget.bind("<ButtonPress>", self._hide)

    def _show(self, _event=None):
        if self.tip_window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=3,
            wraplength=460,
        )
        label.pack()

    def _hide(self, _event=None):
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class TrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MKSC PPO Trainer")
        self.geometry("950x760")

        self.proc: subprocess.Popen | None = None
        self.tb_proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.stop_requested = False
        self.conf_file_path = DEFAULT_CONF_FILE

        self.hyper_entries: dict[str, tk.Entry] = {}
        self.train_entries: dict[str, tk.Entry] = {}
        self.hidden_hyperparams: dict[str, object] = {}
        self.run_ids: list[int] = []
        self.selected_run_id = tk.StringVar(value="")
        self._tooltips: list[ToolTip] = []

        self._build_ui()
        self._load_defaults_into_form()
        self._startup_load_yaml()
        self.refresh_runs()
        self.after(100, self._drain_log_queue)

    def _startup_load_yaml(self):
        try:
            self.load_yaml(create_if_missing=False, show_errors=False)
        except Exception as e:
            self._append_log(f"Startup YAML load failed: {e}\n")

    def _build_ui(self):
        root = tk.Frame(self, padx=10, pady=10)
        root.pack(fill=tk.BOTH, expand=True)

        hp_frame = tk.LabelFrame(root, text="Hyperparameters (saved in YAML)", padx=8, pady=8)
        hp_frame.pack(fill=tk.X)

        row = 0
        for key in HYPERPARAM_FIELDS.keys():
            if key in HIDDEN_HYPERPARAM_FIELDS:
                continue
            lbl = tk.Label(hp_frame, text=key, width=18, anchor="w")
            lbl.grid(row=row, column=0, sticky="w", pady=2)
            entry = tk.Entry(hp_frame, width=26)
            entry.grid(row=row, column=1, sticky="w", pady=2)
            self.hyper_entries[key] = entry
            self._attach_tooltip(lbl, HYPERPARAM_TOOLTIPS.get(key, ""))
            self._attach_tooltip(entry, HYPERPARAM_TOOLTIPS.get(key, ""))
            row += 1

        btn_row = tk.Frame(hp_frame)
        btn_row.grid(row=0, column=2, rowspan=4, padx=12, sticky="n")
        tk.Button(btn_row, text="Load YAML", width=14, command=self.load_yaml).pack(pady=4)
        tk.Button(btn_row, text="Save YAML", width=14, command=self.save_yaml).pack(pady=4)
        tk.Button(btn_row, text="Choose YAML", width=14, command=self.choose_yaml).pack(pady=4)

        train_frame = tk.LabelFrame(root, text="Train", padx=8, pady=8)
        train_frame.pack(fill=tk.X, pady=(10, 0))

        r = 0
        for key, value in TRAIN_FIELDS.items():
            if key in HIDDEN_TRAIN_FIELDS:
                continue
            lbl = tk.Label(train_frame, text=key, width=20, anchor="w")
            lbl.grid(row=r, column=0, sticky="w", pady=2)
            if key == "trained_agent":
                row_frame = tk.Frame(train_frame)
                row_frame.grid(row=r, column=1, sticky="w", pady=2)
                entry = tk.Entry(row_frame, width=34)
                if value:
                    entry.insert(0, value)
                entry.pack(side=tk.LEFT)
                tk.Button(row_frame, text="Browse", width=8, command=self.choose_trained_agent).pack(side=tk.LEFT, padx=6)
                self.train_entries[key] = entry
            else:
                entry = tk.Entry(train_frame, width=40)
                if value != "":
                    entry.insert(0, value)
                entry.grid(row=r, column=1, sticky="w", pady=2)
                self.train_entries[key] = entry
            self._attach_tooltip(lbl, TRAIN_FIELD_TOOLTIPS.get(key, ""))
            self._attach_tooltip(entry, TRAIN_FIELD_TOOLTIPS.get(key, ""))
            r += 1

        action_row = tk.Frame(train_frame)
        action_row.grid(row=0, column=2, rowspan=5, padx=12, sticky="n")
        self.train_btn = tk.Button(action_row, text="Train", width=14, command=self.start_training)
        self.train_btn.pack(pady=4)
        self.resume_btn = tk.Button(action_row, text="Resume", width=14, command=self.start_resume_training)
        self.resume_btn.pack(pady=4)
        self.stop_btn = tk.Button(action_row, text="Stop", width=14, command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(pady=4)
        self.tb_btn = tk.Button(action_row, text="TensorBoard", width=14, command=self.start_tensorboard)
        self.tb_btn.pack(pady=4)

        runs_frame = tk.LabelFrame(root, text="Runs", padx=8, pady=8)
        runs_frame.pack(fill=tk.X, pady=(10, 0))

        self.runs_label = tk.Label(runs_frame, text="Found 0 runs", width=18, anchor="w")
        self.runs_label.grid(row=0, column=0, sticky="w")

        self.runs_combo = ttk.Combobox(runs_frame, textvariable=self.selected_run_id, width=10, state="readonly")
        self.runs_combo.grid(row=0, column=1, sticky="w", padx=(6, 0))

        tk.Button(runs_frame, text="Refresh", width=10, command=self.refresh_runs).grid(row=0, column=2, padx=8)
        tk.Button(runs_frame, text="Enjoy 1 Episode", width=16, command=self.enjoy_one_episode).grid(row=0, column=3)

        log_frame = tk.LabelFrame(root, text="Training Output", padx=8, pady=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        text_container = tk.Frame(log_frame)
        text_container.pack(fill=tk.BOTH, expand=True)

        y_scroll = tk.Scrollbar(text_container, orient=tk.VERTICAL)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll = tk.Scrollbar(text_container, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.log_text = tk.Text(
            text_container,
            height=18,
            wrap=tk.NONE,
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.config(command=self.log_text.yview)
        x_scroll.config(command=self.log_text.xview)

    def _load_defaults_into_form(self):
        for key, value in DEFAULT_HYPERPARAMS.items():
            if key in HIDDEN_HYPERPARAM_FIELDS:
                self.hidden_hyperparams[key] = value
                continue
            self.hyper_entries[key].delete(0, tk.END)
            self.hyper_entries[key].insert(0, str(value))

    def _append_log(self, line: str):
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)

    def _attach_tooltip(self, widget: tk.Widget, text: str):
        if not text:
            return
        self._tooltips.append(ToolTip(widget, text))

    def _drain_log_queue(self):
        while not self.log_queue.empty():
            self._append_log(self.log_queue.get_nowait())
        self.after(100, self._drain_log_queue)

    def choose_yaml(self):
        selected = filedialog.askopenfilename(
            title="Choose YAML file",
            filetypes=[("YAML files", "*.yml *.yaml"), ("All files", "*.*")],
        )
        if selected:
            self.conf_file_path = selected
            self._append_log(f"Using config file: {self.conf_file_path}\n")

    def choose_trained_agent(self):
        selected = filedialog.askopenfilename(
            title="Choose saved model (.zip)",
            filetypes=[("Saved model", "*.zip"), ("All files", "*.*")],
        )
        if selected:
            entry = self.train_entries.get("trained_agent")
            if entry is not None:
                entry.delete(0, tk.END)
                entry.insert(0, selected)
            self._append_log(f"Using trained agent: {selected}\n")

    def _conf_path(self) -> str:
        return self.conf_file_path if self.conf_file_path else DEFAULT_CONF_FILE

    def _env_id(self) -> str:
        entry = self.train_entries.get("env_id")
        if entry is None:
            return TRAIN_FIELDS["env_id"]
        value = entry.get().strip()
        return value if value else TRAIN_FIELDS["env_id"]

    def _train_value(self, key: str) -> str:
        entry = self.train_entries.get(key)
        if entry is None:
            return str(TRAIN_FIELDS[key])
        value = entry.get().strip()
        return value if value else str(TRAIN_FIELDS[key])

    def _log_folder(self) -> str:
        value = self._train_value("log_folder").strip()
        return value if value else "logs"

    def refresh_runs(self):
        log_folder = self._log_folder()
        algo_dir = os.path.join(log_folder, TRAIN_ALGO)
        env_id = TRAIN_FIELDS["env_id"]

        run_ids: list[int] = []
        if os.path.isdir(algo_dir):
            for name in os.listdir(algo_dir):
                # Expected: {env_id}_{N}, e.g. MKSC-v0_2
                prefix = env_id + "_"
                if not name.startswith(prefix):
                    continue
                try:
                    run_id = int(name[len(prefix) :])
                except ValueError:
                    continue
                if os.path.isdir(os.path.join(algo_dir, name)):
                    run_ids.append(run_id)

        run_ids = sorted(set(run_ids))
        self.run_ids = run_ids
        self.runs_label.config(text=f"Found {len(run_ids)} runs")
        self.runs_combo["values"] = [str(x) for x in run_ids]
        if run_ids:
            current = self.selected_run_id.get().strip()
            if current == "" or int(current) not in run_ids:
                self.selected_run_id.set(str(run_ids[-1]))
        else:
            self.selected_run_id.set("")

    def enjoy_one_episode(self):
        rid_str = self.selected_run_id.get().strip()
        if rid_str == "":
            messagebox.showerror("Enjoy Error", "No run selected.")
            return
        try:
            exp_id = int(rid_str)
        except ValueError:
            messagebox.showerror("Enjoy Error", f"Invalid run id: {rid_str}")
            return

        cmd = [
            sys.executable,
            "enjoy_one_episode.py",
            "--algo",
            TRAIN_ALGO,
            "--env",
            TRAIN_FIELDS["env_id"],
            "--gym-packages",
            TRAIN_FIELDS["gym_packages"],
            "--folder",
            self._log_folder(),
            "--exp-id",
            str(exp_id),
        ]
        worker = threading.Thread(target=self._run_generic_cmd, args=(cmd, "[enjoy] "), daemon=True)
        worker.start()

    def _run_generic_cmd(self, cmd: list[str], prefix: str):
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            self.log_queue.put(f"{prefix}Running: {' '.join(cmd)}\n")
            for line in proc.stdout:
                self.log_queue.put(prefix + line)
            rc = proc.wait()
            self.log_queue.put(f"{prefix}Exited with code {rc}\n")
        except Exception as e:
            self.log_queue.put(f"{prefix}Failed: {e}\n")

    def _parse_hparam_value(self, key: str, raw: str):
        if key in SCHEDULE_HYPERPARAM_FIELDS:
            # RL Zoo accepts either schedule strings like `lin_2.5e-4`
            # or numeric YAML values for constant schedules.
            if "_" in raw:
                return raw
            return float(raw)
        cast = HYPERPARAM_FIELDS[key]
        if cast is str:
            return raw
        return cast(raw)

    def _collect_hyperparams(self) -> dict:
        out: dict = {}
        for key in HYPERPARAM_FIELDS.keys():
            if key in HIDDEN_HYPERPARAM_FIELDS:
                out[key] = self.hidden_hyperparams.get(key, DEFAULT_HYPERPARAMS[key])
                continue
            raw = self.hyper_entries[key].get().strip()
            if raw == "":
                raise ValueError(f"Hyperparameter '{key}' is empty.")
            out[key] = self._parse_hparam_value(key, raw)
        return out

    def save_yaml(self):
        try:
            env_id = self._env_id()
            conf_file = self._conf_path()
            hyperparams = self._collect_hyperparams()
            with open(conf_file, "w", encoding="utf-8") as f:
                yaml.safe_dump({env_id: hyperparams}, f, sort_keys=False)
            self._append_log(f"Saved: {conf_file}\n")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_yaml(self, *, create_if_missing: bool = True, show_errors: bool = True):
        try:
            conf_file = self._conf_path()
            if not os.path.exists(conf_file):
                if create_if_missing:
                    self.save_yaml()
                else:
                    self._append_log(f"Config not found: {conf_file} (using defaults)\n")
                return
            with open(conf_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            env_id = self._env_id()
            if env_id not in data:
                raise ValueError(f"Environment '{env_id}' not found in {conf_file}.")
            hp = data[env_id]
            for key in HYPERPARAM_FIELDS.keys():
                if key in HIDDEN_HYPERPARAM_FIELDS:
                    if key in hp:
                        self.hidden_hyperparams[key] = hp[key]
                    continue
                if key in hp:
                    self.hyper_entries[key].delete(0, tk.END)
                    self.hyper_entries[key].insert(0, str(hp[key]))
            self._append_log(f"Loaded: {conf_file}\n")
        except Exception as e:
            if show_errors:
                messagebox.showerror("Load Error", str(e))
            else:
                self._append_log(f"Load YAML error: {e}\n")

    def _training_cmd(self, *, trained_agent: str = "") -> list[str]:
        algo = TRAIN_ALGO
        env_id = self._env_id()
        gym_packages = self._train_value("gym_packages")
        conf_file = self._conf_path()
        vec_env = self._train_value("vec_env")
        device = self._train_value("device")
        eval_freq = self.train_entries["eval_freq"].get().strip() or "-1"
        eval_episodes = self.train_entries["eval_episodes"].get().strip() or "3"
        save_freq = self.train_entries["save_freq"].get().strip() or "-1"
        log_folder = self.train_entries["log_folder"].get().strip() or "logs"
        n_timesteps_override = self.train_entries["n_timesteps_override"].get().strip()

        cmd = [
            sys.executable,
            "-m",
            "rl_zoo3.train",
            "--algo",
            algo,
            "--env",
            env_id,
            "--conf-file",
            conf_file,
            "--vec-env",
            vec_env,
            "--device",
            device,
            "--tensorboard-log",
            DEFAULT_TB_LOGDIR,
            "--log-folder",
            log_folder,
        ]
        try:
            eval_freq_int = int(eval_freq)
        except ValueError:
            eval_freq_int = -1
        cmd.extend(["--eval-freq", str(eval_freq_int)])
        if eval_freq_int >= 0:
            cmd.extend(["--eval-episodes", eval_episodes])
        try:
            save_freq_int = int(save_freq)
        except ValueError:
            save_freq_int = -1
        cmd.extend(["--save-freq", str(save_freq_int)])
        if gym_packages:
            cmd.extend(["--gym-packages", *gym_packages.split()])
        if trained_agent:
            cmd.extend(["--trained-agent", trained_agent])
        if n_timesteps_override:
            cmd.extend(["-n", n_timesteps_override])
        return cmd

    def start_training(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo("Training Running", "A training process is already running.")
            return
        try:
            self.save_yaml()
            cmd = self._training_cmd(trained_agent="")
            self._append_log("Starting training:\n")
            self._append_log(" ".join(cmd) + "\n\n")
            self.train_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.stop_requested = False
            worker = threading.Thread(target=self._run_training, args=(cmd,), daemon=True)
            worker.start()
        except Exception as e:
            self.train_btn.config(state=tk.NORMAL)
            self.resume_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            messagebox.showerror("Train Error", str(e))

    def start_resume_training(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo("Training Running", "A training process is already running.")
            return
        try:
            trained_agent = self.train_entries["trained_agent"].get().strip()
            if not trained_agent:
                raise ValueError("Please choose a trained_agent .zip to resume from.")
            if not os.path.isfile(trained_agent):
                raise ValueError(f"trained_agent not found: {trained_agent}")

            self.save_yaml()
            cmd = self._training_cmd(trained_agent=trained_agent)
            self._append_log("Resuming training:\n")
            self._append_log(" ".join(cmd) + "\n\n")
            self.train_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.stop_requested = False
            worker = threading.Thread(target=self._run_training, args=(cmd,), daemon=True)
            worker.start()
        except Exception as e:
            self.train_btn.config(state=tk.NORMAL)
            self.resume_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            messagebox.showerror("Resume Error", str(e))

    def stop_training(self):
        proc = self.proc
        if proc is None or proc.poll() is not None:
            self.log_queue.put("No active training process to stop.\n")
            self.stop_btn.config(state=tk.DISABLED)
            return
        self.stop_requested = True
        self.log_queue.put("Stop requested. Sending graceful interrupt...\n")
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception as e:
            self.log_queue.put(f"Interrupt failed: {e}\n")
            return

        def _terminate_if_needed():
            p = self.proc
            if p is None:
                return
            if p.poll() is None:
                self.log_queue.put("Still running, sending terminate...\n")
                try:
                    p.terminate()
                except Exception as ex:
                    self.log_queue.put(f"Terminate failed: {ex}\n")

        def _force_kill_if_needed():
            p = self.proc
            if p is None:
                return
            if p.poll() is None:
                self.log_queue.put("Still running, killing process...\n")
                try:
                    p.kill()
                except Exception as ex:
                    self.log_queue.put(f"Kill failed: {ex}\n")

        self.after(2500, _terminate_if_needed)
        self.after(5500, _force_kill_if_needed)

    def start_tensorboard(self):
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            self.log_queue.put(f"TensorBoard already running at http://127.0.0.1:{DEFAULT_TB_PORT}\n")
            return
        try:
            self.tb_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    DEFAULT_TB_LOGDIR,
                    "--port",
                    DEFAULT_TB_PORT,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            time.sleep(0.8)
            rc = self.tb_proc.poll()
            if rc is not None:
                details = ""
                if self.tb_proc.stdout is not None:
                    details = self.tb_proc.stdout.read() or ""
                self.log_queue.put(f"TensorBoard failed to start (exit code {rc}).\n")
                if details.strip():
                    self.log_queue.put(details + ("\n" if not details.endswith("\n") else ""))
                return
            self.log_queue.put(f"TensorBoard started: http://127.0.0.1:{DEFAULT_TB_PORT} (logdir={DEFAULT_TB_LOGDIR})\n")
            worker = threading.Thread(target=self._read_tb_output, daemon=True)
            worker.start()
        except Exception as e:
            self.log_queue.put(f"Failed to start TensorBoard: {e}\n")

    def _read_tb_output(self):
        if self.tb_proc is None or self.tb_proc.stdout is None:
            return
        for line in self.tb_proc.stdout:
            self.log_queue.put(f"[tb] {line}")

    def _run_training(self, cmd: list[str]):
        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(line)
            rc = self.proc.wait()
            if self.stop_requested:
                self.log_queue.put(f"\nTraining stopped by user (exit code {rc}).\n")
            else:
                self.log_queue.put(f"\nTraining exited with code {rc}\n")
        except Exception as e:
            self.log_queue.put(f"\nTraining failed: {e}\n")
        finally:
            self.proc = None
            self.stop_requested = False
            self.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.resume_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def on_close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.tb_proc is not None and self.tb_proc.poll() is None:
            try:
                self.tb_proc.terminate()
            except Exception:
                pass
        self.destroy()


def main():
    app = TrainerGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
