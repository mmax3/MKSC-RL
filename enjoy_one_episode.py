import argparse
import importlib
import os

import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import StoreDict, get_model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=EnvironmentName, default="MKSC-v0", help="environment ID")
    parser.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS.keys()), help="RL algorithm")
    parser.add_argument("-f", "--folder", type=str, default="logs", help="Log folder")
    parser.add_argument("--exp-id", type=int, default=0, help="Experiment ID (0: latest)")
    parser.add_argument("--load-best", action="store_true", default=False, help="Load best model instead of last")
    parser.add_argument("--load-checkpoint", type=int, default=None, help="Load checkpoint at given timesteps")
    parser.add_argument("--load-last-checkpoint", action="store_true", default=False, help="Load last checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=-1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Keyword args for env constructor")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000_000,
        help="Safety cap in case the episode never terminates",
    )
    args = parser.parse_args()

    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    set_random_seed(args.seed)
    if args.num_threads > 0:
        th.set_num_threads(args.num_threads)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    _, model_path, log_path = get_model_path(
        args.exp_id,
        folder,
        algo,
        env_name,
        args.load_best,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )
    print(f"Loading {model_path}")

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, test_mode=True)

    env_kwargs: dict = {}
    if "env_kwargs" in hyperparams:
        env_kwargs.update(hyperparams["env_kwargs"])
        del hyperparams["env_kwargs"]

    # Allow env kwargs saved during training to be reused
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args.get("env_kwargs") is not None:
                env_kwargs = loaded_args["env_kwargs"]

    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env = create_test_env(
        env_name.gym_id,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
        vec_env_cls=ExperimentManager.default_vec_env_cls,
    )

    # Workaround for loading across python versions; mimic rl_zoo3.enjoy
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device)

    obs = env.reset()
    episode_reward = 0.0
    episode_len = 0
    last_info = None

    while True:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, infos = env.step(action)
        # VecEnv returns arrays
        episode_reward += float(reward[0])
        episode_len += 1
        last_info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else None
        if bool(done[0]):
            break
        if episode_len >= args.max_steps:
            print(f"Reached max-steps cap ({args.max_steps}), exiting.")
            break

    try:
        env.close()
    except Exception:
        pass

    print(f"Episode reward: {episode_reward}")
    print(f"Episode length: {episode_len}")
    if last_info is not None:
        # Avoid dumping large structures (arrays/matrices) to the terminal.
        keys_of_interest = [
            "done_reason",
            "lap",
            "checkpoint",
            "stuck_steps",
            "reverse_steps",
            "slow_no_prog_steps",
            "progress_units",
        ]
        summary = {}
        for k in keys_of_interest:
            v = last_info.get(k, None)
            if isinstance(v, (str, int, float, bool)) or v is None:
                summary[k] = v
        print(f"Last info (summary): {summary}")


if __name__ == "__main__":
    main()
