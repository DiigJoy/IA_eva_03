import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import parking_env  # noqa: F401  # para registrar ParkingEnv-v0


def make_env_fn(reward_version=2, noise=False, render_mode=None):
    def _init():
        env = gym.make(
            "ParkingEnv-v0",
            reward_version=reward_version,
            noise=noise,
            render_mode=render_mode,
        )
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--reward_version", type=int, default=2)
    parser.add_argument("--noise", action="store_true", help="Activa ruido en el entorno")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    args = parser.parse_args()

    log_dir = os.path.join("logs", f"{args.algo}_v{args.reward_version}")
    os.makedirs(log_dir, exist_ok=True)

    # PPO: usamos varios entornos en paralelo, DQN: uno solo
    if args.algo == "ppo":
        env = make_vec_env(
            make_env_fn(args.reward_version, args.noise, render_mode=None),
            n_envs=8,
        )
    else:
        # DQN no soporta multi-env en SB3, usamos uno solo
        env = make_env_fn(args.reward_version, args.noise, render_mode=None)()

    # Entorno de evaluación
    eval_env = gym.make(
        "ParkingEnv-v0",
        reward_version=args.reward_version,
        noise=args.noise,
        render_mode=None,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
    )

    model_path = os.path.join(log_dir, f"{args.algo}_parking_final")
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
