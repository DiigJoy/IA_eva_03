"""
Script de entrenamiento para el agente de estacionamiento.

Este script permite entrenar agentes usando diferentes algoritmos de RL:
- PPO (Proximal Policy Optimization): Para acciones discretas
- SAC (Soft Actor-Critic): Para acciones continuas
- DQN (Deep Q-Network): Para acciones discretas

Uso:
    python train_parking.py --algo ppo --reward_version 2 --noise --total_timesteps 300000
    python train_parking.py --algo sac --reward_version 2 --noise --total_timesteps 300000
"""

import argparse
import os
import json
from datetime import datetime

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import parking_env  # noqa: F401 - Registra ParkingEnv-v0 y ParkingEnv-v1


def make_env_fn(
    env_id: str,
    reward_version: int = 2,
    noise: bool = False,
    render_mode=None,
    continuous: bool = False
):
    """
    Crea una función factory para el entorno.

    Args:
        env_id: ID del entorno registrado
        reward_version: Versión de la función de recompensa
        noise: Activar ruido ambiental
        render_mode: Modo de renderizado
        continuous: Si usar acciones continuas
    """
    def _init():
        env = gym.make(
            env_id,
            reward_version=reward_version,
            noise=noise,
            render_mode=render_mode,
            continuous=continuous,
        )
        return env
    return _init


def get_ppo_hyperparams(noise: bool = False) -> dict:
    """Hiperparámetros optimizados para PPO."""
    params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Mayor exploración
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        },
    }

    # Ajustes para entornos con ruido
    if noise:
        params["ent_coef"] = 0.02  # Más exploración con ruido
        params["learning_rate"] = 2e-4  # Aprendizaje más lento

    return params


def get_sac_hyperparams(noise: bool = False) -> dict:
    """Hiperparámetros optimizados para SAC."""
    params = {
        "learning_rate": 3e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",  # Entropía automática
        "policy_kwargs": {
            "net_arch": [256, 256]
        },
    }

    if noise:
        params["learning_rate"] = 1e-4
        params["buffer_size"] = 200000

    return params


def get_dqn_hyperparams(noise: bool = False) -> dict:
    """Hiperparámetros optimizados para DQN."""
    params = {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {
            "net_arch": [256, 256]
        },
    }

    if noise:
        params["exploration_fraction"] = 0.3
        params["exploration_final_eps"] = 0.1

    return params


def train(args):
    """Función principal de entrenamiento."""

    # Configurar directorio de logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_str = "_noise" if args.noise else ""
    log_name = f"{args.algo}_v{args.reward_version}{noise_str}_{timestamp}"
    log_dir = os.path.join("logs", log_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Configuración de Entrenamiento")
    print(f"{'='*60}")
    print(f"Algoritmo: {args.algo.upper()}")
    print(f"Versión de recompensa: {args.reward_version}")
    print(f"Ruido ambiental: {'Sí' if args.noise else 'No'}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Directorio de logs: {log_dir}")
    print(f"{'='*60}\n")

    # Determinar si usar acciones continuas
    continuous = args.algo == "sac"
    env_id = "ParkingEnv-v1" if continuous else "ParkingEnv-v0"

    # Crear entornos
    if args.algo == "ppo":
        # PPO funciona mejor con múltiples entornos en paralelo
        n_envs = args.n_envs
        env = make_vec_env(
            make_env_fn(
                env_id,
                args.reward_version,
                args.noise,
                render_mode=None,
                continuous=continuous
            ),
            n_envs=n_envs,
        )
    else:
        # SAC y DQN usan un solo entorno
        env = make_env_fn(
            env_id,
            args.reward_version,
            args.noise,
            render_mode=None,
            continuous=continuous
        )()
        env = Monitor(env)

    # Entorno de evaluación
    eval_env = gym.make(
        env_id,
        reward_version=args.reward_version,
        noise=args.noise,
        render_mode=None,
        continuous=continuous,
    )
    eval_env = Monitor(eval_env)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(args.eval_freq // (args.n_envs if args.algo == "ppo" else 1), 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // (args.n_envs if args.algo == "ppo" else 1), 1),
        save_path=log_dir,
        name_prefix=f"{args.algo}_checkpoint",
        verbose=0,
    )

    callback_list = CallbackList([eval_callback, checkpoint_callback])

    # Crear modelo según algoritmo
    if args.algo == "ppo":
        hyperparams = get_ppo_hyperparams(args.noise)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **hyperparams
        )
    elif args.algo == "sac":
        hyperparams = get_sac_hyperparams(args.noise)
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **hyperparams
        )
    else:  # dqn
        hyperparams = get_dqn_hyperparams(args.noise)
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **hyperparams
        )

    # Guardar configuración
    config = {
        "algorithm": args.algo,
        "reward_version": args.reward_version,
        "noise": args.noise,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs if args.algo == "ppo" else 1,
        "continuous": continuous,
        "hyperparams": {k: str(v) for k, v in hyperparams.items()},
        "timestamp": timestamp,
    }

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Entrenar
    print(f"\nIniciando entrenamiento con {args.algo.upper()}...")
    print(f"TensorBoard: tensorboard --logdir {log_dir}\n")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")

    # Guardar modelo final
    final_model_path = os.path.join(log_dir, f"{args.algo}_parking_final")
    model.save(final_model_path)
    print(f"\nModelo final guardado en: {final_model_path}")

    # Evaluación final
    print("\n" + "="*60)
    print("Evaluación Final")
    print("="*60)

    rewards = []
    successes = 0
    n_eval = 50

    for i in range(n_eval):
        obs, info = eval_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        if info.get("is_success", False):
            successes += 1

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = successes / n_eval

    print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Tasa de éxito: {success_rate*100:.1f}% ({successes}/{n_eval})")

    # Guardar resultados finales
    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "success_rate": float(success_rate),
        "n_eval_episodes": n_eval,
    }

    with open(os.path.join(log_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Limpiar
    env.close()
    eval_env.close()

    print(f"\nEntrenamiento completado. Logs guardados en: {log_dir}")

    return log_dir, model


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de agente de estacionamiento con RL"
    )

    parser.add_argument(
        "--algo",
        choices=["ppo", "sac", "dqn"],
        default="ppo",
        help="Algoritmo de RL a usar (default: ppo)"
    )
    parser.add_argument(
        "--reward_version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Versión de función de recompensa: 1=simple, 2=shaped (default: 2)"
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Activar ruido ambiental (viento, fricción variable, sensores)"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=300000,
        help="Total de pasos de entrenamiento (default: 300000)"
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Número de entornos paralelos para PPO (default: 8)"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Frecuencia de evaluación en timesteps (default: 10000)"
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
