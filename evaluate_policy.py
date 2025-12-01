import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN

import parking_env  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reward_version", type=int, default=2)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    env = gym.make(
        "ParkingEnv-v0",
        reward_version=args.reward_version,
        noise=args.noise,
        render_mode=render_mode,
    )

    ModelClass = PPO if args.algo == "ppo" else DQN
    model = ModelClass.load(args.model_path)

    rewards = []
    successes = 0

    for ep in range(args.n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            last_info = info

        rewards.append(ep_reward)
        if last_info.get("is_success", False):
            successes += 1

        print(f"Episodio {ep+1}: R = {ep_reward:.2f}, éxito = {last_info.get('is_success', False)}")

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    success_rate = successes / args.n_episodes

    print("\n================= RESULTADOS =================")
    print(f"Recompensa media: {mean_r:.2f} ± {std_r:.2f}")
    print(f"Tasa de éxito: {success_rate*100:.1f}% ({successes}/{args.n_episodes})")

    env.close()


if __name__ == "__main__":
    main()
