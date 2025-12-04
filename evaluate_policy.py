"""
Script de evaluación para políticas entrenadas.

Este script permite:
- Evaluar modelos entrenados con diferentes configuraciones
- Visualizar episodios en tiempo real
- Generar estadísticas detalladas
- Comparar diferentes modelos

Uso:
    python evaluate_policy.py --model_path logs/ppo_v2/best_model --n_episodes 50
    python evaluate_policy.py --model_path logs/sac_v2/best_model --render --n_episodes 10
"""

import argparse
import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN

import parking_env  # noqa: F401


def detect_algorithm(model_path: str) -> str:
    """Detecta el algoritmo basado en el path del modelo."""
    path_lower = model_path.lower()
    if "sac" in path_lower:
        return "sac"
    elif "dqn" in path_lower:
        return "dqn"
    return "ppo"


def load_model(model_path: str, algo: Optional[str] = None):
    """Carga un modelo entrenado."""
    if algo is None:
        algo = detect_algorithm(model_path)

    model_classes = {
        "ppo": PPO,
        "sac": SAC,
        "dqn": DQN
    }

    ModelClass = model_classes.get(algo, PPO)
    return ModelClass.load(model_path), algo


def evaluate_episode(
    model,
    env,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Evalúa un episodio completo.

    Returns:
        Diccionario con métricas del episodio
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    trajectory = []

    # Métricas adicionales
    min_distance = float('inf')
    final_distance = 0.0
    final_angle_error = 0.0
    final_speed = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        # Registrar posición
        if hasattr(env, 'unwrapped'):
            state = env.unwrapped.state
            if state is not None:
                trajectory.append((state[0], state[1], state[3]))

        # Actualizar métricas
        distance = info.get("distance", 0)
        if distance < min_distance:
            min_distance = distance

        done = terminated or truncated

    # Métricas finales
    final_distance = info.get("distance", 0)
    final_angle_error = info.get("angle_error", 0)
    final_speed = info.get("speed", 0)
    is_success = info.get("is_success", False)
    out_of_bounds = info.get("out_of_bounds", False)

    return {
        "total_reward": total_reward,
        "steps": steps,
        "is_success": is_success,
        "out_of_bounds": out_of_bounds,
        "min_distance": min_distance,
        "final_distance": final_distance,
        "final_angle_error": final_angle_error,
        "final_angle_error_deg": np.degrees(final_angle_error),
        "final_speed": final_speed,
        "trajectory": trajectory,
    }


def print_statistics(results: List[Dict[str, Any]]):
    """Imprime estadísticas detalladas de la evaluación."""
    n_episodes = len(results)

    # Extraer métricas
    rewards = [r["total_reward"] for r in results]
    steps = [r["steps"] for r in results]
    successes = sum(1 for r in results if r["is_success"])
    out_of_bounds = sum(1 for r in results if r["out_of_bounds"])
    min_distances = [r["min_distance"] for r in results]
    final_distances = [r["final_distance"] for r in results]
    final_angles = [r["final_angle_error_deg"] for r in results]
    final_speeds = [r["final_speed"] for r in results]

    # Estadísticas de episodios exitosos
    successful_results = [r for r in results if r["is_success"]]
    if successful_results:
        success_rewards = [r["total_reward"] for r in successful_results]
        success_steps = [r["steps"] for r in successful_results]
    else:
        success_rewards = [0]
        success_steps = [0]

    print("\n" + "="*70)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*70)

    print(f"\n{'Métricas Generales':-^50}")
    print(f"  Total de episodios:        {n_episodes}")
    print(f"  Episodios exitosos:        {successes} ({successes/n_episodes*100:.1f}%)")
    print(f"  Salidas de límites:        {out_of_bounds} ({out_of_bounds/n_episodes*100:.1f}%)")
    print(f"  Timeout (sin éxito):       {n_episodes - successes - out_of_bounds}")

    print(f"\n{'Recompensas':-^50}")
    print(f"  Media:                     {np.mean(rewards):.2f}")
    print(f"  Desviación estándar:       {np.std(rewards):.2f}")
    print(f"  Mínimo:                    {np.min(rewards):.2f}")
    print(f"  Máximo:                    {np.max(rewards):.2f}")
    print(f"  Mediana:                   {np.median(rewards):.2f}")

    print(f"\n{'Pasos por Episodio':-^50}")
    print(f"  Media:                     {np.mean(steps):.1f}")
    print(f"  Desviación estándar:       {np.std(steps):.1f}")
    print(f"  Mínimo:                    {np.min(steps)}")
    print(f"  Máximo:                    {np.max(steps)}")

    print(f"\n{'Distancias al Objetivo':-^50}")
    print(f"  Distancia mínima (media):  {np.mean(min_distances):.3f} m")
    print(f"  Distancia final (media):   {np.mean(final_distances):.3f} m")
    print(f"  Error angular final:       {np.mean(final_angles):.1f}°")
    print(f"  Velocidad final (media):   {np.mean(final_speeds):.3f} m/s")

    if successes > 0:
        print(f"\n{'Estadísticas de Episodios Exitosos':-^50}")
        print(f"  Recompensa media:          {np.mean(success_rewards):.2f}")
        print(f"  Pasos promedio:            {np.mean(success_steps):.1f}")
        print(f"  Pasos mínimos:             {np.min(success_steps)}")

    print("\n" + "="*70)


def evaluate(args):
    """Función principal de evaluación."""

    # Detectar algoritmo y configuración
    algo = args.algo if args.algo else detect_algorithm(args.model_path)
    continuous = algo == "sac"
    env_id = "ParkingEnv-v1" if continuous else "ParkingEnv-v0"

    print(f"\nCargando modelo: {args.model_path}")
    print(f"Algoritmo detectado: {algo.upper()}")
    print(f"Entorno: {env_id}")
    print(f"Ruido: {'Sí' if args.noise else 'No'}")

    # Cargar modelo
    model, algo = load_model(args.model_path, algo)

    # Crear entorno
    render_mode = "human" if args.render else None
    env = gym.make(
        env_id,
        reward_version=args.reward_version,
        noise=args.noise,
        render_mode=render_mode,
        continuous=continuous,
    )

    # Evaluar episodios
    results = []

    print(f"\nEvaluando {args.n_episodes} episodios...")

    for ep in range(args.n_episodes):
        result = evaluate_episode(model, env, deterministic=not args.stochastic)
        results.append(result)

        if args.verbose:
            status = "ÉXITO" if result["is_success"] else "FALLO"
            if result["out_of_bounds"]:
                status = "FUERA"
            print(
                f"  Ep {ep+1:3d}: {status:6s} | "
                f"R={result['total_reward']:7.2f} | "
                f"Pasos={result['steps']:3d} | "
                f"Dist={result['final_distance']:.3f}m | "
                f"Ang={result['final_angle_error_deg']:.1f}°"
            )

    # Imprimir estadísticas
    print_statistics(results)

    # Guardar resultados si se especifica
    if args.save_results:
        output_path = args.save_results
        save_data = {
            "model_path": args.model_path,
            "algorithm": algo,
            "n_episodes": args.n_episodes,
            "noise": args.noise,
            "reward_version": args.reward_version,
            "statistics": {
                "mean_reward": float(np.mean([r["total_reward"] for r in results])),
                "std_reward": float(np.std([r["total_reward"] for r in results])),
                "success_rate": float(sum(1 for r in results if r["is_success"]) / len(results)),
                "mean_steps": float(np.mean([r["steps"] for r in results])),
                "mean_final_distance": float(np.mean([r["final_distance"] for r in results])),
            },
            "episodes": [
                {
                    "reward": r["total_reward"],
                    "steps": r["steps"],
                    "is_success": r["is_success"],
                    "final_distance": r["final_distance"],
                    "final_angle_error_deg": r["final_angle_error_deg"],
                }
                for r in results
            ]
        }

        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResultados guardados en: {output_path}")

    env.close()

    return results


def compare_models(model_paths: List[str], args):
    """Compara múltiples modelos."""
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)

    all_results = {}

    for path in model_paths:
        print(f"\nEvaluando: {path}")
        args.model_path = path
        results = evaluate(args)

        successes = sum(1 for r in results if r["is_success"])
        mean_reward = np.mean([r["total_reward"] for r in results])

        all_results[path] = {
            "success_rate": successes / len(results),
            "mean_reward": mean_reward,
            "results": results
        }

    # Tabla comparativa
    print("\n" + "="*70)
    print("TABLA COMPARATIVA")
    print("="*70)
    print(f"{'Modelo':<40} {'Éxito %':>10} {'Recompensa':>12}")
    print("-"*70)

    for path, data in all_results.items():
        model_name = os.path.basename(os.path.dirname(path)) or path
        print(
            f"{model_name:<40} "
            f"{data['success_rate']*100:>9.1f}% "
            f"{data['mean_reward']:>12.2f}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de políticas de estacionamiento"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Ruta al modelo entrenado (.zip)"
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "sac", "dqn"],
        default=None,
        help="Algoritmo usado (auto-detectado si no se especifica)"
    )
    parser.add_argument(
        "--reward_version",
        type=int,
        choices=[1, 2],
        default=2,
        help="Versión de función de recompensa"
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Activar ruido ambiental en evaluación"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=50,
        help="Número de episodios a evaluar"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Visualizar episodios en tiempo real"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Usar política estocástica en lugar de determinística"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar resultados de cada episodio"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Guardar resultados en archivo JSON"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        default=None,
        help="Comparar múltiples modelos"
    )

    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare, args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
