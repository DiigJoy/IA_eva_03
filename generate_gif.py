"""
Script para generar GIFs de demostración del agente de estacionamiento.

Este script genera animaciones GIF mostrando episodios de estacionamiento
exitosos y fallidos para documentación y presentación.

Uso:
    python generate_gif.py --model_path logs/ppo_v2/best_model --output parking_demo.gif
    python generate_gif.py --model_path logs/sac_v2/best_model --n_episodes 5 --fps 15
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN
from PIL import Image

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


def render_frame_pygame(env) -> np.ndarray:
    """Renderiza un frame usando pygame."""
    import pygame

    # Acceder al entorno base (sin wrappers)
    base_env = env.unwrapped

    # Configuración de pantalla
    screen_width = 600
    screen_height = 600
    scale = screen_width / (base_env.x_max - base_env.x_min)

    # Crear superficie
    surface = pygame.Surface((screen_width, screen_height))

    # Colores
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 200, 0)
    DARK_GREEN = (0, 150, 0)
    RED = (200, 0, 0)
    BLUE = (0, 100, 200)
    GRAY = (200, 200, 200)
    LIGHT_GRAY = (230, 230, 230)
    YELLOW = (255, 200, 0)

    # Limpiar pantalla
    surface.fill(WHITE)

    def world_to_screen(wx, wy):
        sx = int((wx - base_env.x_min) * scale)
        sy = int((base_env.y_max - wy) * scale)
        return sx, sy

    # Dibujar grilla
    for i in range(-5, 6):
        start = world_to_screen(i, base_env.y_min)
        end = world_to_screen(i, base_env.y_max)
        pygame.draw.line(surface, LIGHT_GRAY, start, end, 1)
        start = world_to_screen(base_env.x_min, i)
        end = world_to_screen(base_env.x_max, i)
        pygame.draw.line(surface, LIGHT_GRAY, start, end, 1)

    # Dibujar zona de estacionamiento (más visible)
    goal_size = base_env.pos_tolerance * 2 * scale
    gx, gy = world_to_screen(base_env.goal_x, base_env.goal_y)

    # Fondo verde claro para la zona de parking
    goal_rect = pygame.Rect(
        gx - goal_size / 2, gy - goal_size / 2,
        goal_size, goal_size
    )
    pygame.draw.rect(surface, (200, 255, 200), goal_rect)
    pygame.draw.rect(surface, GREEN, goal_rect, 3)

    # Dibujar flecha de orientación objetivo
    arrow_length = 35
    end_x = gx + arrow_length * np.cos(-base_env.goal_theta)
    end_y = gy + arrow_length * np.sin(-base_env.goal_theta)
    pygame.draw.line(surface, DARK_GREEN, (gx, gy), (end_x, end_y), 3)

    # Punta de flecha
    angle = -base_env.goal_theta
    arrow_head_length = 10
    left_x = end_x - arrow_head_length * np.cos(angle - 0.5)
    left_y = end_y - arrow_head_length * np.sin(angle - 0.5)
    right_x = end_x - arrow_head_length * np.cos(angle + 0.5)
    right_y = end_y - arrow_head_length * np.sin(angle + 0.5)
    pygame.draw.polygon(surface, DARK_GREEN, [(end_x, end_y), (left_x, left_y), (right_x, right_y)])

    # Dibujar trayectoria
    if len(base_env.trajectory) > 1:
        points = [world_to_screen(p[0], p[1]) for p in base_env.trajectory]
        pygame.draw.lines(surface, BLUE, False, points, 2)

    # Dibujar vehículo
    if base_env.state is not None:
        x, y, speed, theta, omega = base_env.state
        cx, cy = world_to_screen(x, y)

        # Calcular esquinas del rectángulo del auto
        half_l = base_env.car_length * scale / 2
        half_w = base_env.car_width * scale / 2

        # Rotar esquinas
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        corners = [
            (half_l, half_w), (half_l, -half_w),
            (-half_l, -half_w), (-half_l, half_w)
        ]
        rotated = [
            (cx + c[0] * cos_t - c[1] * sin_t,
             cy + c[0] * sin_t + c[1] * cos_t)
            for c in corners
        ]

        # Dibujar auto
        pygame.draw.polygon(surface, RED, rotated)
        pygame.draw.polygon(surface, BLACK, rotated, 2)

        # Dibujar dirección (frente del auto)
        front_x = cx + half_l * 1.3 * cos_t
        front_y = cy + half_l * 1.3 * sin_t
        pygame.draw.line(surface, YELLOW, (cx, cy), (front_x, front_y), 4)

    # Añadir información de texto
    try:
        font = pygame.font.SysFont("Arial", 18, bold=True)
        small_font = pygame.font.SysFont("Arial", 14)
    except Exception:
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)

    if base_env.state is not None:
        distance = base_env._distance_to_goal()
        angle_error = abs(base_env._angle_diff(base_env.goal_theta, base_env.state[3]))

        # Título
        title = font.render("ParkingEnv-v0 Demo", True, BLACK)
        surface.blit(title, (screen_width // 2 - title.get_width() // 2, 10))

        # Panel de información
        info_y = 40
        info_texts = [
            f"Step: {base_env.steps}/{base_env.max_steps}",
            f"Distance: {distance:.2f}m",
            f"Angle: {np.degrees(angle_error):.1f}°",
            f"Speed: {abs(base_env.state[2]):.2f}m/s",
        ]

        for text in info_texts:
            text_surface = small_font.render(text, True, BLACK)
            surface.blit(text_surface, (10, info_y))
            info_y += 20

        # Indicador de estado
        if distance < base_env.pos_tolerance and angle_error < base_env.angle_tolerance:
            status = "PARKING!"
            status_color = GREEN
        elif distance < 1.0:
            status = "CLOSE"
            status_color = YELLOW
        else:
            status = "APPROACHING"
            status_color = BLUE

        status_text = font.render(status, True, status_color)
        surface.blit(status_text, (screen_width - status_text.get_width() - 10, 10))

    # Convertir a array numpy
    frame = np.transpose(
        np.array(pygame.surfarray.pixels3d(surface)),
        axes=(1, 0, 2)
    )

    return frame


def render_frame_matplotlib(env) -> np.ndarray:
    """Renderiza un frame usando matplotlib (fallback)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrow, Polygon
    import io

    # Acceder al entorno base (sin wrappers)
    base_env = env.unwrapped

    fig, ax = plt.subplots(figsize=(8, 8), dpi=75)
    ax.set_xlim(base_env.x_min, base_env.x_max)
    ax.set_ylim(base_env.y_min, base_env.y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')

    # Zona de estacionamiento
    goal_size = base_env.pos_tolerance * 2
    rect = Rectangle(
        (base_env.goal_x - goal_size / 2, base_env.goal_y - goal_size / 2),
        goal_size, goal_size,
        fill=True, facecolor='lightgreen', edgecolor="green", linewidth=2
    )
    ax.add_patch(rect)

    # Flecha de orientación objetivo
    ax.arrow(
        base_env.goal_x, base_env.goal_y,
        0.6 * np.cos(base_env.goal_theta), 0.6 * np.sin(base_env.goal_theta),
        head_width=0.15, head_length=0.1, fc="darkgreen", ec="darkgreen"
    )

    # Trayectoria
    if len(base_env.trajectory) > 1:
        xs = [p[0] for p in base_env.trajectory]
        ys = [p[1] for p in base_env.trajectory]
        ax.plot(xs, ys, "b-", linewidth=2, alpha=0.7)

    # Vehículo
    if base_env.state is not None:
        x, y, speed, theta, omega = base_env.state

        # Dibujar auto como rectángulo rotado
        half_l = base_env.car_length / 2
        half_w = base_env.car_width / 2

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ])

        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = corners @ rotation.T + np.array([x, y])

        car = Polygon(rotated, fill=True, facecolor='red', edgecolor='black', linewidth=2)
        ax.add_patch(car)

        # Flecha de orientación del auto
        ax.arrow(
            x, y,
            0.4 * cos_t, 0.4 * sin_t,
            head_width=0.12, head_length=0.08, fc="yellow", ec="orange", linewidth=2
        )

        # Información
        distance = base_env._distance_to_goal()
        angle_error = abs(base_env._angle_diff(base_env.goal_theta, theta))
        ax.set_title(
            f"Step: {base_env.steps} | Dist: {distance:.2f}m | Angle: {np.degrees(angle_error):.1f}°",
            fontsize=12
        )

    plt.tight_layout()

    # Convertir a array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    frame = np.array(img.convert('RGB'))
    plt.close(fig)

    return frame


def record_episode(
    model,
    env,
    use_pygame: bool = True,
    max_frames: int = 500
) -> tuple:
    """
    Graba un episodio completo.

    Returns:
        Tuple de (frames, is_success, total_reward)
    """
    frames = []
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    if use_pygame:
        import pygame
        pygame.init()

    while not done and len(frames) < max_frames:
        # Renderizar frame
        if use_pygame:
            try:
                frame = render_frame_pygame(env)
            except Exception:
                frame = render_frame_matplotlib(env)
        else:
            frame = render_frame_matplotlib(env)

        frames.append(frame)

        # Ejecutar acción
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Capturar frame final
    if use_pygame:
        try:
            frame = render_frame_pygame(env)
        except Exception:
            frame = render_frame_matplotlib(env)
    else:
        frame = render_frame_matplotlib(env)
    frames.append(frame)

    is_success = info.get("is_success", False)

    if use_pygame:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass

    return frames, is_success, total_reward


def create_gif(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 15,
    loop: int = 0
):
    """Crea un GIF a partir de una lista de frames."""
    images = [Image.fromarray(frame) for frame in frames]

    # Guardar GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=loop
    )

    print(f"GIF guardado: {output_path}")
    print(f"  - Frames: {len(frames)}")
    print(f"  - FPS: {fps}")
    print(f"  - Duración: {len(frames) / fps:.1f}s")


def generate_demo(args):
    """Genera GIFs de demostración."""

    # Detectar algoritmo y configuración
    algo = args.algo if args.algo else detect_algorithm(args.model_path)
    continuous = algo == "sac"
    env_id = "ParkingEnv-v1" if continuous else "ParkingEnv-v0"

    print(f"\nGenerando GIF de demostración")
    print(f"="*50)
    print(f"Modelo: {args.model_path}")
    print(f"Algoritmo: {algo.upper()}")
    print(f"Ruido: {'Sí' if args.noise else 'No'}")
    print(f"Episodios a grabar: {args.n_episodes}")
    print(f"="*50)

    # Cargar modelo
    model, algo = load_model(args.model_path, algo)

    # Crear entorno
    env = gym.make(
        env_id,
        reward_version=args.reward_version,
        noise=args.noise,
        render_mode=None,
        continuous=continuous,
    )

    # Determinar si usar pygame
    use_pygame = True
    try:
        import pygame
        pygame.init()
        pygame.quit()
    except Exception:
        use_pygame = False
        print("Pygame no disponible, usando matplotlib para renderizado")

    # Grabar episodios
    all_frames = []
    successes = 0
    best_episode = None
    best_reward = float('-inf')

    for ep in range(args.n_episodes):
        print(f"\nGrabando episodio {ep + 1}/{args.n_episodes}...", end=" ")

        frames, is_success, reward = record_episode(
            model, env, use_pygame=use_pygame
        )

        status = "ÉXITO" if is_success else "FALLO"
        print(f"{status} (R={reward:.1f}, {len(frames)} frames)")

        if is_success:
            successes += 1

        # Guardar mejor episodio
        if reward > best_reward:
            best_reward = reward
            best_episode = frames

        # Agregar separador entre episodios (frames en blanco)
        if args.n_episodes > 1:
            separator = np.ones_like(frames[0]) * 255
            all_frames.extend(frames)
            all_frames.extend([separator] * 5)  # 5 frames de pausa
        else:
            all_frames = frames

    # Generar GIF
    output_path = args.output
    if not output_path.endswith('.gif'):
        output_path += '.gif'

    print(f"\n{'='*50}")
    print("Generando GIF...")

    create_gif(all_frames, output_path, fps=args.fps)

    # Si hay múltiples episodios, también guardar el mejor
    if args.n_episodes > 1 and best_episode:
        best_output = output_path.replace('.gif', '_best.gif')
        create_gif(best_episode, best_output, fps=args.fps)

    print(f"\nResumen:")
    print(f"  - Episodios exitosos: {successes}/{args.n_episodes}")
    print(f"  - Mejor recompensa: {best_reward:.1f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generador de GIFs de demostración para ParkingEnv"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Ruta al modelo entrenado (.zip)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="parking_demo.gif",
        help="Nombre del archivo GIF de salida"
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
        help="Activar ruido ambiental"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=3,
        help="Número de episodios a grabar"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames por segundo del GIF"
    )

    args = parser.parse_args()
    generate_demo(args)


if __name__ == "__main__":
    main()
