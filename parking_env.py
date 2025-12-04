"""
ParkingEnv-v0: Entorno personalizado de Gymnasium para estacionamiento 2D.

Este entorno implementa un simulador de estacionamiento donde un agente debe
aprender a controlar un vehículo para estacionarlo en una zona objetivo.

Características:
- Espacios de observación y acción continuos/discretos
- Función de recompensa configurable (reward shaping)
- Ruido ambiental (viento lateral, fricción variable, ruido en sensores)
- Visualización con Pygame o Matplotlib

Autor: Proyecto IA - Evaluación 03
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class ParkingEnv(gym.Env):
    """
    Entorno 2D de estacionamiento con física simplificada.

    Estado (observación):
        [x, y, vx, vy, theta, omega, dx, dy, dtheta]
        - (x, y): Posición del vehículo
        - (vx, vy): Velocidad en componentes x, y
        - theta: Orientación del vehículo (radianes)
        - omega: Velocidad angular
        - (dx, dy): Distancia al objetivo
        - dtheta: Diferencia angular respecto al objetivo

    Acciones (continuas):
        [aceleración, giro]
        - aceleración: [-1, 1] -> [-max_accel, max_accel]
        - giro: [-1, 1] -> [-max_steer, max_steer]

    Acciones (discretas):
        0: no-op
        1: acelerar hacia adelante
        2: frenar / reversa
        3: acelerar + girar izquierda
        4: acelerar + girar derecha
        5: girar izquierda (sin acelerar)
        6: girar derecha (sin acelerar)
        7: reversa + girar izquierda
        8: reversa + girar derecha
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_version: int = 2,
        noise: bool = False,
        continuous: bool = False,
        wind_strength: float = 0.3,
        sensor_noise_std: float = 0.02,
        friction_range: Tuple[float, float] = (0.05, 0.20),
    ):
        """
        Inicializa el entorno de estacionamiento.

        Args:
            render_mode: Modo de renderizado ("human", "rgb_array", None)
            reward_version: Versión de función de recompensa (1=simple, 2=shaped)
            noise: Activar ruido ambiental (viento, fricción variable, sensores)
            continuous: Usar acciones continuas (True) o discretas (False)
            wind_strength: Fuerza máxima del viento lateral
            sensor_noise_std: Desviación estándar del ruido en sensores
            friction_range: Rango de fricción variable (min, max)
        """
        super().__init__()

        self.render_mode = render_mode
        self.reward_version = reward_version
        self.noise = noise
        self.continuous = continuous
        self.wind_strength = wind_strength
        self.sensor_noise_std = sensor_noise_std
        self.friction_range = friction_range

        # Parámetros del mundo
        self.x_min, self.x_max = -5.0, 5.0
        self.y_min, self.y_max = -5.0, 5.0
        self.dt = 0.1  # Paso de tiempo
        self.max_speed = 3.0  # Velocidad máxima
        self.max_omega = 2.0  # Velocidad angular máxima
        self.max_accel = 3.0  # Aceleración máxima
        self.max_steer = 2.0  # Giro máximo del volante
        self.max_steps = 300  # Pasos máximos por episodio

        # Parámetros del vehículo
        self.car_length = 0.8
        self.car_width = 0.4

        # Objetivo de estacionamiento (centro del mapa, orientación horizontal)
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_theta = 0.0  # Apuntando hacia la derecha

        # Tolerancias para considerar éxito
        self.pos_tolerance = 0.4  # metros
        self.angle_tolerance = 0.25  # ~14 grados
        self.speed_tolerance = 0.3  # m/s

        # Espacio de observaciones (9 dimensiones)
        obs_low = np.array([
            self.x_min, self.y_min,           # x, y
            -self.max_speed, -self.max_speed,  # vx, vy
            -np.pi, -self.max_omega,           # theta, omega
            2 * self.x_min, 2 * self.y_min,    # dx, dy (margen amplio)
            -np.pi                              # dtheta
        ], dtype=np.float32)

        obs_high = np.array([
            self.x_max, self.y_max,
            self.max_speed, self.max_speed,
            np.pi, self.max_omega,
            2 * self.x_max, 2 * self.y_max,
            np.pi
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Espacio de acciones
        if self.continuous:
            # Acciones continuas: [aceleración, giro] en [-1, 1]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Acciones discretas: 9 acciones posibles
            self.action_space = spaces.Discrete(9)

        # Estado interno: [x, y, speed, theta, omega]
        self.state: Optional[np.ndarray] = None
        self.steps = 0
        self.prev_distance = 0.0
        self.prev_angle_error = 0.0
        self.friction = 0.1
        self.wind_force = np.array([0.0, 0.0])

        # Pygame para renderizado
        self._screen = None
        self._clock = None
        self._font = None

        # Historial para análisis
        self.trajectory: list = []

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normaliza un ángulo al rango [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        """Calcula la diferencia angular normalizada entre dos ángulos."""
        diff = target - current
        return ParkingEnv._normalize_angle(diff)

    def _distance_to_goal(self) -> float:
        """Calcula la distancia euclidiana al objetivo."""
        x, y = self.state[0], self.state[1]
        return np.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)

    def _get_obs(self) -> np.ndarray:
        """Construye el vector de observación."""
        x, y, speed, theta, omega = self.state

        # Velocidad en componentes
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)

        # Distancia y ángulo al objetivo
        dx = self.goal_x - x
        dy = self.goal_y - y
        dtheta = self._angle_diff(self.goal_theta, theta)

        obs = np.array([x, y, vx, vy, theta, omega, dx, dy, dtheta], dtype=np.float32)

        # Agregar ruido a sensores si está activado
        if self.noise:
            noise = self.np_random.normal(0.0, self.sensor_noise_std, size=obs.shape)
            obs = obs + noise.astype(np.float32)

        return obs

    def _apply_wind(self) -> np.ndarray:
        """Genera fuerza de viento lateral aleatoria."""
        if not self.noise:
            return np.array([0.0, 0.0])

        # Viento con componente aleatoria que cambia gradualmente
        wind_change = self.np_random.normal(0.0, 0.1, size=2)
        self.wind_force = 0.9 * self.wind_force + 0.1 * wind_change
        self.wind_force = np.clip(
            self.wind_force, -self.wind_strength, self.wind_strength
        )
        return self.wind_force

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno para un nuevo episodio."""
        super().reset(seed=seed)
        self.steps = 0
        self.trajectory = []

        # Fricción variable por episodio
        if self.noise:
            self.friction = self.np_random.uniform(*self.friction_range)
            self.wind_force = np.array([0.0, 0.0])
        else:
            self.friction = 0.1

        # Posición inicial: parte inferior del mapa con orientación aleatoria
        x = self.np_random.uniform(-3.5, 3.5)
        y = self.np_random.uniform(-4.5, -2.0)
        speed = 0.0
        theta = self.np_random.uniform(-np.pi, np.pi)
        omega = 0.0

        self.state = np.array([x, y, speed, theta, omega], dtype=np.float32)
        self.prev_distance = self._distance_to_goal()
        self.prev_angle_error = abs(self._angle_diff(self.goal_theta, theta))

        # Guardar posición inicial en trayectoria
        self.trajectory.append((x, y, theta))

        obs = self._get_obs()
        info = {
            "friction": self.friction,
            "initial_distance": self.prev_distance,
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def _decode_action(self, action) -> Tuple[float, float]:
        """Convierte la acción al formato (aceleración, giro)."""
        if self.continuous:
            # Escalar de [-1, 1] a rangos reales
            accel = float(action[0]) * self.max_accel
            steer = float(action[1]) * self.max_steer
            return accel, steer

        # Acciones discretas
        action_map = {
            0: (0.0, 0.0),      # no-op
            1: (2.5, 0.0),      # acelerar adelante
            2: (-2.0, 0.0),     # frenar/reversa
            3: (2.0, 1.5),      # adelante + izquierda
            4: (2.0, -1.5),     # adelante + derecha
            5: (0.5, 1.5),      # girar izquierda (lento)
            6: (0.5, -1.5),     # girar derecha (lento)
            7: (-1.5, 1.0),     # reversa + izquierda
            8: (-1.5, -1.0),    # reversa + derecha
        }
        # Convertir numpy array a int si es necesario
        action_key = int(action) if hasattr(action, 'item') else action
        return action_map.get(action_key, (0.0, 0.0))

    def _compute_reward(
        self,
        distance: float,
        angle_error: float,
        speed: float,
        success: bool,
        out_of_bounds: bool,
        truncated: bool
    ) -> float:
        """
        Calcula la recompensa según la versión configurada.

        Versión 1: Recompensa simple basada en distancia
        Versión 2: Reward shaping elaborado con múltiples factores
        """
        if self.reward_version == 1:
            # Versión simple
            reward = -distance * 0.5  # Penalizar distancia

            if success:
                reward += 100.0
            if out_of_bounds:
                reward -= 30.0
            if truncated and not success:
                reward -= 10.0

            return float(reward)

        # Versión 2: Reward shaping avanzado
        reward = 0.0

        # 1. Mejora en distancia (potential-based shaping)
        delta_distance = self.prev_distance - distance
        reward += 8.0 * delta_distance

        # 2. Mejora en ángulo
        delta_angle = self.prev_angle_error - angle_error
        reward += 3.0 * delta_angle

        # Actualizar valores previos
        self.prev_distance = distance
        self.prev_angle_error = angle_error

        # 3. Penalizar velocidad excesiva (especialmente cerca del objetivo)
        if distance < 1.0:
            speed_penalty = 0.3 * (speed ** 2)
        else:
            speed_penalty = 0.05 * (speed ** 2)
        reward -= speed_penalty

        # 4. Bonus por estar cerca y bien orientado
        if distance < 1.0 and angle_error < 0.5:
            reward += 2.0 * (1.0 - distance) * (1.0 - angle_error / 0.5)

        # 5. Costo por tiempo (incentivar eficiencia)
        reward -= 0.02

        # 6. Recompensas/penalizaciones terminales
        if success:
            # Bonus escalado por qué tan bien se estacionó
            parking_quality = max(0, 1.0 - distance / self.pos_tolerance)
            angle_quality = max(0, 1.0 - angle_error / self.angle_tolerance)
            speed_quality = max(0, 1.0 - speed / self.speed_tolerance)
            quality_bonus = 50.0 * (parking_quality + angle_quality + speed_quality) / 3
            reward += 100.0 + quality_bonus

        if out_of_bounds:
            reward -= 50.0

        if truncated and not success:
            # Penalización proporcional a qué tan lejos quedó
            reward -= 20.0 * (1.0 + distance / 5.0)

        return float(reward)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ejecuta un paso de simulación."""
        assert self.action_space.contains(action), f"Acción inválida: {action}"

        x, y, speed, theta, omega = self.state

        # Decodificar acción
        accel, steer = self._decode_action(action)

        # Aplicar viento lateral
        wind = self._apply_wind()

        # Actualizar física del vehículo

        # Velocidad angular (giro del volante afecta según velocidad)
        omega = steer * (0.5 + 0.5 * min(abs(speed), 1.0))
        omega = np.clip(omega, -self.max_omega, self.max_omega)

        # Actualizar orientación
        theta = theta + omega * self.dt
        theta = self._normalize_angle(theta)

        # Actualizar velocidad
        speed = speed + accel * self.dt

        # Aplicar fricción
        friction_force = self.friction * np.sign(speed) * (speed ** 2)
        speed = speed - friction_force * self.dt

        # Limitar velocidad (permite reversa)
        speed = np.clip(speed, -self.max_speed * 0.5, self.max_speed)

        # Actualizar posición
        x = x + speed * np.cos(theta) * self.dt + wind[0] * self.dt
        y = y + speed * np.sin(theta) * self.dt + wind[1] * self.dt

        # Actualizar estado
        self.state = np.array([x, y, speed, theta, omega], dtype=np.float32)
        self.steps += 1

        # Guardar en trayectoria
        self.trajectory.append((x, y, theta))

        # Calcular métricas
        distance = self._distance_to_goal()
        angle_error = abs(self._angle_diff(self.goal_theta, theta))
        abs_speed = abs(speed)

        # Condiciones de término
        success = (
            distance < self.pos_tolerance and
            angle_error < self.angle_tolerance and
            abs_speed < self.speed_tolerance
        )

        out_of_bounds = (
            x < self.x_min or x > self.x_max or
            y < self.y_min or y > self.y_max
        )

        terminated = success or out_of_bounds
        truncated = self.steps >= self.max_steps

        # Calcular recompensa
        reward = self._compute_reward(
            distance=distance,
            angle_error=angle_error,
            speed=abs_speed,
            success=success,
            out_of_bounds=out_of_bounds,
            truncated=truncated
        )

        obs = self._get_obs()
        info = {
            "distance": float(distance),
            "angle_error": float(angle_error),
            "speed": float(abs_speed),
            "is_success": bool(success),
            "out_of_bounds": bool(out_of_bounds),
            "steps": self.steps,
            "wind": self.wind_force.tolist() if self.noise else [0, 0],
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renderiza el entorno."""
        if self.render_mode is None:
            return None

        try:
            import pygame
            return self._render_pygame()
        except ImportError:
            return self._render_matplotlib()

    def _render_pygame(self) -> Optional[np.ndarray]:
        """Renderiza usando Pygame."""
        import pygame

        # Configuración de pantalla
        screen_width = 600
        screen_height = 600
        scale = screen_width / (self.x_max - self.x_min)

        if self._screen is None:
            pygame.init()
            pygame.display.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self._screen = pygame.Surface((screen_width, screen_height))
            pygame.display.set_caption("ParkingEnv-v0")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("Arial", 16)

        # Colores
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREEN = (0, 200, 0)
        RED = (200, 0, 0)
        BLUE = (0, 100, 200)
        GRAY = (150, 150, 150)
        YELLOW = (255, 255, 0)

        # Limpiar pantalla
        self._screen.fill(WHITE)

        # Función para convertir coordenadas del mundo a pantalla
        def world_to_screen(wx, wy):
            sx = int((wx - self.x_min) * scale)
            sy = int((self.y_max - wy) * scale)  # Invertir Y
            return sx, sy

        # Dibujar grilla
        for i in range(-5, 6):
            start = world_to_screen(i, self.y_min)
            end = world_to_screen(i, self.y_max)
            pygame.draw.line(self._screen, GRAY, start, end, 1)
            start = world_to_screen(self.x_min, i)
            end = world_to_screen(self.x_max, i)
            pygame.draw.line(self._screen, GRAY, start, end, 1)

        # Dibujar zona de estacionamiento
        goal_size = self.pos_tolerance * 2 * scale
        gx, gy = world_to_screen(self.goal_x, self.goal_y)
        goal_rect = pygame.Rect(
            gx - goal_size / 2, gy - goal_size / 2,
            goal_size, goal_size
        )
        pygame.draw.rect(self._screen, GREEN, goal_rect, 3)

        # Dibujar flecha de orientación objetivo
        arrow_length = 30
        end_x = gx + arrow_length * np.cos(-self.goal_theta)
        end_y = gy + arrow_length * np.sin(-self.goal_theta)
        pygame.draw.line(self._screen, GREEN, (gx, gy), (end_x, end_y), 2)

        # Dibujar trayectoria
        if len(self.trajectory) > 1:
            points = [world_to_screen(p[0], p[1]) for p in self.trajectory]
            pygame.draw.lines(self._screen, BLUE, False, points, 2)

        # Dibujar vehículo
        if self.state is not None:
            x, y, speed, theta, omega = self.state
            cx, cy = world_to_screen(x, y)

            # Calcular esquinas del rectángulo del auto
            half_l = self.car_length * scale / 2
            half_w = self.car_width * scale / 2

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
            pygame.draw.polygon(self._screen, RED, rotated)
            pygame.draw.polygon(self._screen, BLACK, rotated, 2)

            # Dibujar dirección
            front_x = cx + half_l * 1.5 * cos_t
            front_y = cy + half_l * 1.5 * sin_t
            pygame.draw.line(self._screen, YELLOW, (cx, cy), (front_x, front_y), 3)

        # Mostrar información
        if self.state is not None:
            distance = self._distance_to_goal()
            angle_error = abs(self._angle_diff(self.goal_theta, self.state[3]))

            info_texts = [
                f"Step: {self.steps}/{self.max_steps}",
                f"Distance: {distance:.2f}m",
                f"Angle Error: {np.degrees(angle_error):.1f}deg",
                f"Speed: {abs(self.state[2]):.2f}m/s",
            ]

            if self.noise:
                info_texts.append(f"Friction: {self.friction:.2f}")
                info_texts.append(f"Wind: ({self.wind_force[0]:.2f}, {self.wind_force[1]:.2f})")

            for i, text in enumerate(info_texts):
                surface = self._font.render(text, True, BLACK)
                self._screen.blit(surface, (10, 10 + i * 20))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)),
                axes=(1, 0, 2)
            )

    def _render_matplotlib(self) -> Optional[np.ndarray]:
        """Renderiza usando Matplotlib (fallback)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrow
        import io

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Zona de estacionamiento
        goal_size = self.pos_tolerance * 2
        rect = Rectangle(
            (self.goal_x - goal_size / 2, self.goal_y - goal_size / 2),
            goal_size, goal_size,
            fill=False, edgecolor="green", linewidth=2
        )
        ax.add_patch(rect)

        # Flecha de orientación objetivo
        ax.arrow(
            self.goal_x, self.goal_y,
            0.5 * np.cos(self.goal_theta), 0.5 * np.sin(self.goal_theta),
            head_width=0.1, head_length=0.05, fc="green", ec="green"
        )

        # Trayectoria
        if len(self.trajectory) > 1:
            xs = [p[0] for p in self.trajectory]
            ys = [p[1] for p in self.trajectory]
            ax.plot(xs, ys, "b-", linewidth=1, alpha=0.5)

        # Vehículo
        if self.state is not None:
            x, y, speed, theta, omega = self.state
            ax.plot(x, y, "ro", markersize=10)

            # Flecha de orientación
            ax.arrow(
                x, y,
                0.5 * np.cos(theta), 0.5 * np.sin(theta),
                head_width=0.15, head_length=0.1, fc="red", ec="red"
            )

        # Información
        distance = self._distance_to_goal() if self.state is not None else 0
        ax.set_title(f"Step: {self.steps} | Distance: {distance:.2f}m")

        if self.render_mode == "human":
            plt.pause(0.001)
            plt.show(block=False)
            return None
        else:
            # Convertir a array RGB
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            from PIL import Image
            img = Image.open(buf)
            plt.close(fig)
            return np.array(img)

    def close(self):
        """Cierra el entorno y libera recursos."""
        if self._screen is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._screen = None
            self._clock = None
            self._font = None
        super().close()

    def get_trajectory(self) -> list:
        """Retorna la trayectoria del episodio actual."""
        return self.trajectory.copy()


# Registrar el entorno con Gymnasium
from gymnasium.envs.registration import register

# Registrar versión discreta
try:
    register(
        id="ParkingEnv-v0",
        entry_point="parking_env:ParkingEnv",
        kwargs={"continuous": False},
    )
except Exception:
    pass

# Registrar versión continua para SAC
try:
    register(
        id="ParkingEnv-v1",
        entry_point="parking_env:ParkingEnv",
        kwargs={"continuous": True},
    )
except Exception:
    pass
