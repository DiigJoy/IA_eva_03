import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ParkingEnv(gym.Env):
    """
    Entorno 2D simplificado de estacionamiento.
    Estado:
        [x, y, vx, vy, theta, omega, dx, dy, dtheta]
    Acciones discretas:
        0: no-op
        1: acelerar recto
        2: frenar
        3: acelerar + girar izquierda
        4: acelerar + girar derecha
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, reward_version=2, noise=False):
        super().__init__()

        self.render_mode = render_mode
        self.reward_version = reward_version
        self.noise = noise

        # Mundo
        self.x_min, self.x_max = -5.0, 5.0
        self.y_min, self.y_max = -5.0, 5.0
        self.dt = 0.1
        self.max_speed = 5.0
        self.max_omega = 2.0
        self.max_steps = 200

        # Objetivo (posición y orientación de estacionamiento)
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_theta = 0.0

        # Espacio de observaciones
        low = np.array(
            [
                self.x_min, self.y_min,          # x, y
                -self.max_speed, -self.max_speed,  # vx, vy
                -np.pi, -self.max_omega,        # theta, omega
                2 * self.x_min, 2 * self.y_min, # dx, dy (margen amplio)
                -np.pi                          # dtheta
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.x_max, self.y_max,
                self.max_speed, self.max_speed,
                np.pi, self.max_omega,
                2 * self.x_max, 2 * self.y_max,
                np.pi
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Acciones discretas: 5 acciones posibles
        self.action_space = spaces.Discrete(5)

        # Estado interno: [x, y, speed, theta, omega]
        self.state = None
        self.steps = 0
        self.prev_distance = None
        self.friction = 0.1  # se ajusta por ruido

        # Render
        self._fig = None
        self._ax = None

    @staticmethod
    def _angle_diff(target, current):
        """Diferencia angular normalizada a [-pi, pi]."""
        diff = (target - current + np.pi) % (2 * np.pi) - np.pi
        return diff

    def _distance_to_goal(self):
        x, y, speed, theta, omega = self.state
        dx = self.goal_x - x
        dy = self.goal_y - y
        return np.sqrt(dx * dx + dy * dy)

    def _get_obs(self):
        x, y, speed, theta, omega = self.state
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        dx = self.goal_x - x
        dy = self.goal_y - y
        dtheta = self._angle_diff(self.goal_theta, theta)

        obs = np.array([x, y, vx, vy, theta, omega, dx, dy, dtheta], dtype=np.float32)

        if self.noise:
            # Ruido gaussiano pequeño en sensores
            obs += self.np_random.normal(0.0, 0.01, size=obs.shape).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Fricción variable por episodio (ruido de entorno)
        if self.noise:
            self.friction = 0.05 + self.np_random.uniform(0.0, 0.15)  # [0.05, 0.2]
        else:
            self.friction = 0.1

        # Posición inicial: parte baja del mapa
        x = self.np_random.uniform(-4.0, 4.0)
        y = self.np_random.uniform(-4.0, -1.0)
        speed = 0.0
        theta = self.np_random.uniform(-np.pi, np.pi)
        omega = 0.0

        self.state = np.array([x, y, speed, theta, omega], dtype=np.float32)
        self.prev_distance = self._distance_to_goal()

        obs = self._get_obs()
        info = {}
        return obs, info

    def _compute_reward(self, distance, angle_error, speed, success, out_of_bounds, truncated):
        # Versión 1: recompensa simple (solo distancia + bonus/penalty)
        if self.reward_version == 1:
            reward = -distance  # mientras más cerca, menos negativo

            if success:
                reward += 50.0
            if out_of_bounds:
                reward -= 20.0
            if truncated and not success:
                reward -= 5.0

            return float(reward)

        # Versión 2: reward shaping más elaborado
        reward = 0.0

        # Mejora en distancia como potencial (distancia anterior - distancia actual)
        delta_d = self.prev_distance - distance
        reward += 5.0 * delta_d
        self.prev_distance = distance

        # Penalizar error angular
        reward -= 0.1 * angle_error

        # Penalizar velocidad alta (especialmente cerca del objetivo)
        reward -= 0.05 * (speed ** 2)

        # Pequeño living-cost para incentivar estacionar rápido
        reward -= 0.01

        if success:
            reward += 100.0
        if out_of_bounds:
            reward -= 50.0
        if truncated and not success:
            reward -= 10.0

        return float(reward)

    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"

        x, y, speed, theta, omega = self.state

        # Decodificar acción discreta a aceleración y giro
        accel = 0.0
        steer = 0.0

        if action == 0:
            # no-op: solo fricción
            accel = 0.0
            steer = 0.0
        elif action == 1:
            # acelerar recto
            accel = 2.0
            steer = 0.0
        elif action == 2:
            # frenar
            accel = -3.0
            steer = 0.0
        elif action == 3:
            # avanzar y girar izquierda
            accel = 1.5
            steer = +1.0
        elif action == 4:
            # avanzar y girar derecha
            accel = 1.5
            steer = -1.0

        # Actualizar dinámica simplificada
        speed = speed + accel * self.dt
        # fricción (no dejar velocidad negativa)
        speed = speed * (1.0 - self.friction * self.dt)
        speed = np.clip(speed, 0.0, self.max_speed)

        omega = np.clip(steer, -self.max_omega, self.max_omega)
        theta = theta + omega * self.dt
        theta = self._angle_diff(theta, 0.0)  # normalizar alrededor de 0

        x = x + speed * np.cos(theta) * self.dt
        y = y + speed * np.sin(theta) * self.dt

        self.state = np.array([x, y, speed, theta, omega], dtype=np.float32)
        self.steps += 1

        # Calcular métricas
        distance = self._distance_to_goal()
        angle_error = abs(self._angle_diff(self.goal_theta, theta))

        # Condiciones de término
        success = (
            distance < 0.3 and
            angle_error < 0.2 and  # ~11 grados
            speed < 0.2
        )

        out_of_bounds = (
            x < self.x_min or x > self.x_max or
            y < self.y_min or y > self.y_max
        )

        terminated = success or out_of_bounds
        truncated = self.steps >= self.max_steps

        reward = self._compute_reward(
            distance=distance,
            angle_error=angle_error,
            speed=speed,
            success=success,
            out_of_bounds=out_of_bounds,
            truncated=truncated,
        )

        obs = self._get_obs()
        info = {
            "distance": float(distance),
            "angle_error": float(angle_error),
            "is_success": bool(success),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()
            plt.ion()

        ax = self._ax
        ax.clear()
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect("equal")

        # Dibujar zona de estacionamiento
        goal_size = 0.6
        rect = Rectangle(
            (self.goal_x - goal_size / 2, self.goal_y - goal_size / 2),
            goal_size,
            goal_size,
            fill=False,
            edgecolor="green",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Dibujar auto
        x, y, speed, theta, omega = self.state
        ax.plot(self.goal_x, self.goal_y, "gx")
        ax.plot(x, y, "ro")

        # Flecha indicando orientación
        dx = 0.5 * np.cos(theta)
        dy = 0.5 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.1, length_includes_head=True)

        ax.set_title("ParkingEnv-v0")

        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        super().close()


# Registrar el entorno con Gymnasium
from gymnasium.envs.registration import register

try:
    register(
        id="ParkingEnv-v0",
        entry_point="parking_env:ParkingEnv",
    )
except Exception:
    # Si ya está registrado, ignorar
    pass
