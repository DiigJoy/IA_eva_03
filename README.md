# ParkingEnv-v0: Agente de Aprendizaje por Refuerzo para Estacionamiento 2D

Proyecto de Aprendizaje por Refuerzo (RL) para entrenar un agente capaz de estacionar un vehículo en un entorno bidimensional utilizando Gymnasium y Stable-Baselines3.

## Descripción

Este proyecto implementa un entorno personalizado de Gymnasium donde un agente aprende a controlar un vehículo para estacionarlo correctamente en una zona objetivo. El agente debe:

- Navegar desde una posición inicial aleatoria hasta el objetivo
- Alinearse correctamente con la orientación del estacionamiento
- Detenerse con velocidad baja dentro de la zona de tolerancia

## Características

- **Entorno personalizado**: `ParkingEnv-v0` (discreto) y `ParkingEnv-v1` (continuo)
- **Múltiples algoritmos**: PPO, SAC, DQN
- **Reward shaping avanzado**: Dos versiones de función de recompensa
- **Robustez**: Soporte para ruido ambiental (viento, fricción variable, sensores)
- **Visualización**: Renderizado con Pygame o Matplotlib
- **Generación de GIFs**: Script para crear demostraciones animadas

## Estructura del Proyecto

```
IA_eva_03/
├── parking_env.py       # Entorno Gymnasium personalizado
├── train_parking.py     # Script de entrenamiento
├── evaluate_policy.py   # Script de evaluación
├── generate_gif.py      # Generador de GIFs de demostración
├── requirements.txt     # Dependencias del proyecto
├── informe.txt          # Informe técnico detallado
├── README.md            # Este archivo
└── logs/                # Modelos y logs de TensorBoard
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd IA_eva_03
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python -c "import gymnasium; import stable_baselines3; import parking_env; print('OK')"
```

## Uso

### Entrenamiento

#### Entrenar con PPO (acciones discretas)

```bash
python train_parking.py --algo ppo --reward_version 2 --total_timesteps 300000
```

#### Entrenar con SAC (acciones continuas)

```bash
python train_parking.py --algo sac --reward_version 2 --total_timesteps 300000
```

#### Entrenar con ruido ambiental (bonus robustez)

```bash
python train_parking.py --algo ppo --reward_version 2 --noise --total_timesteps 400000
```

#### Opciones de entrenamiento

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--algo` | Algoritmo: ppo, sac, dqn | ppo |
| `--reward_version` | Versión de recompensa: 1 (simple), 2 (shaped) | 2 |
| `--noise` | Activar ruido ambiental | False |
| `--total_timesteps` | Pasos de entrenamiento | 300000 |
| `--n_envs` | Entornos paralelos (PPO) | 8 |
| `--eval_freq` | Frecuencia de evaluación | 10000 |

### Evaluación

#### Evaluar un modelo entrenado

```bash
python evaluate_policy.py --model_path logs/<modelo>/best_model --n_episodes 50
```

#### Evaluar con visualización

```bash
python evaluate_policy.py --model_path logs/<modelo>/best_model --render --n_episodes 10
```

#### Comparar múltiples modelos

```bash
python evaluate_policy.py --model_path logs/ppo/best_model \
    --compare logs/sac/best_model logs/dqn/best_model --n_episodes 30
```

### Generación de GIFs

#### Generar demostración

```bash
python generate_gif.py --model_path logs/<modelo>/best_model --output demo.gif
```

#### Generar múltiples episodios

```bash
python generate_gif.py --model_path logs/<modelo>/best_model --n_episodes 5 --fps 20
```

### Monitoreo con TensorBoard

```bash
tensorboard --logdir logs/
```

## Descripción del Entorno

### Espacio de Observación (9 dimensiones)

| Variable | Descripción | Rango |
|----------|-------------|-------|
| x, y | Posición del vehículo | [-5, 5] |
| vx, vy | Velocidad en componentes | [-3, 3] |
| theta | Orientación (rad) | [-π, π] |
| omega | Velocidad angular | [-2, 2] |
| dx, dy | Distancia al objetivo | [-10, 10] |
| dtheta | Error angular al objetivo | [-π, π] |

### Espacio de Acciones

#### Discretas (ParkingEnv-v0)

| Acción | Descripción |
|--------|-------------|
| 0 | No-op (solo fricción) |
| 1 | Acelerar adelante |
| 2 | Frenar / Reversa |
| 3 | Adelante + Izquierda |
| 4 | Adelante + Derecha |
| 5 | Girar izquierda (lento) |
| 6 | Girar derecha (lento) |
| 7 | Reversa + Izquierda |
| 8 | Reversa + Derecha |

#### Continuas (ParkingEnv-v1)

| Dimensión | Descripción | Rango |
|-----------|-------------|-------|
| aceleración | Control de velocidad | [-1, 1] |
| giro | Control de dirección | [-1, 1] |

### Condiciones de Término

- **Éxito**: Distancia < 0.4m, error angular < 14°, velocidad < 0.3 m/s
- **Fallo**: Salir de los límites del mapa
- **Truncado**: Superar 300 pasos

### Función de Recompensa

#### Versión 1 (Simple)
- Penalización proporcional a la distancia
- Bonus de +100 por estacionar exitosamente
- Penalización de -30 por salir de límites

#### Versión 2 (Shaped - Recomendada)
- Mejora en distancia (potential-based shaping)
- Mejora en alineación angular
- Penalización por velocidad excesiva cerca del objetivo
- Bonus por proximidad y orientación correcta
- Costo por tiempo (eficiencia)
- Recompensa terminal escalada por calidad del estacionamiento

### Ruido Ambiental (Bonus Robustez)

Cuando se activa `--noise`:

1. **Viento lateral**: Fuerza aleatoria que cambia gradualmente
2. **Fricción variable**: Coeficiente diferente en cada episodio
3. **Ruido en sensores**: Perturbación gaussiana en observaciones

## Arquitectura de Red Neural

- **Política (Actor)**: MLP [256, 256] neuronas
- **Valor (Critic)**: MLP [256, 256] neuronas
- **Activación**: ReLU

## Hiperparámetros Optimizados

### PPO
```python
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
```

### SAC
```python
learning_rate: 3e-4
buffer_size: 100000
batch_size: 256
tau: 0.005
gamma: 0.99
ent_coef: auto
```

## Resultados Esperados

Con 300,000 pasos de entrenamiento:

| Algoritmo | Tasa de Éxito | Recompensa Media |
|-----------|---------------|------------------|
| PPO | 70-85% | 80-120 |
| SAC | 75-90% | 90-130 |
| DQN | 50-70% | 40-80 |

*Nota: Los resultados pueden variar según la semilla aleatoria y configuración.*

## Solución de Problemas

### El entrenamiento no converge

1. Aumentar `total_timesteps` a 500,000+
2. Usar `reward_version=2` para mejor shaping
3. Ajustar `ent_coef` para más exploración

### Pygame no funciona

El sistema automáticamente usa Matplotlib como fallback para visualización.

### Error de memoria

Reducir `n_envs` en PPO o `buffer_size` en SAC/DQN.

## Referencias

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)

## Licencia

Este proyecto fue desarrollado como parte de una evaluación académica de Inteligencia Artificial.
