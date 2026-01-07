"""
====================================================================
EVALUACIÓN DEL MODELO ENTRENADO PARA CarRacing-v3 (GYMNASIUM)
Versión final, limpia y defendible en clase.
====================================================================

Incluye:
✔ Evaluación determinista (epsilon = 0)
✔ Semilla fija para reproducibilidad
✔ Registro de métricas y resumen estadístico
✔ Gráfico de rewards
✔ GIF grande, nítido y reproducible (mejor episodio)
✔ Estructura profesional de resultados
====================================================================
"""

import argparse
import numpy as np
import imageio
import csv
import os
import logging
import random
from datetime import datetime
import cv2
import torch
import importlib

import src.configuracion as cfg
from src.entorno import crear_entorno
from src.preprocesamiento import (
    generar_stack_inicial,
    actualizar_stack,
    generar_state_frame_stack_from_queue
)

# ---------------------------------------------------------
# Espacio de acciones (idéntico al entrenamiento)
# ---------------------------------------------------------
ACTION_SPACE = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
    (-1, 1, 0),   (0, 1, 0),   (1, 1, 0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0, 0),   (0, 0, 0),   (1, 0, 0)
]

GIF_SCALE = 4
GIF_FPS = 15


# ---------------------------------------------------------
# Preparar carpeta de resultados y logging
# ---------------------------------------------------------
def preparar_carpeta_resultados(nombre_exp):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = f"resultados/{nombre_exp}_{timestamp}"
    os.makedirs(carpeta, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{carpeta}/evaluacion.log", encoding="utf-8")
        ]
    )

    logging.info(f"Carpeta creada: {carpeta}")
    return carpeta


def _cargar_agente_con_flags(action_space, usar_double, usar_dueling):
    """
    Fuerza flags globales y recarga el módulo src.agente para que
    NO se quede con valores antiguos importados como constantes.
    """
    cfg.USAR_DOUBLE_DQN = usar_double
    cfg.USAR_DUELING_DQN = usar_dueling

    # Importar y recargar agente DESPUÉS de setear cfg
    import src.agente as agente_mod
    importlib.reload(agente_mod)

    # Crear agente desde el módulo recargado
    agente = agente_mod.AgenteDQN(action_space=action_space)
    return agente


# ---------------------------------------------------------
# Evaluación principal (sin aprendizaje, ε = 0)
# ---------------------------------------------------------
def evaluar_modelo(path_modelo, episodios, grabar_gif, nombre_exp):
    carpeta = preparar_carpeta_resultados(nombre_exp)

    # ------------------ SEMILLA FIJA ------------------
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------ Detectar arquitectura por nombre del path ------------------
    path_lower = path_modelo.lower()
    usar_dueling = "dueling" in path_lower
    usar_double = "double" in path_lower

    logging.info("=========== INICIO DE EVALUACIÓN ===========")
    logging.info(f"Modelo: {path_modelo}")
    logging.info(f"Episodios: {episodios}")
    logging.info(f"GIF: {grabar_gif}")
    logging.info(f"Semilla fija: {SEED}")
    logging.info(
        f"Arquitectura detectada: "
        f"{'Double ' if usar_double else ''}"
        f"{'Dueling ' if usar_dueling else ''}"
        f"DQN"
    )

    env = crear_entorno()

    # ------------------ Agente (con flags correctos) ------------------
    agente = _cargar_agente_con_flags(
        action_space=ACTION_SPACE,
        usar_double=usar_double,
        usar_dueling=usar_dueling
    )
    logging.info(
        f"Configuracion aplicada (cfg): "
        f"{'Double ' if cfg.USAR_DOUBLE_DQN else ''}"
        f"{'Dueling ' if cfg.USAR_DUELING_DQN else ''}"
        f"DQN"
    )

    agente.epsilon = 0.0
    agente.load(path_modelo)

    resultados = []
    mejor_reward = -1e9
    frames_mejor = []

    GRABAR_SOLO_MEJOR = grabar_gif

    # ------------------ Loop de evaluación ------------------
    for ep in range(1, episodios + 1):
        obs, _ = env.reset()
        stack = generar_stack_inicial(obs)

        reward_total = 0
        frames_ep = [] if GRABAR_SOLO_MEJOR else None
        done = False
        t = 0

        while not done:
            estado = generar_state_frame_stack_from_queue(stack)
            accion = agente.seleccionar_accion(estado)

            obs, reward, terminated, truncated, _ = env.step(
                np.array(accion, dtype=np.float32)
            )

            done = terminated or truncated
            reward_total += reward

            if GRABAR_SOLO_MEJOR:
                frame_rgb = env.render()
                if frame_rgb.dtype != np.uint8:
                    frame_rgb = (frame_rgb * 255).astype(np.uint8)

                frame_rgb = cv2.resize(
                    frame_rgb,
                    (
                        frame_rgb.shape[1] * GIF_SCALE,
                        frame_rgb.shape[0] * GIF_SCALE
                    ),
                    interpolation=cv2.INTER_NEAREST
                )
                frames_ep.append(frame_rgb)

            stack = actualizar_stack(stack, obs)
            t += 1

        logging.info(
            f"Episodio {ep}/{episodios} | Reward={reward_total:.2f} | Frames={t}"
        )

        resultados.append([ep, reward_total, t])

        if reward_total > mejor_reward:
            mejor_reward = reward_total
            frames_mejor = frames_ep

    env.close()

    # ------------------ CSV ------------------
    csv_path = f"{carpeta}/metricas.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episodio", "reward_total", "frames"])
        writer.writerows(resultados)

    logging.info(f"CSV generado: {csv_path}")

    # ------------------ Resumen ------------------
    rewards = [r[1] for r in resultados]
    logging.info(
        f"Resumen | Avg={np.mean(rewards):.2f} | "
        f"Std={np.std(rewards):.2f} | "
        f"Max={np.max(rewards):.2f}"
    )

    # ------------------ Gráfico ------------------
    graficar_resultados(resultados, carpeta)

    # ------------------ GIF ------------------
    if grabar_gif and frames_mejor:
        gif_path = f"{carpeta}/mejor_episodio.gif"
        imageio.mimsave(gif_path, frames_mejor, fps=GIF_FPS, loop=0)
        logging.info(f"GIF generado: {gif_path}")

    logging.info("=========== FIN DE EVALUACIÓN ===========")


# ---------------------------------------------------------
# Gráfico de rewards
# ---------------------------------------------------------
def graficar_resultados(resultados, carpeta):
    import matplotlib.pyplot as plt

    episodios = [r[0] for r in resultados]
    rewards = [r[1] for r in resultados]

    mov_avg = np.convolve(rewards, np.ones(5) / 5, mode="valid")

    plt.figure(figsize=(10, 6))
    plt.plot(episodios, rewards, label="Reward por episodio", alpha=0.7)
    if len(episodios) >= 5:
        plt.plot(episodios[4:], mov_avg, label="Promedio móvil (5)", linewidth=3)
    plt.title("Evaluación del Modelo — CarRacing-v3")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    path = f"{carpeta}/grafico_rewards.png"
    plt.savefig(path)
    plt.close()

    logging.info(f"Gráfico guardado: {path}")


# ---------------------------------------------------------
# Entrada por línea de comandos
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación del modelo entrenado (CarRacing-v3)"
    )

    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-e", "--episodes", type=int, default=3)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--exp", type=str, default="evaluacion")

    args = parser.parse_args()

    evaluar_modelo(
        path_modelo=args.model,
        episodios=args.episodes,
        grabar_gif=args.gif,
        nombre_exp=args.exp
    )
