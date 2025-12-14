"""
====================================================================
EVALUACIÓN DEL MODELO ENTRENADO PARA CarRacing-v3 (GYMNASIUM)
Versión final, limpia y defendible en clase.
====================================================================

Incluye:
✔ Evaluación determinista (epsilon = 0)
✔ Registro de métricas
✔ Gráfico de rewards
✔ GIF grande, nítido y reproducible
✔ Estructura profesional de resultados
====================================================================
"""

import argparse
import numpy as np
import imageio
import csv
import os
import logging
from datetime import datetime
import cv2

from src.entorno import crear_entorno
from src.agente import AgenteDQN
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

GIF_SCALE = 6
GIF_FPS = 20



# Preparar carpeta de resultados
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


# Evaluación principal
def evaluar_modelo(path_modelo, episodios, grabar_gif, nombre_exp):
    carpeta = preparar_carpeta_resultados(nombre_exp)

    logging.info("=========== INICIO DE EVALUACIÓN ===========")
    logging.info(f"Modelo: {path_modelo}")
    logging.info(f"Episodios: {episodios}")
    logging.info(f"GIF: {grabar_gif}")

    env = crear_entorno()

    agente = AgenteDQN(action_space=ACTION_SPACE)
    agente.epsilon = 0.0
    agente.load(path_modelo)

    resultados = []
    mejor_reward = -1e9
    frames_mejor = []

    for ep in range(1, episodios + 1):
        obs, _ = env.reset()
        stack = generar_stack_inicial(obs)

        reward_total = 0
        frames_ep = []
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

            # ------------------ FRAME REAL DEL ENTORNO ------------------
            if grabar_gif:
                frame_rgb = env.render()  # ESTE es el frame bueno

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
            # ------------------------------------------------------------

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


    # genera CSV
    csv_path = f"{carpeta}/metricas.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episodio", "reward_total", "frames"])
        writer.writerows(resultados)

    logging.info(f"CSV generado: {csv_path}")


    # Graficos
    graficar_resultados(resultados, carpeta)

    # GIF FINAL
    if grabar_gif and frames_mejor:
        gif_path = f"{carpeta}/mejor_episodio.gif"
        imageio.mimsave(
            gif_path,
            frames_mejor,
            fps=GIF_FPS,
            loop=0
        )
        logging.info(f"GIF generado: {gif_path}")

    logging.info("=========== FIN DE EVALUACIÓN ===========")



# Gráfico de rewards
def graficar_resultados(resultados, carpeta):
    import matplotlib.pyplot as plt

    episodios = [r[0] for r in resultados]
    rewards = [r[1] for r in resultados]

    mov_avg = np.convolve(rewards, np.ones(5) / 5, mode="valid")

    plt.figure(figsize=(10, 6))
    plt.plot(episodios, rewards, label="Reward por episodio", alpha=0.7)
    plt.plot(episodios[4:], mov_avg, label="Promedio móvil (5)", linewidth=3)
    plt.title("Evaluación del Modelo DQN — CarRacing-v3")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    path = f"{carpeta}/grafico_rewards.png"
    plt.savefig(path)
    plt.close()

    logging.info(f"Gráfico guardado: {path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación del modelo DQN (CarRacing-v3)"
    )

    parser.add_argument("-m", "--model", required=True, help="Ruta al modelo .pth")
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
