# =============================================================
# MÓDULO DE ENTRENAMIENTO DEL AGENTE DQN PARA CarRacing-v3
# -------------------------------------------------------------
# Este archivo implementa el ciclo de Aprendizaje por Refuerzo:
#
# 1. Interacción agente–entorno
# 2. Frame stacking (modelado del estado temporal)
# 3. Política ε-greedy (exploración vs explotación)
# 4. Experience Replay
# 5. Actualización de la red objetivo (DQN / Double DQN)
# 6. Registro de métricas en CSV
# 7. Generación de gráficos para análisis académico:
#       - Reward total por episodio
#       - Evolución de epsilon (exploración)
#       - Tamaño del Replay Buffer
#       - Loss promedio por episodio
#
# NOTA:
# Este módulo se encarga exclusivamente del ENTRENAMIENTO.
# La EVALUACIÓN del modelo entrenado se realiza en un módulo
# separado, sin exploración (ε = 0) ni aprendizaje.
# =============================================================

import numpy as np
import logging
import os
import csv
import matplotlib.pyplot as plt

from src.entorno import crear_entorno
from src.agente import AgenteDQN
from src.preprocesamiento import (
    generar_stack_inicial,
    generar_state_frame_stack_from_queue,
    actualizar_stack
)

from src.configuracion import (
    STARTING_EPISODE, ENDING_EPISODE,
    SKIP_FRAMES, TRAINING_BATCH_SIZE,
    SAVE_TRAINING_FREQUENCY, UPDATE_TARGET_MODEL_FREQUENCY,
    GAMMA, LR
)

# =============================================================
# ESPACIO DE ACCIONES
# -------------------------------------------------------------
# CarRacing tiene un control continuo (steer, gas, brake).
# Para poder aplicar DQN (algoritmo para acciones discretas),
# discretizamos el espacio continuo en 12 acciones fijas.
#
# Esta discretización es una decisión de ingeniería necesaria
# para adaptar el problema a DQN / Double DQN.
#
# Referencia:
# https://gymnasium.farama.org/environments/box2d/car_racing/
# =============================================================
ACTION_SPACE = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
    (-1, 1, 0),   (0, 1, 0),   (1, 1, 0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0, 0),   (0, 0, 0),   (1, 0, 0)
]


# =============================================================
# FUNCIÓN PRINCIPAL: ENTRENAMIENTO COMPLETO
# =============================================================
def entrenar(
    episodio_inicio=STARTING_EPISODE,
    episodio_fin=ENDING_EPISODE,
    epsilon_inicial=1.0,
    gamma=GAMMA,
    lr=LR,
    batch_size=TRAINING_BATCH_SIZE,
    nombre_experimento="experimento"
):

    # ---------------------------------------------------------
    # CREAR CARPETAS DEL EXPERIMENTO
    # ---------------------------------------------------------
    carpeta_exp = f"resultados/{nombre_experimento}"
    carpeta_modelos = f"{carpeta_exp}/modelos"
    os.makedirs(carpeta_exp, exist_ok=True)
    os.makedirs(carpeta_modelos, exist_ok=True)

    # ---------------------------------------------------------
    # ARCHIVO CSV PARA MÉTRICAS DE ENTRENAMIENTO
    # ---------------------------------------------------------
    csv_path = f"{carpeta_exp}/metricas_entrenamiento.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["episodio", "reward_total", "epsilon", "buffer", "loss"]
        )

    logging.info(f"Guardando métricas de entrenamiento en: {csv_path}")

    # ---------------------------------------------------------
    # CREAR ENTORNO CarRacing-v3
    # ---------------------------------------------------------
    env = crear_entorno()

    # ---------------------------------------------------------
    # CREAR AGENTE + APLICAR HIPERPARÁMETROS
    # ---------------------------------------------------------
    agente = AgenteDQN(action_space=ACTION_SPACE)

    agente.epsilon = epsilon_inicial
    agente.gamma = gamma

    # Learning rate dinámico (sobrescribible desde CLI)
    for g in agente.optim.param_groups:
        g["lr"] = lr

    logging.info(
        f"Hiperparámetros → gamma={gamma} | lr={lr} | batch_size={batch_size}"
    )

    # ---------------------------------------------------------
    # CONTENEDORES PARA GRÁFICAS
    # ---------------------------------------------------------
    historial_rewards = []
    historial_epsilons = []
    historial_buffer = []
    historial_loss = []

    # =========================================================
    # BUCLE PRINCIPAL DE ENTRENAMIENTO
    # =========================================================
    for episodio in range(episodio_inicio, episodio_fin + 1):

        frame, _ = env.reset()
        stack_frames = generar_stack_inicial(frame)

        recompensa_total = 0
        contador_castigos = 0
        terminado = False
        t = 1

        losses_del_ep = []

        # -----------------------------------------------------
        # LOOP DE PASOS DENTRO DEL EPISODIO
        # -----------------------------------------------------
        while not terminado:

            # 1. Construir estado actual (stack de frames)
            estado_actual = generar_state_frame_stack_from_queue(stack_frames)

            # 2. Seleccionar acción con política ε-greedy
            accion_tupla = agente.seleccionar_accion(estado_actual)
            accion = np.array(accion_tupla, dtype=np.float32)

            # 3. Ejecutar acción con frame skipping
            recompensa = 0
            for _ in range(SKIP_FRAMES + 1):
                frame_sig, r, terminado_env, truncado, _ = env.step(accion)
                recompensa += r
                if terminado_env or truncado:
                    terminado = True
                    break

            recompensa_total += recompensa

            # 4. Heurística de castigo por mala racha
            if t > 100 and recompensa < 0:
                contador_castigos += 1
            else:
                contador_castigos = 0

            # 5. Actualizar stack y generar siguiente estado
            stack_frames = actualizar_stack(stack_frames, frame_sig)
            estado_siguiente = generar_state_frame_stack_from_queue(stack_frames)

            # 6. Guardar transición en replay buffer
            agente.memorize(
                estado_actual,
                accion_tupla,
                recompensa,
                estado_siguiente,
                terminado
            )

            # 7. Condición de fin de episodio
            if terminado or contador_castigos >= 25 or recompensa_total < 0:
                logging.info(
                    f"EP {episodio}/{episodio_fin} | "
                    f"Reward={recompensa_total:.2f} | "
                    f"ε={agente.epsilon:.3f}"
                )
                break

            # 8. Entrenamiento (si hay suficientes muestras)
            if len(agente.memoria) > batch_size:
                loss_val = agente.replay(batch_size)
                if loss_val is not None:
                    losses_del_ep.append(loss_val)

            t += 1

        # -----------------------------------------------------
        # REGISTRO DE MÉTRICAS DEL EPISODIO
        # -----------------------------------------------------
        loss_promedio = np.mean(losses_del_ep) if losses_del_ep else 0
        historial_rewards.append(recompensa_total)
        historial_epsilons.append(agente.epsilon)
        historial_buffer.append(len(agente.memoria))
        historial_loss.append(loss_promedio)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                episodio,
                recompensa_total,
                agente.epsilon,
                len(agente.memoria),
                loss_promedio
            ])

        # -----------------------------------------------------
        # ACTUALIZAR RED OBJETIVO (DQN / Double DQN)
        # -----------------------------------------------------
        # En DQN:
        #   La red objetivo estima max Q(s', a)
        #
        # En Double DQN:
        #   La red online selecciona la acción
        #   La red objetivo evalúa dicha acción
        #
        # En ambos casos, la sincronización periódica
        # estabiliza el aprendizaje.
        # -----------------------------------------------------
        if episodio % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agente.actualizar_red_objetivo()

        # -----------------------------------------------------
        # GUARDAR MODELOS INTERMEDIOS
        # -----------------------------------------------------
        if episodio % SAVE_TRAINING_FREQUENCY == 0:
            modelo_path = f"{carpeta_modelos}/modelo_ep_{episodio}.pth"
            agente.save(modelo_path)
            logging.info(f"Modelo guardado: {modelo_path}")

    # ---------------------------------------------------------
    # GUARDAR MODELO FINAL (PARA EVALUACIÓN)
    # ---------------------------------------------------------
    modelo_final_path = f"{carpeta_modelos}/modelo_final.pth"
    agente.save(modelo_final_path)
    logging.info(f"Modelo final guardado: {modelo_final_path}")

    env.close()

    # =========================================================
    # GENERACIÓN DE GRÁFICOS FINALES
    # =========================================================

    # Reward total por episodio
    plt.figure(figsize=(10, 6))
    plt.plot(historial_rewards, label="Reward total por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Reward acumulado")
    plt.title("Reward total obtenido por episodio")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{carpeta_exp}/grafico_reward.png")
    plt.close()

    # Evolución de epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(historial_epsilons, label="Epsilon (ε)")
    plt.xlabel("Episodio")
    plt.ylabel("Nivel de exploración")
    plt.title("Evolución del parámetro ε durante el entrenamiento")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{carpeta_exp}/grafico_epsilon.png")
    plt.close()

    # Tamaño del Replay Buffer
    plt.figure(figsize=(10, 6))
    plt.plot(historial_buffer, label="Tamaño del replay buffer")
    plt.xlabel("Episodio")
    plt.ylabel("Transiciones almacenadas")
    plt.title("Crecimiento del Replay Buffer")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{carpeta_exp}/grafico_buffer.png")
    plt.close()

    # Loss promedio por episodio
    plt.figure(figsize=(10, 6))
    plt.plot(historial_loss, label="Loss promedio (MSE)")
    plt.xlabel("Episodio")
    plt.ylabel("Error TD promedio")
    plt.title("Evolución de la función de pérdida")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{carpeta_exp}/grafico_loss.png")
    plt.close()

    logging.info(
        f"=== ENTRENAMIENTO FINALIZADO | "
        f"Experimento: {nombre_experimento} | "
        f"Modelo listo para evaluación ==="
    )
