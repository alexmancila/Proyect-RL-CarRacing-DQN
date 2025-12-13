# =============================================================
# MÓDULO DE ENTRENAMIENTO DEL AGENTE DQN PARA CarRacing-v3
# -------------------------------------------------------------
# Este archivo implementa el ciclo de Aprendizaje por Refuerzo:
#
# 1. Interacción agente–entorno
# 2. Frame stacking (modelado del estado temporal)
# 3. Política ε-greedy (exploración vs explotación)
# 4. Experience Replay
# 5. Actualización de la red objetivo (DQN clásico)
# 6. Registro de métricas en CSV
# 7. Generación de 4 gráficos para análisis académico:
#       - Reward total por episodio
#       - Evolución de epsilon (exploración)
#       - Tamaño del Replay Buffer
#       - Loss promedio por episodio
#
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
# Esto es EXACTAMENTE igual al repositorio original.
# =============================================================
ACTION_SPACE = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
    (-1, 1, 0), (0, 1, 0), (1, 1, 0),
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
    (-1, 0, 0), (0, 0, 0), (1, 0, 0)
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

    # CSV con métricas
    csv_path = f"{carpeta_exp}/metricas_entrenamiento.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episodio", "reward_total", "epsilon", "buffer", "loss"])

    logging.info(f"Guardando métricas en: {csv_path}")

    # ---------------------------------------------------------
    # CREAR ENTORNO CarRacing-v3
    # ---------------------------------------------------------
    env = crear_entorno()

    # ---------------------------------------------------------
    # CREAR AGENTE + APLICAR HIPERPARÁMETROS
    # ---------------------------------------------------------
    agente = AgenteDQN(action_space=ACTION_SPACE)

    agente.epsilon = epsilon_inicial      # Exploración inicial
    agente.gamma = gamma                  # Factor de descuento

    # Learning rate dinámico (modificable desde CLI)
    for g in agente.optim.param_groups:
        g["lr"] = lr

    logging.info(f"Hiperparámetros → gamma={gamma} | lr={lr} | batch={batch_size}")

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

        # Reset del entorno y creación del stack temporal
        frame, _ = env.reset()
        stack_frames = generar_stack_inicial(frame)

        recompensa_total = 0
        contador_castigos = 0
        terminado = False
        t = 1

        losses_del_ep = []  # variación de la función de pérdida

        # -----------------------------------------------------
        # LOOP DE PASOS DENTRO DEL EPISODIO
        # -----------------------------------------------------
        while not terminado:

            # 1. Construir estado actual (stack 4 frames)
            estado_actual = generar_state_frame_stack_from_queue(stack_frames)

            # 2. Seleccionar acción según política epsilon-greedy
            accion_tupla = agente.seleccionar_accion(estado_actual)
            accion = np.array(accion_tupla, dtype=np.float32)

            # 3. Ejecutar frame skipping (acción se repite varios frames)
            recompensa = 0
            for _ in range(SKIP_FRAMES + 1):
                frame_sig, r, terminado_env, truncado, info = env.step(accion)
                recompensa += r
                if terminado_env or truncado:
                    terminado = True
                    break

            recompensa_total += recompensa

            # 4. Penalización por malas rachas (del repositorio original)
            if t > 100 and recompensa < 0:
                contador_castigos += 1
            else:
                contador_castigos = 0

            # 5. Actualizar stack temporal y generar siguiente estado
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

            # 7. Condiciones de fin de episodio
            if terminado or contador_castigos >= 25 or recompensa_total < 0:
                logging.info(
                    f"EP {episodio}/{episodio_fin} | "
                    f"Reward={recompensa_total:.2f} | "
                    f"ε={agente.epsilon:.3f}"
                )
                break

            # 8. Entrenamiento DQN (solo si hay suficientes muestras)
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
        # COPIAR PESOS A LA RED OBJETIVO (DQN clásico)
        # -----------------------------------------------------
        if episodio % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agente.actualizar_red_objetivo()

        # -----------------------------------------------------
        # GUARDAR MODELO
        # -----------------------------------------------------
        if episodio % SAVE_TRAINING_FREQUENCY == 0:
            modelo_path = f"{carpeta_modelos}/modelo_ep_{episodio}.pth"
            agente.save(modelo_path)
            logging.info(f"Modelo guardado: {modelo_path}")

    env.close()


    # GENERACION DE GRAFICOS FINALES
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


    # Evolución de epsilon (exploracion)

    plt.figure(figsize=(10, 6))
    plt.plot(historial_epsilons, label="Epsilon (ε)")
    plt.xlabel("Episodio")
    plt.ylabel("Valor de ε (nivel de exploración)")
    plt.title("Evolución del parámetro ε durante el entrenamiento")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{carpeta_exp}/grafico_epsilon.png")
    plt.close()

    # Tamaño del Replay Buffer...
    plt.figure(figsize=(10, 6))
    plt.plot(historial_buffer, label="Tamaño del replay buffer")
    plt.xlabel("Episodio")
    plt.ylabel("Número de transiciones almacenadas")
    plt.title("Crecimiento del Replay Buffer a lo largo del entrenamiento")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{carpeta_exp}/grafico_buffer.png")
    plt.close()

    # Loss promedio por episodio.
    plt.figure(figsize=(10, 6))
    plt.plot(historial_loss, label="Loss promedio (MSE)")
    plt.xlabel("Episodio")
    plt.ylabel("Error temporal-diferencial promedio")
    plt.title("Evolución de la función de pérdida (DQN)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{carpeta_exp}/grafico_loss.png")
    plt.close()

    logging.info(
        f"=== ENTRENAMIENTO FINALIZADO | Experimento: {nombre_experimento} ==="
    )
