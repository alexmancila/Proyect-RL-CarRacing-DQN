"""
==============================================================
PRUEBA BÁSICA DEL ENTORNO CarRacing-v3 (GYMNASIUM)
Descripción:
    Este script NO entrena ni evalúa ningún modelo.
    Su único objetivo es verificar que el entorno CarRacing-v3:
==============================================================
"""

import gymnasium as gym
import time


def probar_carracing():
    """
    Inicializa el entorno CarRacing-v3 y ejecuta 200 pasos con
    acciones aleatorias tomadas directamente del espacio de acciones.

    Esto permite validar:
    - correcto funcionamiento del renderizado
    - correcta devolución de (obs, reward, terminated, truncated, info)
    - estabilidad del entorno

    No tiene relación con el agente DQN.
    """

    # Crear el entorno con renderizado humano (ventana gráfica).
    env = gym.make("CarRacing-v3", render_mode="human")

    # reset() devuelve observación inicial + información adicional.
    obs, info = env.reset()

    total_reward = 0.0

    # Ejecutar 200 pasos aleatorios para verificar estabilidad.
    for paso in range(200):

        # Acción aleatoria: NO proviene de ningún modelo.
        accion = env.action_space.sample()

        # Ejecutar un paso en el entorno.
        obs, reward, terminated, truncated, info = env.step(accion)

        # Acumular recompensa solo como referencia.
        total_reward += reward

        # Pequeña pausa para que el render no vaya demasiado rápido.
        time.sleep(1 / 30)

        # Si el episodio termina antes de los 200 pasos, salir.
        if terminated or truncated:
            print("Episodio terminó en el paso", paso)
            break

    # Cerrar ventana del entorno.
    env.close()

    print("Recompensa total del episodio:", total_reward)


# Ejecución directa del script.
if __name__ == "__main__":
    probar_carracing()
