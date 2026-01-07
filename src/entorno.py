# =============================================================
# Creación y descripción del entorno CarRacing-v3 (Gymnasium)
# -------------------------------------------------------------
# Este módulo encapsula la construcción del entorno utilizado
# para entrenar y evaluar al agente DQN / Double DQN.
#
# Se usa CarRacing-v3, versión moderna y estable de Gymnasium.
#
# La documentación incorporada aquí permite comprender:
#   - qué características tiene CarRacing-v3
#   - qué mejoras aporta frente a v0/v1/v2
#   - por qué es un buen entorno para Aprendizaje por Refuerzo
#   - cómo se configura la visualización (render_mode)
#   - cómo se controla la aleatoriedad del entorno
# =============================================================

import gymnasium as gym
import logging
from src.configuracion import RENDER

# =============================================================
# CAR RACING: DESCRIPCIÓN ACADÉMICA DEL ENTORNO
# -------------------------------------------------------------
# CarRacing es un entorno clásico de RL visual:
#
#   • Observación: imágenes RGB de 96×96 píxeles.
#   • Acciones:    control continuo (steering, gas, brake).
#   • Recompensa:  basada en el avance limpio por la pista.
#   • Motor físico: Box2D.
#
# El agente aprende a conducir usando únicamente visión,
# sin sensores explícitos de posición, velocidad o mapa.
#
# -------------------------------------------------------------
# EVOLUCIÓN DEL ENTORNO (v0 → v3)
# -------------------------------------------------------------
# v0 (Gym clásico, obsoleto):
#   - Problemas de renderizado (especialmente en MacOS).
#   - API antigua y errores gráficos.
#
# v1 / v2:
#   - Mejoras parciales.
#   - Persistían problemas de estabilidad.
#
# v3 (Gymnasium):
#   ✔ API moderna (reset, step separados correctamente)
#   ✔ Físicas estabilizadas (Box2D actualizado)
#   ✔ Mejor rendimiento en modo "rgb_array"
#   ✔ Eliminación de pistas degeneradas
#   ✔ Estándar actual en investigación de RL visual
#
# -------------------------------------------------------------
# RENDER_MODE:
# -------------------------------------------------------------
#   - "human": ventana visible (debug / demostración).
#   - "rgb_array": sin ventana, ideal para entrenamiento y GIFs.
#
# -------------------------------------------------------------
# CONTROL DE ALEATORIEDAD:
# -------------------------------------------------------------
# domain_randomize=False asegura que:
#   - todas las evaluaciones usan la misma distribución de pistas
#   - las comparaciones DQN vs Double DQN sean justas
# =============================================================


def crear_entorno():
    """
    Crea y devuelve una instancia funcional del entorno CarRacing-v3.

    Configuración clave:
    - render_mode controlado globalmente desde configuracion.py
    - continuous=True (el entorno sigue siendo continuo)
    - domain_randomize=False (comparaciones justas y reproducibles)

    Nota:
    Aunque el entorno es continuo, el agente DQN discretiza
    externamente las acciones como decisión de ingeniería.
    """

    modo = "human" if RENDER else "rgb_array"

    logging.info(
        f"[ENTORNO] Creando CarRacing-v3 | "
        f"render_mode='{modo}' | "
        f"continuous=True | domain_randomize=False"
    )

    env = gym.make(
        "CarRacing-v3",
        render_mode=modo,
        continuous=True,
        domain_randomize=False
    )

    return env
