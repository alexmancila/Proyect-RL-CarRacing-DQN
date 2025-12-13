# =============================================================
# Creación y descripción del entorno CarRacing-v3 (Gymnasium)
# -------------------------------------------------------------
# Este módulo encapsula la construcción del entorno utilizado
# para entrenar al agente DQN. Se usa CarRacing-v3, una versión
# moderna y mejorada respecto a las versiones antiguas de Gym.
#
# La documentación incorporada aquí permite comprender:
#   - qué características tiene CarRacing-v3
#   - qué mejoras aporta frente a v0/v1/v2
#   - por qué es un buen entorno para Aprendizaje por Refuerzo
#   - cómo se configura la visualización (render_mode)
# =============================================================

import gymnasium as gym
import logging
from src.configuracion import RENDER

# =============================================================
# CAR RACING: DESCRIPCIÓN ACADÉMICA DEL ENTORNO
# -------------------------------------------------------------
# CarRacing es uno de los entornos clásicos en RL visual:
#
#   • Entrada:   imágenes RGB de 96×96 píxeles.
#   • Acciones:  control continuo (steering, gas, brake).
#   • Recompensa: basada en el avance limpio por la pista.
#   • Motor físico: Box2D.
#
# Es ideal para demostrar cómo un agente aprende a conducir
# usando únicamente visión (sin sensores explícitos de posición
# o velocidad). Esto lo hace comparable a problemas reales de
# conducción autónoma en condiciones simplificadas.
#
# -------------------------------------------------------------
# EVOLUCIÓN DEL ENTORNO (v0 → v3)
# -------------------------------------------------------------
# CAR RACING v0 (Gym clásico, ahora obsoleto):
#   - Uso de OpenGL antiguo con bugs en MacOS.
#   - Problemas de flickering (parpadeo).
#   - Frecuentes errores de renderizado.
#   - API clásica obsoleta.
#
# CAR RACING v1/v2:
#   - Mejoras menores.
#   - Persistían errores en el render.
#
# CAR RACING v3 (Gymnasium):
#   ✔ API moderna: reset() → (obs, info)
#   ✔ step() → (obs, reward, terminated, truncated, info)
#   ✔ Físicas estabilizadas con Box2D actualizado
#   ✔ Mejor rendimiento en “rgb_array”
#   ✔ Eliminación de segmentos repetidos en la pista
#   ✔ Estándar actual para investigación de RL visual
#
# -------------------------------------------------------------
# RENDER_MODE:
# -------------------------------------------------------------
#   - "human": abre ventana con visualización.
#   - "rgb_array": genera frames sin abrir ventana (ideal para entrenar).
# =============================================================


def crear_entorno():
    """
    Crea y devuelve una instancia funcional del entorno CarRacing-v3.

    Si RENDER = True → render_mode = "human"  (visualización en tiempo real)
    Si RENDER = False → render_mode = "rgb_array" (rápido, sin abrir ventana)

    Se retorna un entorno limpio, moderno y compatible con Gymnasium,
    listo para usarse en entrenamiento o evaluación.
    """

    modo = "human" if RENDER else "rgb_array"

    logging.info(f"[ENTORNO] Creando CarRacing-v3 con render_mode='{modo}'")

    return gym.make("CarRacing-v3", render_mode=modo)
