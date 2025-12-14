# =============================================================
# Procesamiento visual para CarRacing-v3 (Gymnasium)
# -------------------------------------------------------------
# Este módulo transforma las imágenes crudas del entorno en
# representaciones compactas y útiles para la red neuronal DQN.
#
# Las operaciones realizadas buscan:
#   1) reducir la dimensionalidad de la imagen
#   2) eliminar información irrelevante (color)
#   3) normalizar la intensidad para mejorar la estabilidad
#   4) capturar el movimiento mediante un "stack" de frames
#
# =============================================================

import cv2
import numpy as np
import logging
from collections import deque
from src.configuracion import STATE_STACK



# Procesar un frame individual
def process_state_image(frame):
    """
    Recibe:
        frame → imagen RGB de Gymnasium con shape (H, W, 3)

    Devuelve:
        imagen procesada con shape (96, 96) en escala de grises,
        tipo float32, valores normalizados entre [0, 1].

    Este formato es ideal para la red convolucional del agente.
    """

    logging.debug("Procesando frame: grises, resize y normalización.")

    # Gymnasium entrega frames en RGB. OpenCV trabaja en BGR por defecto.
    # Aquí convertimos a grises (el objetivo es eliminar color y simplificar).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensionar a 96×96 (tamaño de entrada del modelo)
    resized = cv2.resize(gray, (96, 96))

    # Normalización para mejorar la estabilidad numérica
    return resized.astype(np.float32) / 255.0


# Crear el stack inicial de frames


def generar_stack_inicial(frame):
    """
    Crea el primer estado completo del episodio.

    - Procesa el frame inicial.
    - Crea un deque con STATE_STACK copias de ese frame.
    - El deque permite desplazar el stack como una ventana deslizante.

    Esto modela correctamente una observación temporal.
    """

    logging.debug("Generando stack inicial de frames...")

    procesado = process_state_image(frame)


    return deque([procesado] * STATE_STACK, maxlen=STATE_STACK)



# Actualizar el stack con un nuevo frame


def actualizar_stack(stack, frame):
    """
    Desplaza la ventana temporal añadiendo el nuevo frame.

    stack → deque existente con STATE_STACK imágenes
    frame → nuevo frame del entorno

    Devuelve:
        stack actualizado
    """
    logging.debug("Actualizando stack con un nuevo frame procesado.")

    stack.append(process_state_image(frame))
    return stack



# Convertir el stack en un tensor (stack, H, W)


def generar_state_frame_stack_from_queue(queue):
    """
    Convierte un deque de frames en un tensor NumPy con shape:

        (STATE_STACK, 96, 96)

    donde:
        - STATE_STACK   → dimensión temporal
        - 96, 96        → dimensiones espaciales de la imagen

    Este tensor es la representación final del "estado" que
    el agente recibe para decidir acciones.
    """

    logging.debug("Convirtiendo deque a tensor de estado (stack,96,96).")

    return np.stack(queue, axis=0)
