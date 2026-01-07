# =============================================================
# Procesamiento visual para CarRacing-v3 (Gymnasium)
# -------------------------------------------------------------
# Este módulo transforma las imágenes crudas del entorno en
# representaciones compactas y útiles para la red neuronal DQN.
#
# El enfoque (grises + normalización + stack de frames) sigue el
# pipeline clásico usado en DQN con entradas visuales (estilo Atari):
#   - Mnih et al. (2013): https://arxiv.org/abs/1312.5602
#   - Mnih et al. (2015): https://doi.org/10.1038/nature14236
#
# Las operaciones realizadas buscan:
#   1) reducir la dimensionalidad de la imagen
#   2) eliminar información irrelevante (color)
#   3) normalizar la intensidad para mejorar la estabilidad numérica
#   4) capturar el movimiento mediante un "stack" temporal de frames
#
# =============================================================

import cv2
import numpy as np
import logging
from collections import deque
from src.configuracion import STATE_STACK


# =============================================================
# PROCESAR UN FRAME INDIVIDUAL
# =============================================================
def process_state_image(frame):
    """
    Recibe:
        frame → imagen RGB de Gymnasium con shape (H, W, 3)

    Devuelve:
        imagen procesada con shape (96, 96),
        escala de grises, tipo float32,
        valores normalizados en el rango [0, 1].

    Este formato es ideal para la red convolucional del agente DQN.
    """

    logging.debug("Procesando frame: conversión a grises y normalización.")

    # Protección defensiva: en casos raros el entorno puede devolver None
    if frame is None:
        logging.warning("Frame nulo recibido del entorno. Se retorna frame vacío.")
        return np.zeros((96, 96), dtype=np.float32)

    # Gymnasium entrega frames en formato RGB.
    # OpenCV trabaja internamente en BGR, por lo que se debe
    # especificar explícitamente la conversión correcta.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # El entorno CarRacing-v3 ya entrega imágenes de 96×96.
    # Se mantiene el resize explícito como medida defensiva ante
    # posibles cambios futuros en la resolución del entorno.
    resized = cv2.resize(gray, (96, 96))

    # Normalización para mejorar estabilidad numérica durante el entrenamiento
    return resized.astype(np.float32) / 255.0


# =============================================================
# CREAR STACK INICIAL DE FRAMES
# =============================================================
def generar_stack_inicial(frame):
    """
    Crea el primer estado completo del episodio.

    Procedimiento:
        - Procesa el frame inicial.
        - Duplica el frame procesado STATE_STACK veces.
        - Almacena los frames en un deque con ventana deslizante.

    Esto permite modelar correctamente un estado temporal desde
    el primer paso del episodio.
    """

    logging.debug("Generando stack inicial de frames.")

    procesado = process_state_image(frame)

    return deque(
        [procesado] * STATE_STACK,
        maxlen=STATE_STACK
    )


# =============================================================
# ACTUALIZAR STACK TEMPORAL
# =============================================================
def actualizar_stack(stack, frame):
    """
    Actualiza el stack temporal añadiendo un nuevo frame procesado.

    Parámetros:
        stack → deque existente con STATE_STACK frames
        frame → nuevo frame crudo del entorno

    Devuelve:
        stack actualizado (ventana temporal desplazada)
    """

    logging.debug("Actualizando stack con nuevo frame procesado.")

    stack.append(process_state_image(frame))
    return stack


# =============================================================
# CONVERTIR STACK A TENSOR DE ESTADO
# =============================================================
def generar_state_frame_stack_from_queue(queue):
    """
    Convierte un deque de frames en un tensor NumPy con shape:

        (STATE_STACK, 96, 96)

    donde:
        - STATE_STACK → dimensión temporal
        - 96×96       → dimensiones espaciales de la imagen

    Este tensor representa el estado final que se entrega
    al agente para la toma de decisiones.
    """

    logging.debug("Convirtiendo stack de frames a tensor NumPy.")

    return np.stack(queue, axis=0)
