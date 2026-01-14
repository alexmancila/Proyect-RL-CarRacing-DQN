# -----------------------------------------------------------
# Archivo principal del proyecto.
#
# Funciones clave:
#   - Leer argumentos desde la línea de comandos.
#   - Activar o desactivar render del entorno.
#   - Sobrescribir hiperparámetros (gamma, lr, batch_size).
#   - Inicializar sistema de logging profesional.
#   - Fijar semillas aleatorias para reproducibilidad.
#   - Registrar explícitamente el algoritmo activo (DQN / Double DQN).
#   - Lanzar el entrenamiento del agente.
#
# Este archivo representa la entrada formal y reproducible
# al sistema de Aprendizaje por Refuerzo.
# -----------------------------------------------------------

import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

from src.entrenamiento import entrenar
from src.configuracion import (
    set_render_mode,
    GAMMA,
    LR,
    TRAINING_BATCH_SIZE,
    USAR_DOUBLE_DQN,
    USAR_DUELING_DQN
)

# -----------------------------------------------------------
# SISTEMA DE LOGGING
# -----------------------------------------------------------
def configurar_logging(nombre_experimento: str):
    """
    Crea un archivo de log en la carpeta /logs con nombre:
        {nombre_experimento}_{timestamp}.log

    Permite:
        • registrar progreso del entrenamiento
        • guardar métricas y estados
        • diferenciar experimentos (DQN vs Double DQN)
        • justificar decisiones en el informe académico
    """
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_log = f"logs/{nombre_experimento}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ruta_log, encoding="utf-8")
        ]
    )

    logging.info(f"Logging iniciado — Archivo: {ruta_log}")


# -----------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------
    # SEMILLA ALEATORIA (REPRODUCIBILIDAD)
    # -------------------------------------------------------
    # Se fijan semillas para garantizar que las comparaciones
    # entre DQN y Double DQN sean justas y reproducibles.
    # -------------------------------------------------------
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # -------------------------------------------------------
    # PARSER DE ARGUMENTOS
    # -------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Entrena el agente DQN para CarRacing-v3 usando PyTorch."
    )

    # -------------------------------------------------------
    # ARGUMENTOS BÁSICOS DE ENTRENAMIENTO
    # -------------------------------------------------------
    parser.add_argument(
        "-m", "--model",
        help="Ruta del modelo .pth previamente entrenado para continuar entrenamiento."
    )

    parser.add_argument(
        "-s", "--start",
        type=int,
        help="Episodio inicial del entrenamiento (default = 1)."
    )

    parser.add_argument(
        "-e", "--end",
        type=int,
        help="Episodio final del entrenamiento (default = 1000)."
    )

    parser.add_argument(
        "-p", "--epsilon",
        type=float,
        default=1.0,
        help="Valor inicial de epsilon para la política ε-greedy."
    )

    # -------------------------------------------------------
    # CONTROL DE RENDERIZADO
    # -------------------------------------------------------
    parser.add_argument(
        "--render",
        choices=["on", "off"],
        default="off",
        help="Activa ('on') o desactiva ('off') la ventana gráfica del entorno."
    )

    # -------------------------------------------------------
    # HIPERPARÁMETROS EDITABLES DESDE CONSOLA
    # -------------------------------------------------------
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=f"Factor de descuento gamma (default={GAMMA})."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"Learning rate del optimizador Adam (default={LR})."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Tamaño del minibatch para replay (default={TRAINING_BATCH_SIZE})."
    )

    # -------------------------------------------------------
    # META: Información del experimento
    # -------------------------------------------------------
    parser.add_argument(
        "--nombre-exp",
        type=str,
        default="experimento_carracing",
        help="Nombre base del experimento (logs y resultados)."
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # RESOLVER VALORES POR DEFECTO
    # -------------------------------------------------------
    gamma = args.gamma if args.gamma is not None else GAMMA
    lr = args.lr if args.lr is not None else LR
    batch_size = (
        args.batch_size if args.batch_size is not None else TRAINING_BATCH_SIZE
    )

    # Validaciones básicas de seguridad
    if not (0 < gamma <= 1):
        raise ValueError("gamma debe estar en el rango (0, 1]")

    if lr <= 0:
        raise ValueError("learning rate debe ser positivo")

    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero")

    # -------------------------------------------------------
    # APLICAR CONFIGURACIONES GLOBALES
    # -------------------------------------------------------
    set_render_mode(args.render == "on")

    # -------------------------------------------------------
    # Nombre de experimento (retrocompatible)
    # -------------------------------------------------------
    # Se añade un sufijo de algoritmo para diferenciar resultados.
    # Importante: NO debe romper los resultados ya corridos.
    #
    # Reglas:
    # - Si el usuario ya termina con un sufijo estándar, no se modifica.
    # - Si el usuario ya incluyó "dueling" en el nombre base, no lo repetimos.
    #
    # Sufijos estándar:
    #   dqn | double_dqn | dueling_dqn | dueling_double_dqn
    base = args.nombre_exp
    base_lower = base.lower()

    sufijos_estandar = [
        "dqn",
        "double_dqn",
        "dueling_dqn",
        "dueling_double_dqn",
    ]

    if any(base_lower.endswith(s) for s in sufijos_estandar):
        nombre_experimento = base
    else:
        partes = []
        if USAR_DUELING_DQN and ("dueling" not in base_lower):
            partes.append("dueling")
        if USAR_DOUBLE_DQN:
            partes.append("double")
        partes.append("dqn")

        sufijo_algoritmo = "_".join(partes)
        nombre_experimento = f"{base}_{sufijo_algoritmo}"

    configurar_logging(nombre_experimento)

    logging.info(f"Semilla fija para el experimento: {SEED}")
    logging.info(
        f"Algoritmo activo: {'Double DQN' if USAR_DOUBLE_DQN else 'DQN clasico'}"
    )

    logging.info(
        f"Hiperparámetros iniciales | "
        f"gamma={gamma} | lr={lr} | batch_size={batch_size} | epsilon={args.epsilon}"
    )

    # -------------------------------------------------------
    # EJECUTAR ENTRENAMIENTO
    # -------------------------------------------------------
    entrenar(
        episodio_inicio=args.start or 1,
        episodio_fin=args.end or 1000,
        epsilon_inicial=args.epsilon,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        nombre_experimento=nombre_experimento
    )
