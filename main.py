# -----------------------------------------------------------
# Archivo principal del proyecto.
#
# Funciones clave:
#   - Leer argumentos desde la línea de comandos.
#   - Activar o desactivar render del entorno.
#   - Sobrescribir hiperparámetros (gamma, lr, batch_size).
#   - Inicializar sistema de logging profesional.
#   - Lanzar el entrenamiento del agente DQN.
#
# Este archivo representa la entrada formal al sistema.
# -----------------------------------------------------------

import argparse
import logging
import os
from datetime import datetime

from src.entrenamiento import entrenar
from src.configuracion import set_render_mode, GAMMA, LR, TRAINING_BATCH_SIZE


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
        help="Nombre del experimento (logs y carpeta de resultados)."
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # RESOLVER VALORES POR DEFECTO
    # -------------------------------------------------------
    gamma = args.gamma if args.gamma is not None else GAMMA
    lr = args.lr if args.lr is not None else LR
    batch_size = args.batch_size if args.batch_size is not None else TRAINING_BATCH_SIZE

    # Validaciones básicas
    if not (0 < gamma <= 1):
        raise ValueError("gamma debe estar en el rango (0, 1]")

    if lr <= 0:
        raise ValueError("learning rate debe ser positivo")

    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero")

    # -------------------------------------------------------
    # APLICAR CONFIGURACIONES
    # -------------------------------------------------------
    set_render_mode(args.render == "on")
    configurar_logging(args.nombre_exp)

    logging.info("=== Iniciando entrenamiento de CarRacing-v3 (DQN) ===")
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
        nombre_experimento=args.nombre_exp
    )
