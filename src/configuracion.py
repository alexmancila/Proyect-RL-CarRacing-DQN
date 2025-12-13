# =============================================================
# CONFIGURACIÓN DE RENDER
# -------------------------------------------------------------
# RENDER indica si el entorno CarRacing-v3 usa:
#
#   RENDER = True  → render_mode="human"     (ventana visible)
#   RENDER = False → render_mode="rgb_array" (entrenamiento rápido)
#
# =============================================================

RENDER = False  # modo por defecto: entrenamiento sin ventana


def set_render_mode(valor):
    """
    Modifica globalmente el modo de renderizado.

    Parámetro:
        valor: puede ser bool, int, o string ("on", "off", "1", "0").

    Conversión:
        True  → modo visible con ventana
        False → modo rápido sin ventana
    """
    global RENDER

    # Normalizar entradas tipo string
    if isinstance(valor, str):
        valor = valor.strip().lower()
        if valor in ["on", "1", "true", "sí", "si"]:
            valor = True
        else:
            valor = False

    RENDER = bool(valor)
    print(f"[CONFIG] Render activado → {RENDER}")
    return RENDER


# -------------------------------------------------------------
# GAMMA: factor de descuento
# -------------------------------------------------------------
# Controla cuánto valora el agente las recompensas futuras.
#   γ → 1: comportamiento previsor (ideal en conducción).
#   γ → 0: comportamiento inmediato, muy miope.
# -------------------------------------------------------------
GAMMA = 0.99

# -------------------------------------------------------------
# LR: learning rate del optimizador Adam
# -------------------------------------------------------------
# Controla el tamaño del paso en gradiente.
#   Muy alto  → explosión/divergencia
#   Muy bajo  → aprendizaje lento
#
# 1e-4 es estándar para imágenes + DQN.
# -------------------------------------------------------------
LR = 1e-4

# -------------------------------------------------------------
# POLÍTICA ε-greedy
# -------------------------------------------------------------
# EPSILON_INICIAL = 1.0:
#   máxima exploración al inicio (agente no sabe nada)
#
# EPSILON_MINIMO:
#   límite inferior de aleatoriedad para evitar políticas rígidas
#
# DECAY_EPSILON:
#   tasa multiplicativa de reducción de ε
#   Ejemplo:
#       ε = ε * 0.995  → exploración disminuye lentamente
# -------------------------------------------------------------
EPSILON_INICIAL = 1.0
EPSILON_MINIMO = 0.05
DECAY_EPSILON = 0.995

# -------------------------------------------------------------
# Replay Buffer: memoria de experiencias
# -------------------------------------------------------------
TAMANO_MEMORIA = 50000


# CarRacing-v3 provee imágenes RGB de 96×96.
# Nosotros apilamos STATE_STACK frames en escala de grises,
# permitiendo al agente inferir velocidad y movimiento.
# =============================================================

STATE_STACK = 4  # n° de frames consecutivos apilados
STATE_HEIGHT = 96
STATE_WIDTH = 96

# =============================================================
# PARÁMETROS DEL ENTRENAMIENTO
# -------------------------------------------------------------
# Controlan la duración del entrenamiento, la estabilidad y
# cuántos checkpoints se generan.
# =============================================================

# Episodios de entrenamiento
STARTING_EPISODE = 1
ENDING_EPISODE = 1000

# Frame Skip (reduce cómputo y suaviza acciones)
SKIP_FRAMES = 2

# Tamaño de mini-batch para Experience Replay
TRAINING_BATCH_SIZE = 64

# Frecuencia para guardar modelos
SAVE_TRAINING_FREQUENCY = 25

# Frecuencia para actualizar la red objetivo
UPDATE_TARGET_MODEL_FREQUENCY = 5
