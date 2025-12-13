"""
=====================================================================
CONTROL MANUAL DEL ENTORNO CarRacing-v3 MEDIANTE TECLADO
Descripción:
    Este script permite jugar CarRacing-v3 manualmente usando el teclado.
    El propósito NO es entrenar ni evaluar un modelo de RL, sino:

    - Comprender el espacio continuo de acciones (steer, gas, brake)
    - Observar cómo responde la física del entorno
    - Comparar comportamiento humano vs comportamiento del agente
    - Validar la interfaz de entrada y el renderizado del entorno

    Esta herramienta es útil durante la fase de modelado del problema
    y análisis del entorno, lo cual forma parte de la rúbrica del curso.
=====================================================================
"""

import gymnasium as gym

# Estados de las teclas en tiempo real
tecla_izquierda  = False
tecla_derecha    = False
tecla_acelerar   = False
tecla_frenar     = False
tecla_salir      = False

# Valores continuos de las acciones (igual al agente DQN)
direccion      = 0.0   # rango: [-1, 1]
acelerador     = 0.0   # rango: [0, 1]
freno          = 0.0   # rango: [0, 1]


# ================================================================
# FUNCIONES DE MANEJO DEL TECLADO
# Estas funciones son llamadas automáticamente por la ventana
# del entorno CarRacing-v3.
# ================================================================

def key_press(key, mod):
    """
    Evento al presionar una tecla.
    """
    global tecla_izquierda, tecla_derecha, tecla_acelerar, tecla_frenar, tecla_salir

    if key == 65361:   # Flecha izquierda
        tecla_izquierda = True
    if key == 65363:   # Flecha derecha
        tecla_derecha = True
    if key == 32:      # Espacio = acelerar
        tecla_acelerar = True
    if key == 65505:   # Shift = frenar
        tecla_frenar = True
    if key == 65307:   # ESC = salir
        tecla_salir = True


def key_release(key, mod):
    """
    Evento al soltar una tecla.
    """
    global tecla_izquierda, tecla_derecha, tecla_acelerar, tecla_frenar

    if key == 65361:
        tecla_izquierda = False
    if key == 65363:
        tecla_derecha = False
    if key == 32:
        tecla_acelerar = False
    if key == 65505:
        tecla_frenar = False


# ================================================================
# ACTUALIZACIÓN CONTINUA DE LAS ACCIONES
# Convierte el estado de teclas en un vector de acción válido:
#   [steering (-1..1), gas (0..1), brake (0..1)]
# ================================================================

def actualizar_accion():
    """
    Produce un vector de acción continuo en base a teclas presionadas.
    Imita exactamente la dinámica usada en el agente RL.
    """
    global direccion, acelerador, freno

    # --------------------------
    # DIRECCIÓN (steering wheel)
    # --------------------------
    if tecla_izquierda ^ tecla_derecha:   # XOR: solo una tecla activa
        if tecla_izquierda:
            direccion = max(-1.0, direccion - 0.1)
        else:
            direccion = min( 1.0, direccion + 0.1)
    else:
        # Retorno suave al centro
        if abs(direccion) < 0.1:
            direccion = 0.0
        elif direccion > 0:
            direccion -= 0.1
        else:
            direccion += 0.1

    # --------------------------
    # ACELERADOR (gas)
    # --------------------------
    if tecla_acelerar:
        acelerador = min(1.0, acelerador + 0.1)
    else:
        acelerador = max(0.0, acelerador - 0.1)

    # --------------------------
    # FRENO (brake)
    # --------------------------
    if tecla_frenar:
        freno = min(1.0, freno + 0.1)
    else:
        freno = max(0.0, freno - 0.1)


# ================================================================
# BLOQUE PRINCIPAL
# ================================================================

def jugar_manual():
    """
    Permite controlar CarRacing-v3 manualmente.
    Se ejecuta hasta que el usuario presione ESC.
    """

    env = gym.make("CarRacing-v3", render_mode="human")

    # Registrar eventos de teclado en la ventana del entorno
    viewer = env.unwrapped.viewer
    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release

    frame, info = env.reset()

    pasos = 0
    recompensa_total = 0

    print("\n=== CONTROL MANUAL ACTIVADO ===")
    print("Usa:")
    print("  ←  →  para girar")
    print("  SPACE para acelerar")
    print("  SHIFT para frenar")
    print("  ESC   para salir\n")

    while not tecla_salir:
        env.render()

        # Actualizar acción continua desde teclado
        actualizar_accion()
        action = [direccion, acelerador, freno]

        # Ejecutar acción en el entorno
        frame, reward, terminated, truncated, info = env.step(action)

        pasos += 1
        recompensa_total += reward

        print(f"Paso {pasos:4d} | Acción: {action} | Reward: {reward:.3f}")

        if terminated or truncated:
            print(f"Reinicio del episodio | Reward total: {recompensa_total:.2f}")
            frame, info = env.reset()
            pasos = 0
            recompensa_total = 0

    env.close()
    print("\nJuego finalizado por el usuario.")


if __name__ == "__main__":
    jugar_manual()
