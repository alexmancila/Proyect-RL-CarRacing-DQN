# =============================================================
# Agente DQN para CarRacing-v3 (Gymnasium)
# Versión completamente documentada para fines educativos.
# =============================================================

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from src.configuracion import (
    GAMMA, LR, EPSILON_INICIAL, EPSILON_MINIMO, DECAY_EPSILON,
    TAMANO_MEMORIA, STATE_STACK, STATE_HEIGHT, STATE_WIDTH
)

# =============================================================
# 1. RED NEURONAL DQN (Deep Q-Network)
# -------------------------------------------------------------
# Esta red neuronal es el “cerebro” del agente:
# recibe un estado visual (frames del juego) y devuelve 12 valores,
# uno por cada acción posible. Cada valor representa Q(s,a):
# la estimación de tan buena es cada acción en ese estado.
# =============================================================

class DQN(nn.Module):
    def __init__(self, num_acciones):
        """
        num_acciones: cantidad de acciones disponibles (12 en CarRacing)

        Arquitectura:
        - 2 convoluciones
        - 2 MaxPool
        - Flatten
        - Dense(216)
        - Dense(num_acciones)
        """
        super().__init__()

        # ---------------------------------------------------------
        # Bloque convolucional
        # ---------------------------------------------------------
        # Conv2D recibe: (batch, canales, alto, ancho)
        # STATE_STACK = número de frames apilados (4).
        #
        # Primer Conv:
        # - 6 filtros
        # - kernel 7x7
        # - stride 3 (bloque grueso para reducir tamaño rápido)
        #
        # MaxPool reduce la resolución a la mitad.
        # ---------------------------------------------------------

        self.conv = nn.Sequential(
            nn.Conv2d(STATE_STACK, 6, kernel_size=7, stride=3),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(6, 12, kernel_size=4),
            nn.ReLU(),

            nn.MaxPool2d(2),
        )

        # ---------------------------------------------------------
        # Cálculo automático del tamaño de la capa densa
        # ---------------------------------------------------------
        # Se pasa un tensor "dummy" para ver cuál es el tamaño final
        # después de todas las convoluciones y MaxPool.
        # Esto evita errores de dimensiones.
        # ---------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, STATE_STACK, STATE_HEIGHT, STATE_WIDTH)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.numel()

        # ---------------------------------------------------------
        # Bloque denso (fully-connected)
        # ---------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 216),
            nn.ReLU(),
            nn.Linear(216, num_acciones)  # salida = 12 valores Q
        )

    def forward(self, x):
        """
        Propagación hacia adelante:
        estado → convoluciones → densas → valores Q.
        """
        return self.fc(self.conv(x))


# =============================================================
# 2. CLASE AgenteDQN
# -------------------------------------------------------------
# Implementa:
# - Politica ε-greedy
# - Replay buffer
# - Actualización de redes (online y target)
# - Cálculo de la ecuación de Bellman
#
# Es el núcleo del aprendizaje por refuerzo.
# =============================================================

class AgenteDQN:

    def __init__(self, action_space):
        """
        action_space:
            Lista de tuplas (steer, gas, brake)
            Ejemplo: (-1, 1, 0.2)
        """

        # ---------------------------------------------------------
        # Definir espacio de acciones
        # ---------------------------------------------------------
        self.action_space = action_space
        self.num_acciones = len(action_space)

        # ---------------------------------------------------------
        # Replay Buffer (Memoria de experiencias)
        # ---------------------------------------------------------
        self.memoria = deque(maxlen=TAMANO_MEMORIA)

        # ---------------------------------------------------------
        # Hiperparámetros del algoritmo DQN
        # ---------------------------------------------------------
        self.gamma = GAMMA                     # descuento futuro
        self.epsilon = EPSILON_INICIAL         # probabilidad de explorar
        self.epsilon_min = EPSILON_MINIMO      # mínimo ε
        self.epsilon_decay = DECAY_EPSILON     # factor de decaimiento

        # ---------------------------------------------------------
        # Creación de redes neuronal:
        # - q_red: red principal que aprende
        # - q_red_objetivo: copia fija actualizada cada N episodios
        # Esto evita oscilaciones inestables.
        # ---------------------------------------------------------
        self.q_red = DQN(self.num_acciones)
        self.q_red_objetivo = DQN(self.num_acciones)
        self.actualizar_red_objetivo()

        # Optimizador Adam
        self.optim = optim.Adam(self.q_red.parameters(), lr=LR)

    # =========================================================
    # SELECCIÓN DE ACCIÓN: política ε-greedy
    # ---------------------------------------------------------
    # Si random < ε → exploración (acción aleatoria)
    # Si random ≥ ε → explotación (acción que maximiza Q)
    # =========================================================

    def seleccionar_accion(self, estado):
        """
        Recibe: estado con shape (STATE_STACK, 96, 96)
        Devuelve: una de las 12 acciones del action_space.
        """

        # Exploración aleatoria
        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        # Explotación: elegir la mejor acción según la red
        estado_t = torch.tensor(estado, dtype=torch.float32).unsqueeze(0)
        q_vals = self.q_red(estado_t)          # obtener valores Q
        action_idx = torch.argmax(q_vals).item()

        return self.action_space[action_idx]

    # =========================================================
    # REPLAY BUFFER: guardar transiciones
    # =========================================================

    def memorize(self, state, action, reward, next_state, done):
        """
        Guardamos el índice de la acción (no la tupla completa) para que la
        transición quede compacta y sea fácil de almacenar/muestrear.
        """
        action_index = self.action_space.index(action)
        self.memoria.append((state, action_index, reward, next_state, done))

    # =========================================================
    # ENTRENAMIENTO (Mini-batch Replay)
    # ---------------------------------------------------------
    # Implementa la ecuación de Bellman:
    #
    # Q(s,a) ← r + γ * max(Q(s', a'))
    #
    # Ahora:
    #   - actualiza la red
    #   - aplica epsilon decay
    #   - DEVUELVE el valor del loss (float) para logging
    # =========================================================

    def replay(self, batch_size):
        """
        Ejecuta un paso de entrenamiento DQN sobre un mini-batch.

        Devuelve:
            - loss.item() (float) si se entrenó
            - None si no había suficientes muestras en memoria
        """

        # Si no hay suficientes muestras, no entrenamos
        if len(self.memoria) < batch_size:
            return None

        # Selección aleatoria del batch
        minibatch = random.sample(self.memoria, batch_size)

        estados, acciones, recompensas, estados_sig, dones = zip(*minibatch)


        estados = torch.tensor(np.array(estados), dtype=torch.float32)
        estados_sig = torch.tensor(np.array(estados_sig), dtype=torch.float32)
        acciones = torch.tensor(acciones)
        recompensas = torch.tensor(recompensas, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # ---------------------------------------------------------
        # Q_pred = Q(s,a) usando q_red
        # gather permite seleccionar la columna correspondiente
        # a cada acción del batch.
        # ---------------------------------------------------------
        pred = self.q_red(estados)
        q_pred = pred.gather(1, acciones.unsqueeze(1)).squeeze(1)

        # ---------------------------------------------------------
        # Q_target = r + γ * max(Q(s',a')) usando red objetivo
        # Si done = True, no hay recompensa futura.
        # ---------------------------------------------------------
        with torch.no_grad():
            q_next = self.q_red_objetivo(estados_sig).max(1)[0]
            q_target = recompensas + self.gamma * q_next * (1 - dones)

        # Cálculo de pérdida (MSE)
        loss = nn.MSELoss()(q_pred, q_target)

        # Backpropagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # ---------------------------------------------------------
        # Decaimiento del epsilon (menos exploración en el tiempo)
        # ---------------------------------------------------------
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Devuelve el valor escalar del loss para logging y gráficos
        return float(loss.item())

    # =========================================================
    # ACTUALIZAR RED OBJETIVO
    # ---------------------------------------------------------
    # Se copia la red principal a la red objetivo.
    # Esto estabiliza el entrenamiento.
    # =========================================================

    def actualizar_red_objetivo(self):
        self.q_red_objetivo.load_state_dict(self.q_red.state_dict())

    # =========================================================
    # GUARDAR / CARGAR MODELO
    # =========================================================

    def save(self, ruta):
        """
        Guarda únicamente los pesos (state_dict) de q_red.
        """
        torch.save(self.q_red.state_dict(), ruta)

    def load(self, ruta):
        """
        Carga el modelo y sincroniza la red objetivo.
        """
        self.q_red.load_state_dict(torch.load(ruta))
        self.actualizar_red_objetivo()
