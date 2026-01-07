# =============================================================
# Agente DQN / Double DQN / Dueling DQN para CarRacing-v3
# -------------------------------------------------------------
# Implementa:
#   - DQN clásico
#   - Double DQN (flag USAR_DOUBLE_DQN)
#   - Dueling DQN (flag USAR_DUELING_DQN)
#
# Todo comparte:
#   - Replay Buffer
#   - Política ε-greedy
#   - Target Network
#
# Importante:
#   - Si cambias USAR_DUELING_DQN, NO puedes cargar modelos viejos
#     entrenados con otra arquitectura (state_dict no coincide).
# =============================================================

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from src.configuracion import (
    GAMMA, LR, EPSILON_INICIAL, EPSILON_MINIMO, DECAY_EPSILON,
    TAMANO_MEMORIA, STATE_STACK, STATE_HEIGHT, STATE_WIDTH,
    USAR_DOUBLE_DQN, USAR_DUELING_DQN
)

# =============================================================
# 1. RED NEURONAL DQN (Deep Q-Network)
# =============================================================

class DQN(nn.Module):
    """
    Red convolucional que aproxima la función Q(s, a).

    Entrada:
        - Estado visual con shape (STATE_STACK, 96, 96)

    Salida:
        - Vector Q con un valor por cada acción discreta
    """

    def __init__(self, num_acciones):
        super().__init__()

        # Bloque convolucional
        self.conv = nn.Sequential(
            nn.Conv2d(STATE_STACK, 6, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 12, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Cálculo dinámico del tamaño de la capa fully-connected
        with torch.no_grad():
            dummy = torch.zeros(1, STATE_STACK, STATE_HEIGHT, STATE_WIDTH)
            flat_dim = self.conv(dummy).numel()

        # Bloque denso
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 216),
            nn.ReLU(),
            nn.Linear(216, num_acciones)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# =============================================================
# 2. RED DUELING DQN
# =============================================================

class DuelingDQN(nn.Module):
    """
    Arquitectura Dueling:
    separa el Valor del estado V(s) y la Ventaja A(s,a)

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,*)))
    """

    def __init__(self, num_acciones):
        super().__init__()

        # Mismo extractor convolucional que DQN (comparación justa)
        self.conv = nn.Sequential(
            nn.Conv2d(STATE_STACK, 6, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 12, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, STATE_STACK, STATE_HEIGHT, STATE_WIDTH)
            flat_dim = self.conv(dummy).numel()

        # Para mantener capacidad similar a tu DQN original (216)
        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 216),
            nn.ReLU(),
            nn.Linear(216, 1)
        )

        self.advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 216),
            nn.ReLU(),
            nn.Linear(216, num_acciones)
        )

    def forward(self, x):
        features = self.conv(x)
        value = self.value(features)
        advantage = self.advantage(features)

        # Q(s,a) = V(s) + (A(s,a) − mean(A))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# =============================================================
# 3. CLASE AgenteDQN
# =============================================================

class AgenteDQN:
    """
    Agente basado en:
      - DQN (por defecto)
      - Double DQN si USAR_DOUBLE_DQN = True
      - Dueling DQN si USAR_DUELING_DQN = True

    Nota:
      - Dueling y Double son compatibles: "Dueling Double DQN"
    """

    def __init__(self, action_space):

        # Espacio de acciones discretizado
        self.action_space = action_space
        self.num_acciones = len(action_space)

        # Replay Buffer
        self.memoria = deque(maxlen=TAMANO_MEMORIA)

        # Hiperparámetros
        self.gamma = GAMMA
        self.epsilon = EPSILON_INICIAL
        self.epsilon_min = EPSILON_MINIMO
        self.epsilon_decay = DECAY_EPSILON

        # -----------------------------------------------------
        # Selección de arquitectura sin romper DQN/DoubleDQN
        # -----------------------------------------------------
        ModeloQ = DuelingDQN if USAR_DUELING_DQN else DQN

        self.q_red = ModeloQ(self.num_acciones)
        self.q_red_objetivo = ModeloQ(self.num_acciones)
        self.actualizar_red_objetivo()

        # Optimizador
        self.optim = optim.Adam(self.q_red.parameters(), lr=LR)

    # =========================================================
    # POLÍTICA ε-GREEDY
    # =========================================================

    def seleccionar_accion(self, estado):
        """
        Selecciona una acción usando política ε-greedy.

        - Exploración: acción aleatoria
        - Explotación: argmax Q(s, a) usando la red online
        """

        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        estado_t = torch.tensor(
            estado, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            q_vals = self.q_red(estado_t)
            action_idx = torch.argmax(q_vals).item()

        return self.action_space[action_idx]

    # =========================================================
    # REPLAY BUFFER
    # =========================================================

    def memorize(self, state, action, reward, next_state, done):
        """
        Guarda una transición en el replay buffer.

        Se almacena el índice de la acción para optimizar memoria.
        """
        action_index = self.action_space.index(action)
        self.memoria.append(
            (state, action_index, reward, next_state, done)
        )

    # =========================================================
    # ENTRENAMIENTO (DQN / DOUBLE DQN / DUELING)
    # =========================================================

    def replay(self, batch_size):
        """
        Ejecuta un paso de entrenamiento usando Experience Replay.

        Devuelve:
            - loss (float) si se entrenó
            - None si no hay suficientes muestras
        """

        if len(self.memoria) < batch_size:
            return None

        minibatch = random.sample(self.memoria, batch_size)
        estados, acciones, recompensas, estados_sig, dones = zip(*minibatch)

        estados = torch.tensor(np.array(estados), dtype=torch.float32)
        estados_sig = torch.tensor(np.array(estados_sig), dtype=torch.float32)
        acciones = torch.tensor(acciones, dtype=torch.long)
        recompensas = torch.tensor(recompensas, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q(s,a) predicho por la red online
        q_pred = self.q_red(estados).gather(
            1, acciones.unsqueeze(1)
        ).squeeze(1)

        # -----------------------------------------------------
        # TARGET: DQN vs DOUBLE DQN (compatible con Dueling)
        # -----------------------------------------------------
        with torch.no_grad():

            if USAR_DOUBLE_DQN:
                # 1) La red online selecciona la acción
                acciones_online = self.q_red(estados_sig).argmax(1)

                # 2) La red objetivo evalúa esa acción
                q_next = self.q_red_objetivo(estados_sig).gather(
                    1, acciones_online.unsqueeze(1)
                ).squeeze(1)
            else:
                # DQN clásico:
                q_next = self.q_red_objetivo(estados_sig).max(1)[0]

            q_target = recompensas + self.gamma * q_next * (1 - dones)

        # Función de pérdida (MSE)
        loss = nn.MSELoss()(q_pred, q_target)

        # Backpropagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    # =========================================================
    # ACTUALIZAR RED OBJETIVO
    # =========================================================

    def actualizar_red_objetivo(self):
        """
        Copia los pesos de la red online a la red objetivo.
        """
        self.q_red_objetivo.load_state_dict(
            self.q_red.state_dict()
        )

    # =========================================================
    # GUARDAR / CARGAR MODELO
    # =========================================================

    def save(self, ruta):
        """
        Guarda los pesos de la red online.
        """
        torch.save(self.q_red.state_dict(), ruta)

    def load(self, ruta):
        """
        Carga los pesos del modelo y sincroniza la red objetivo.

        Ojo:
        Si el modelo fue entrenado con otra arquitectura
        (DQN vs DuelingDQN), el state_dict no va a coincidir.
        """
        self.q_red.load_state_dict(torch.load(ruta))
        self.actualizar_red_objetivo()
