# =============================================================
# Genera una tabla resumen comparativa de múltiples experimentos
# a partir de los CSV de metricas_entrenamiento.csv
# =============================================================

import csv
import numpy as np

RUTA_RESULTADOS = "resultados"

def cargar_metricas(nombre_exp):
    """Lee el CSV de entrenamiento de un experimento y devuelve un resumen simple."""
    csv_path = f"{RUTA_RESULTADOS}/{nombre_exp}/metricas_entrenamiento.csv"

    rewards = []
    epsilons = []
    losses = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["reward_total"]))
            epsilons.append(float(row["epsilon"]))
            losses.append(float(row["loss"]))

    return {
        "episodios": len(rewards),
        "reward_final": rewards[-1],
        "reward_promedio": np.mean(rewards),
        "epsilon_final": epsilons[-1],
        "loss_promedio": np.mean(losses)
    }


def generar_tabla(experimentos):
    """Imprime una tabla comparativa (consola) a partir de varios experimentos."""
    print("\n===== RESUMEN DE EXPERIMENTOS =====\n")
    print(f"{'Experimento':20} {'Reward Final':15} {'Reward Prom':15} {'Loss Prom':15}")
    print("-" * 70)

    for exp in experimentos:
        m = cargar_metricas(exp)
        print(f"{exp:20} {m['reward_final']:15.2f} {m['reward_promedio']:15.2f} {m['loss_promedio']:15.4f}")


if __name__ == "__main__":
    experimentos = [
        "exp_gamma_099",
        "exp_gamma_095"
    ]

    generar_tabla(experimentos)
