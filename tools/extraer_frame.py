"""
Script para extraer un frame del GIF de CarRacing-v3 y guardarlo como imagen estática.
"""
import imageio.v2 as imageio
import os

# Obtener la ruta base del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Ruta del GIF de entrada (usaremos el mejor episodio de Double DQN a 1000 episodios)
gif_path = os.path.join(project_root, "resultados", "eval_1000_double_dqn", "mejor_episodio.gif")

# Ruta de salida
output_path = os.path.join(project_root, "artículo", "figuras", "carracing_entorno.png")

# Verificar que el directorio de salida existe
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Leer el GIF
reader = imageio.get_reader(gif_path)

# Extraer un frame intermedio (frame 30, que muestra el coche en acción en la pista)
frame_number = 30
frame = reader.get_data(frame_number)

# Guardar el frame como PNG
imageio.imwrite(output_path, frame)

print(f"✓ Frame extraído exitosamente")
print(f"✓ Guardado en: {output_path}")
