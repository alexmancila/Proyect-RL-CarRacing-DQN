# Proyecto de Aprendizaje por Refuerzo — CarRacing-v3
Curso de Aprendizaje por Refuerzo de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (Lima, 2025-2)

**Integrantes (Grupo 4)**

* Koc Góngora, Luis Enrique
* Mancilla Antaya, Alex Felipe
* Meléndez García, Herbert Antonio
* Paitán Cano, Dennis Jack

Este proyecto implementa agentes de **Aprendizaje por Refuerzo** para el entorno **CarRacing-v3** (Gymnasium), usando una familia de algoritmos basados en **Deep Q-Learning** con entrada visual (frames) y *experience replay*.

Algoritmos soportados en el código:

- **DQN** (Deep Q-Network)
- **Double DQN** (reduce sobreestimación de Q)
- **Dueling DQN** (separa valor/ventaja)
- **Dueling Double DQN** (combinación)

Flujo típico:

- Entrenar y guardar métricas/modelos.
- Evaluar modelos guardados sin exploración (ε = 0) y generar GIF.
- Analizar y comparar resultados en notebooks (incluye tablas/gráficas).

Curso: Aprendizaje por Refuerzo (MIA-204)  
Grupo: 4

## Reporte (notebooks con resultados)

El reporte final (incluye comparación entre algoritmos y conclusiones) está en:

- [notebooks/entrenamiento_y_resultados_exfinal.ipynb](notebooks/entrenamiento_y_resultados_exfinal.ipynb)

El cuaderno parcial previo se mantiene como referencia:

- [notebooks/entrenamiento_y_resultados_exparcial.ipynb](notebooks/entrenamiento_y_resultados_exparcial.ipynb)

## Artículo (LaTeX)

El artículo final en LaTeX está en:

- [artículo/RL_Car_Racing_Grupo4.tex](artículo/RL_Car_Racing_Grupo4.tex)

Para compilar en Windows (PowerShell), se incluye un script que ejecuta el ciclo completo (pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX):

```
powershell -ExecutionPolicy Bypass -File tools/compilar_articulo.ps1
```

Si quieres limpiar los archivos auxiliares y recompilar desde cero:

```
powershell -ExecutionPolicy Bypass -File tools/compilar_articulo.ps1 -Clean
```

## Estructura del proyecto (actual)

```
Proyect-RL-CarRacing-DQN/
│
├── enlace_repositorio.txt
├── main.py
├── readme.md
├── requirements.txt
├── diagrama.png
│
├── docs/
│   ├── preguntas_proyecto_rl.pdf
│   └── respuestas_detalladas_proyecto_rl.pdf
│
├── notebooks/
│   ├── entrenamiento_y_resultados_exfinal.ipynb
│   └── entrenamiento_y_resultados_exparcial.ipynb
│
├── logs/
│   └── <nombre_experimento>_<timestamp>.log
│
├── resultados/
│   ├── experimento_*/
│   │   ├── metricas_entrenamiento.csv
│   │   ├── grafico_reward.png
│   │   ├── grafico_loss.png
│   │   ├── grafico_epsilon.png
│   │   ├── grafico_buffer.png
│   │   └── modelos/
│   │       ├── modelo_ep_<N>.pth
│   │       └── modelo_final.pth
│   │
│   ├── eval_*/
│   │   ├── evaluacion.log
│   │   ├── grafico_rewards.png
│   │   ├── (opcional) mejor_episodio.gif
│   │   └── metricas.csv
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── agente.py
│   ├── configuracion.py
│   ├── entorno.py
│   ├── entrenamiento.py
│   └── preprocesamiento.py
│
├── tests/
│   ├── probar_entorno.py
│   ├── probar_modelo.py
│   ├── probar_modelo_teclado.py
│   └── resumen_experimentos.py
│
└── tools/
   ├── generar_pdf_preguntas.py
   └── generar_pdf_respuestas_detalladas.py
```

## Instalación

Este proyecto está desarrollado en Python y requiere algunos paquetes y bibliotecas. Para instalar todo lo necesario, siga los pasos a continuación:

1. **Clonar el repositorio:**

   ```
   git clone https://github.com/alexmancila/Proyect-RL-CarRacing-DQN.git
   cd Proyect-RL-CarRacing-DQN
   ```
2. **Instalar dependencias:**

   Se recomienda utilizar un entorno virtual. Puedes crear uno con:

   ```
   python -m venv .venv
   # Linux/Mac:
   source .venv/bin/activate
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   ```

   Luego, instala las dependencias:

   ```
   pip install -r requirements.txt
   ```

   Los paquetes principales son:

   - `gymnasium[box2d]` (para el entorno CarRacing-v3)
   - `torch` (para el modelo DQN)
   - `torchvision` (utilidades complementarias de PyTorch)
   - `opencv-python` (para el procesamiento de imágenes)
   - `numpy` (para el manejo de matrices)
   - `matplotlib` (para la visualización de resultados)
   - `pandas` (para el análisis de métricas en el cuaderno)
   - `pygame` (dependencia de render/ventana en algunos entornos)
   - `imageio` y `imageio-ffmpeg` (para la creación de GIFs)

## Ejecución rápida (recomendado)

> Nota: el algoritmo (DQN / Double / Dueling / Dueling Double) se selecciona actualmente desde [src/configuracion.py](src/configuracion.py) mediante flags globales (ver sección “Selección de algoritmo”).

### Entrenamiento

Entrenar desde cero (por defecto llega a 1000 episodios y guarda checkpoints):

```
python main.py --nombre-exp experimento_1000ep --end 1000 --epsilon 1.0
```

Entrenar con render (más lento, pero sirve para observar):

```
python main.py --render on --nombre-exp experimento_1000ep
```

### Selección de algoritmo (DQN / Double / Dueling)

La selección se controla en [src/configuracion.py](src/configuracion.py) con estos flags:

- `USAR_DOUBLE_DQN`
- `USAR_DUELING_DQN`

Combinaciones soportadas:

| Algoritmo | `USAR_DOUBLE_DQN` | `USAR_DUELING_DQN` |
|---|---:|---:|
| DQN | `False` | `False` |
| Double DQN | `True` | `False` |
| Dueling DQN | `False` | `True` |
| Dueling Double DQN | `True` | `True` |

Recomendación práctica para mantener trazabilidad: usa nombres de experimento que incluyan el algoritmo (por ejemplo `experimento_1000ep_dqn`, `experimento_1000ep_double_dqn`, `experimento_1000ep_dueling_double_dqn`). Esto ayuda tanto en el orden de carpetas como en la evaluación (ver siguiente sección).

Nota: [main.py](main.py) añade automáticamente un sufijo estándar según los flags (`dqn`, `double_dqn`, `dueling_dqn`, `dueling_double_dqn`) y evita duplicarlo si ya lo incluiste en `--nombre-exp`. Además, si el nombre base ya contiene `dueling`, no lo repite en el sufijo.

### Evaluación

Evaluar un modelo guardado (sin exploración) y generar GIF:

```
python tests/probar_modelo.py --model "resultados/experimento_1000ep/modelos/modelo_ep_1000.pth" --episodes 5 --gif --exp eval_1000
```

Ejemplos por algoritmo (rutas coherentes con la carpeta `resultados/` del proyecto):

```
# DQN
python tests/probar_modelo.py --model "resultados/experimento_1000ep_dqn/modelos/modelo_ep_1000.pth" --episodes 5 --gif --exp eval_1000_dqn

# Double DQN
python tests/probar_modelo.py --model "resultados/experimento_1000ep_double_dqn/modelos/modelo_ep_1000.pth" --episodes 5 --gif --exp eval_1000_double_dqn

# Dueling Double DQN
python tests/probar_modelo.py --model "resultados/experimento_1000ep_dueling_double_dqn/modelos/modelo_ep_1000.pth" --episodes 5 --gif --exp eval_1000_dueling_double_dqn
```

Nota: por defecto el evaluador crea una carpeta con timestamp (por ejemplo `resultados/eval_1000_YYYYMMDD_HHMMSS/`). Si quieres una carpeta fija como `resultados/eval_1000/`, puedes renombrarla después.

Importante: [tests/probar_modelo.py](tests/probar_modelo.py) detecta automáticamente la arquitectura mirando el **nombre del path** del modelo (busca los substrings `"double"` y `"dueling"`). La ruta/carpeta del modelo debe contener esas palabras cuando corresponda para un adetección correcta.

## Ejecución (detallada)

Para entrenar el modelo, ejecuta:

```
python main.py --start 1 --end 1000 --epsilon 1.0 --nombre-exp experimento_1000ep
```

Si tienes un modelo previo y quieres continuar, puedes cargarlo desde el código (o extender `main.py` para reanudar con `--model`).

Para evaluar un modelo entrenado, ejecuta:

```
python tests/probar_modelo.py --model "ruta/del/modelo.pth" --episodes 5 --gif
```

## Descripción del Código

El código se organiza en módulos separados:

### `agente.py`

Implementa la clase `AgenteDQN` (PyTorch) con soporte para:

- DQN clásico
- Double DQN (target con acción seleccionada por la red online)
- Dueling DQN (cabezas de valor y ventaja)
- Dueling Double DQN

Nota importante: si cambias `USAR_DUELING_DQN`, no podrás cargar pesos entrenados con otra arquitectura (el `state_dict` no coincide).

### `entorno.py`

Este archivo contiene el código para inicializar el entorno `CarRacing-v3` de **Gymnasium**, y permite la interacción con el agente.

### `preprocesamiento.py`

Aquí se encuentran las funciones para procesar los frames del entorno (escala de grises, normalización y *frame stacking*). Este diseño sigue el pipeline típico usado en DQN con entradas visuales (estilo Atari) para reducir dimensionalidad y capturar dinámica temporal (Mnih et al., 2013; Mnih et al., 2015).

### `configuracion.py`

Contiene los parámetros de configuración globales (`GAMMA`, `LR`, `TRAINING_BATCH_SIZE`, etc.) y los flags que activan las variantes del algoritmo (`USAR_DOUBLE_DQN`, `USAR_DUELING_DQN`).

### `entrenamiento.py`

El loop de entrenamiento. Soporta DQN/Double/Dueling según flags. Guarda:

- `metricas_entrenamiento.csv`
- checkpoints en `resultados/<experimento>/modelos/`
- gráficos (`grafico_reward.png`, `grafico_loss.png`, etc.)

### `probar_modelo.py`

Evalúa un modelo entrenado en modo determinista (ε=0), guarda métricas, genera gráfico y (opcionalmente) un GIF del mejor episodio.

### `probar_entorno.py`

Un script básico para comprobar que el entorno CarRacing-v3 se está cargando correctamente y funciona sin errores.

## Diagrama de Arquitectura del Proyecto

![Diagrama de arquitectura del proyecto](diagrama.png)

## Resultados

Los resultados se organizan en la carpeta `resultados/`, separando **entrenamiento** y **evaluación**, lo que facilita el análisis y la comparación entre algoritmos/variantes.

### Resultados del entrenamiento

En cada carpeta `resultados/experimento_*/` se almacenan las métricas generadas durante el entrenamiento, entre ellas:

- `grafico_reward.png`: evolución de la recompensa a lo largo de los episodios de entrenamiento.
- `grafico_loss.png`: comportamiento de la función de pérdida durante el aprendizaje.
- `grafico_epsilon.png`: evolución de epsilon (exploración), que en este experimento cae rápido al inicio y luego se mantiene cerca del mínimo.
- `grafico_buffer.png`: crecimiento y estabilización del buffer de experiencias.
- `metricas_entrenamiento.csv`: métricas numéricas del entrenamiento por episodio.

Estos archivos permiten analizar cómo el agente fue aprendiendo y estabilizando su comportamiento con el paso del tiempo.

### Resultados de la evaluación

Para evaluar el desempeño del modelo entrenado, se realizaron ejecuciones controladas sin exploración (`epsilon = 0`) en distintos puntos del entrenamiento y/o para distintas variantes:

- `eval_200*`: evaluaciones del modelo entrenado con ~200 episodios.
- `eval_1000*`: evaluaciones del modelo entrenado con ~1000 episodios.

Cada carpeta de evaluación contiene:

- `metricas.csv`: resultados de reward y duración por episodio evaluado.
- `grafico_rewards.png`: recompensa obtenida en cada episodio de evaluación.
- `mejor_episodio.gif`: animación del mejor episodio registrado.
- `evaluacion.log`: registro detallado de la ejecución.

---

## Referencias (base teórica)

* Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). *Playing Atari with Deep Reinforcement Learning* (arXiv:1312.5602). *arXiv*. https://arxiv.org/abs/1312.5602
* Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*, 529–533. https://doi.org/10.1038/nature14236
* van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. In *Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16)* (pp. 2094–2100). AAAI Press. https://doi.org/10.1609/aaai.v30i1.10295
* Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. In M. F. Balcan & K. Q. Weinberger (Eds.), *Proceedings of the 33rd International Conference on Machine Learning* (Vol. 48, pp. 1995–2003). PMLR. https://proceedings.mlr.press/v48/wangf16.html
* Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. In *International Conference on Learning Representations (ICLR 2016)*. https://arxiv.org/abs/1511.05952
* Mnih, V., Puigdomenech Badia, A., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In M. F. Balcan & K. Q. Weinberger (Eds.), *Proceedings of the 33rd International Conference on Machine Learning* (Vol. 48, pp. 1928–1937). PMLR. https://proceedings.mlr.press/v48/mniha16.html
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms (arXiv:1707.06347). *arXiv*. https://arxiv.org/abs/1707.06347
* Farama Foundation. (n.d.). *Car Racing (CarRacing-v3)*. Gymnasium Documentation. Recuperado el 13 de diciembre de 2025, de https://gymnasium.farama.org/environments/box2d/car_racing/
* Paszke, A., & Towers, M. (2025, 16 de junio). *Reinforcement Learning (DQN) Tutorial*. PyTorch Tutorials. https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

### Software y librerías 

* Bradski, G. (2000). The OpenCV Library. *Dr. Dobb’s Journal of Software Tools*.
* Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., Del Río, J. F., Wiebe, M., Peterson, P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C., & Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
* Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
* McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 56–61). https://conference.scipy.org/proceedings/scipy2010/mckinney.html
* Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. Fox, & R. Garnett (Eds.), *Advances in Neural Information Processing Systems 32*. https://arxiv.org/abs/1912.01703

## Transparencia (uso de herramientas)

El desarrollo del proyecto (diseño, implementación, ejecución de experimentos y análisis) fue realizado por el Grupo 4 del curso de Aprendizaje por Refuerzo de la Maestría de Inteligencia Artificial de la Universidad Nacional de Ingeniería (Lima, semestre 2025-2). Se utilizaron herramientas de apoyo (incluyendo asistentes de IA) para mejorar redacción, documentación y consistencia del reporte.
