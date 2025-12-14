# Proyecto Aprendizaje por Refuerzo en el entorno CarRacing-v3
Curso de Aprendizaje por reforzamienbto de la Maestría en Inteligenia artificial de la Universidad de Ingeniería (Lima, 2025-2)

**Integrantes Grupo 4**:

* Koc Góngora, Luis Enrique
* Mancilla Antaya, Alex Felipe
* Meléndez García, Herbert Antonio
* Paitán Cano, Dennis Jack

Este proyecto implementa un agente de **Aprendizaje por Refuerzo** utilizando el algoritmo **DQN** (Deep Q-Network) para aprender a jugar al entorno **CarRacing-v3** proporcionado por **Gymnasium** (Mnih et al., 2015; Farama Foundation, n.d.).

El flujo principal es:

- Entrenar el agente y guardar métricas/modelos.
- Evaluar modelos guardados sin exploración (ε = 0).
- Analizar resultados en el cuaderno.

Curso: Aprendizaje por Refuerzo (MIA-204)
Grupo: 4

## Reporte (cuaderno con resultados)

El reporte completo del proyecto está en el cuaderno de Jupyter, donde se muestran los resultados ya corridos (gráficas, tablas de métricas y conclusiones):

- [notebooks/entrenamiento_y_resultados.ipynb](notebooks/entrenamiento_y_resultados.ipynb)

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```
Proyect-RL-CarRacing-DQN/
│
├── .venv/ # Entorno virtual de Python
│
├── logs/
│ └── experimento_1000ep_.log # Logs del entrenamiento
│
├── notebooks/
│ └── entrenamiento_y_resultados.ipynb
│ # Análisis gráfico y estadístico del entrenamiento y evaluación
│
├── resultados/
│ ├── eval_200/
│ │ ├── evaluacion.log
│ │ ├── grafico_rewards.png
│ │ ├── mejor_episodio.gif
│ │ └── metricas.csv
│ │
│ ├── eval_1000/
│ │ ├── evaluacion.log
│ │ ├── grafico_rewards.png
│ │ ├── mejor_episodio.gif
│ │ └── metricas.csv
│ │
│ └── experimento_1000ep/
│ ├── modelos/
│ │ └── modelo_ep_.pth
│ ├── grafico_loss.png
│ ├── grafico_reward.png
│ ├── grafico_epsilon.png
│ ├── grafico_buffer.png
│ └── metricas_entrenamiento.csv
│
├── src/
│ ├── agente.py # Implementación del agente DQN (PyTorch)
│ ├── entorno.py # Creación del entorno CarRacing-v3
│ ├── preprocesamiento.py # Procesamiento de frames y stack de estados
│ └── configuracion.py # Hiperparámetros y configuración
│
├── tests/
│ ├── probar_entorno.py # Verificación del entorno
│ ├── probar_modelo.py # Evaluación automática del modelo
│ ├── probar_modelo_teclado.py # Control manual del entorno
│ └── resumen_experimentos.py # Resumen de métricas
│
├── main.py # Script principal de entrenamiento
├── requirements.txt
└── readme.md
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

### Entrenamiento

Entrenar desde cero (por defecto llega a 1000 episodios y guarda checkpoints):

```
python main.py --nombre-exp experimento_1000ep --end 1000 --epsilon 1.0
```

Entrenar con render (más lento, pero sirve para observar):

```
python main.py --render on --nombre-exp experimento_1000ep
```

### Evaluación

Evaluar un modelo guardado (sin exploración) y generar GIF:

```
python tests/probar_modelo.py --model "resultados/experimento_1000ep/modelos/modelo_ep_1000.pth" --episodes 5 --gif --exp eval_1000
```

Nota: por defecto el evaluador crea una carpeta con timestamp (por ejemplo `resultados/eval_1000_YYYYMMDD_HHMMSS/`). Si quieres una carpeta fija como `resultados/eval_1000/`, puedes renombrarla después.

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

Implementa la clase `AgenteDQN`, que utiliza una red neuronal profunda (DQN) para aprender a jugar al entorno. La red neuronal está basada en **PyTorch**.

### `entorno.py`

Este archivo contiene el código para inicializar el entorno `CarRacing-v3` de **Gymnasium**, y permite la interacción con el agente.

### `preprocesamiento.py`

Aquí se encuentran las funciones para procesar los frames del entorno (escala de grises, normalización y *frame stacking*). Este diseño sigue el pipeline típico usado en DQN con entradas visuales (estilo Atari) para reducir dimensionalidad y capturar dinámica temporal (Mnih et al., 2013; Mnih et al., 2015).

### `configuracion.py`

Contiene los parámetros de configuración globales, como la tasa de descuento `GAMMA`, el tamaño de la memoria `TAMANO_MEMORIA`, y otros hiperparámetros del agente.

### `entrenamiento.py`

El script encargado de entrenar el agente utilizando el algoritmo **DQN**. A lo largo del entrenamiento, el agente aprende a tomar decisiones a partir de su experiencia en el entorno.

### `probar_modelo.py`

Este archivo permite cargar un modelo entrenado previamente y evaluarlo en varios episodios para medir su rendimiento.

### `probar_entorno.py`

Un script básico para comprobar que el entorno CarRacing-v3 se está cargando correctamente y funciona sin errores.

## Diagrama de Arquitectura del Proyecto

![Diagrama de arquitectura del proyecto](diagrama.png)

## Resultados

Los resultados del proyecto se organizan en la carpeta `resultados`, separando claramente **entrenamiento** y **evaluación**, lo que facilita el análisis y la comparación entre distintos niveles de aprendizaje del agente.

### Resultados del entrenamiento

En la carpeta `experimento_1000ep/` se almacenan las métricas generadas durante el proceso de entrenamiento del agente DQN, entre ellas:

- `grafico_reward.png`: evolución de la recompensa a lo largo de los episodios de entrenamiento.
- `grafico_loss.png`: comportamiento de la función de pérdida durante el aprendizaje.
- `grafico_epsilon.png`: evolución de epsilon (exploración), que en este experimento cae rápido al inicio y luego se mantiene cerca del mínimo.
- `grafico_buffer.png`: crecimiento y estabilización del buffer de experiencias.
- `metricas_entrenamiento.csv`: métricas numéricas del entrenamiento por episodio.

Estos archivos permiten analizar cómo el agente fue aprendiendo y estabilizando su comportamiento con el paso del tiempo.

### Resultados de la evaluación

Para evaluar el desempeño del modelo entrenado, se realizaron ejecuciones controladas sin exploración (`epsilon = 0`) en distintos puntos del entrenamiento:

- `eval_200/`: evaluación del modelo entrenado con 200 episodios.
- `eval_1000/`: evaluación del modelo entrenado con 1000 episodios.

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
