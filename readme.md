
# Proyecto Aprendizaje por Refuerzo en el entorno CarRacing-v3

Este proyecto implementa un agente de **Aprendizaje por Refuerzo** utilizando el algoritmo **DQN** (Deep Q-Network) para aprender a jugar al entorno **CarRacing-v3** proporcionado por **Gymnasium**.

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
└── README.md
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
    python3 -m venv .venv
    source .venv/bin/activate  # En Linux/Mac
    .venv\Scriptsctivate  # En Windows
    ```

    Luego, instala las dependencias:

    ```
    pip install -r requirements.txt
    ```

    Los paquetes principales son:
    - `gymnasium` (para el entorno CarRacing-v3)
    - `torch` (para el modelo DQN)
    - `opencv-python` (para el procesamiento de imágenes)
    - `numpy` (para el manejo de matrices)
    - `matplotlib` (para la visualización de resultados)
    - `imageio` (para la creación de GIFs)

## Ejecución

Para entrenar el modelo, ejecute:

```
python src/main.py --model "ruta/del/modelo.pth" --start 1 --end 1000 --epsilon 1.0
```

Si no tienes un modelo previo, el sistema entrenará desde el inicio. Los modelos entrenados se guardarán en la carpeta `modelos`.

Para probar el modelo entrenado, ejecuta:

```
python src/probar_modelo.py --model "ruta/del/modelo.pth" --episodes 3 --gif
```

## Descripción del Código

El código se organiza en módulos separados:

### `agente.py`
Implementa la clase `AgenteDQN`, que utiliza una red neuronal profunda (DQN) para aprender a jugar al entorno. La red neuronal está basada en **PyTorch**.

### `entorno.py`
Este archivo contiene el código para inicializar el entorno `CarRacing-v3` de **Gymnasium**, y permite la interacción con el agente.

### `preprocesamiento.py`
Aquí se encuentran las funciones para procesar los frames del entorno, convirtiéndolos a escala de grises y reduciendo su tamaño para que el agente pueda aprender de ellos.

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
- `grafico_epsilon.png`: reducción progresiva del parámetro epsilon (exploración).
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

