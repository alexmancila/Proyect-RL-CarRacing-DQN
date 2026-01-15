# Presentación: Aprendizaje por Refuerzo Profundo en CarRacing-v3

## Descripción

Presentación extendida en Beamer LaTeX para la defensa del trabajo final ante el docente. La presentación es **completa y detallada**, cubriendo todos los aspectos del proyecto: problema, metodología, algoritmos, arquitectura, resultados con gráficos, análisis y conclusiones.

## Contenido Detallado (30+ diapositivas)

### 1. **Introducción y Motivación** (2 diapositivas)
   - Problema y motivación del trabajo
   - Visualización del entorno CarRacing-v3

### 2. **Metodología** (5 diapositivas)
   - Discretización de acciones (problema y solución)
   - Preprocesamiento de imágenes (pipeline completo)
   - Código: Preprocesamiento en Python
   - Configuración del entrenamiento (hiperparámetros)
   - Código: Configuración del proyecto

### 3. **Algoritmos** (4 diapositivas)
   - Deep Q-Network (DQN) - Base teórica con ecuaciones
   - Double DQN - Reducción de sesgo con formulación matemática
   - Dueling DQN - Arquitectura mejorada
   - Código: Arquitectura DQN en PyTorch

### 4. **Arquitectura del Proyecto** (3 diapositivas)
   - Estructura modular: descripción de cada módulo (src/)
   - Código: Bucle de entrenamiento simplificado
   - Flujo de ejecución general (inicialización, entrenamiento, evaluación)

### 5. **Resultados** (10 diapositivas)
   - Evaluación a 200 episodios (tabla con análisis)
   - Evaluación a 1000 episodios (tabla con hallazgos)
   - **Gráfico**: Reward durante entrenamiento (Double DQN) + interpretación
   - **Gráfico**: Loss durante entrenamiento (Double DQN) + interpretación
   - **Gráfico**: Decaimiento de epsilon (exploración) + interpretación
   - **Gráfico**: Crecimiento del replay buffer + interpretación
   - **Gráfico**: Evaluación - Double DQN a 1000 episodios + interpretación
   - **Gráfico**: Comparación DQN Base a 1000 episodios + análisis

### 6. **Análisis Detallado** (3 diapositivas)
   - Comparación cuantitativa (tabla de mejora porcentual)
   - Análisis de estabilidad (Std a 200 vs 1000 episodios)
   - Limitaciones estadísticas y metodológicas

### 7. **Conclusiones y Trabajo Futuro** (4 diapositivas)
   - Conclusiones principales (objetivos cumplidos, hallazgos)
   - Trabajo futuro - Corto plazo (robustez, hiperparámetros, discretización)
   - Trabajo futuro - Técnicas avanzadas (PER, multi-step, DDPG, SAC, PPO)
   - Diapositiva final con contacto

## Características Especiales

✓ **Altamente Visual**
- 8 gráficos de resultados integrados
- Imágenes del entorno
- Tablas de comparación bien formateadas
- Ecuaciones matemáticas claras

✓ **Código Integrado**
- 4 ejemplos de código Python con syntax highlighting
- Fragmentos de los módulos principales
- Código accesible para docentes y estudiantes

✓ **Análisis Profundo**
- Interpretación línea a línea de cada gráfico
- Comparativas cuantitativas
- Análisis de trade-offs
- Conclusiones respaldadas por datos

✓ **Profesional**
- Tema Madrid (colores corporativos)
- Estructura lógica y progresiva
- Numeración automática de diapositivas
- Referencias y vínculos

## Archivos

- **presentacion.tex**: Código fuente LaTeX (800+ líneas)
- **presentacion.pdf**: PDF compilado listo para presentar (30+ diapositivas)
- **README.md**: Este archivo

## Cómo Compilar

```bash
cd Presentación
pdflatex -interaction=nonstopmode presentacion.tex
```

Nota: Se requiere tener LaTeX instalado (MiKTeX en Windows, TeX Live en Linux/Mac)

## Duración

- **Presentación completa**: 15-20 minutos
- **Presentación resumida** (saltando gráficos análisis profundo): 8-10 minutos
- **Presentación ejecutiva** (solo conclusiones): 5-7 minutos

**Recomendación**: Presentar la versión completa (15-20 min) para máximo impacto ante el docente.

## Nota Importante

La presentación está diseñada para ser:
- **Autosuficiente**: el docente puede entenderla sin necesidad del informe
- **Detallada**: cubre código, teoría, resultados y conclusiones
- **Visualmente atractiva**: gráficos y ecuaciones bien integrados
- **Académicamente rigurosa**: respaldada por tablas y análisis

## Estructura de Secciones

1. **Introducción** (2 slides)
2. **Metodología** (5 slides) 
3. **Algoritmos** (4 slides)
4. **Arquitectura** (3 slides)
5. **Resultados** (10 slides con gráficos)
6. **Análisis** (3 slides)
7. **Conclusiones** (4 slides)

**Total**: 31 diapositivas

## Autor

Grupo 4: Koc Góngora, Mancilla Antaya, Meléndez García, Paitán Cano

**Curso**: Aprendizaje por Refuerzo  
**Instituto**: Universidad Nacional de Ingeniería (UNI)  
**Fecha**: 15 de enero de 2026

