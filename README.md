# implementacion_ML_sin_framework

Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)

## Descripción
Este proyecto es una implementación básica de un modelo de regresión lineal para predecir el precio de viviendas en los Estados Unidos. Se utiliza un conjunto de datos que contiene información sobre casas en los Estados Unidos. Por el momento este modelo solo se enfoca en las variables `sqft_living` y `price`, se descartan las demás features.

La implementación está hecha completamente desde cero en Python, sin el uso de bibliotecas de machine learning avanzadas. El código incluye funciones para la limpieza de datos, normalización, entrenamiento del modelo, predicción y una pequeña evaluación del rendimiento.

## Descripción de los datos
Los datos utilizados provienen de [Kaggle](https://www.kaggle.com/code/engelhernndezgonzlez/usa-house-price-eda](https://www.kaggle.com/datasets/fratzcan/usa-house-prices). El dataset incluye 18 variables en total, contiene instancias en las que no se hizo el cálculo del precio (`price = 0`) y los datos tienen outliers muy marcados. Además, por la cantidad de de datos que hay en el dataset y para tener un programa más rápido y eficiente se utilizarán solo las casas que tienen un área cuadrada que va de 0 a 3000.

## Bibliotecas necesarias
Para ejecutar este proyecto, se requieren las siguientes bibliotecas de Python:

- `pandas`
- `numpy`
- `matplotlib`

## Cómo correrlo en la terminal
1. Asegúrate de tener Python instalado en tu sistema.
2. Instala las bibliotecas necesarias usando pip:
    ```bash
    pip install pandas numpy matplotlib
    ```
3. Coloca los archivos CSV del dataset en la carpeta `./data/`.
4. Ejecuta el script principal desde la terminal:
    ```bash
    python main.py
    ```
5. Durante la ejecución, el programa te pedirá si deseas utilizar datos precalculados o randomizar los datos de entrenamiento y prueba.

## Resultados de los datos precalculados

## Conclusión
Este proyecto demuestra cómo se puede implementar un modelo de regresión lineal básico desde cero en Python, sin depender de frameworks de machine learning. El código incluye todos los pasos esenciales, desde la limpieza y preparación de los datos hasta la evaluación del modelo, lo que permite comprender los fundamentos del aprendizaje automático.
