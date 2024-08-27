import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

# Elimina los outliers de un DataFrame en una columna específica
def find_outliers(df, column):
    # Calcula los cuartiles y el rango intercuartil
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define los límites inferior y superior
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Elimina los outliers
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df

# Limpia los datos de los dataframes de manera que se puedan utilizar para el entrenamiento
def clean_data(df):
    # Error en algunas filas con valores 0 en 'price'
    df = df[df['price'] != 0]

    # Limitando el número de datos para visualización
    df = df[df['sqft_living'] <= 3000]

    # Eliminando outliers en 'price'
    df = find_outliers(df, 'price')

    # Revolver los datos
    # print(df[['price', 'street', 'city']].head())
    df = df.sample(frac=1).reset_index(drop=True)
    # print(df[['price', 'street', 'city']].head())

    # Definiendo variables X y Y
    X = df['sqft_living'].values
    y = df['price'].values

    return X, y  

# Imprime los datos en un gráfico de dispersión
def plot_data(X_train, y_train, X_test, y_test, title='USA Housing'):
    plt.scatter(X_train, y_train, color='blue', alpha=0.5)
    plt.scatter(X_test, y_test, color='green', alpha=0.5)        
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    plt.title(title)
    # xlim agarrando el mínimo y máximo entre X_train y X_test
    plt.xlim(min(min(X_train), min(X_test)), max(max(X_train), max(X_test)) + (max(max(X_train), max(X_test)) * 0.1))
    # ylim agarrando el mínimo y máximo entre y_train y y_test
    plt.ylim(min(min(y_train), min(y_test)), max(max(y_train), max(y_test)) + (max(max(y_train), max(y_test)) * 0.1))
    plt.ticklabel_format(style='plain', axis='y')
    formatter = FuncFormatter(lambda x, _: f'{int(x):,}')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.savefig('./img/data.png')
    plt.show()
 
# Función para recalcular w y b en cada epoch
def update_w_and_b(X, y, w, b, alpha):
    # Inicializa las variables
    dl_dw = 0.0
    dl_db = 0.0
    N = len(X)

    # Calcula las derivadas parciales de la función de costo
    for i in range(N):
        dl_dw += -2*X[i]*(y[i] - (w*X[i] + b))
        dl_db += -2*(y[i] - (w*X[i] + b))

    # Actualización de w y b
    w = w - (1/float(N))*dl_dw*alpha
    b = b - (1/float(N))*dl_db*alpha
    return w, b

# Función para graficar el modelo de regresión lineal en determinado epoch
def plot_epoch(X, y, w, b, e, loss):
    plt.scatter(X, y, color='blue', alpha=0.5)
    plt.plot([min(X), max(X)], [min(X) * w + b, max(X) * w + b], color='red')
    plt.title("Epoch {} | Loss: {:.2f} | w:{:.4f}, b:{:.4f}".format(e, loss, w, b))
    plt.savefig('./img/epoch.png')
    plt.show()

# Función que realiza el entrenamiento del modelo y grafica el progreso, para en el momento que la diferencia de costo entre epoch actual y pasado sea menor a 0.0001
def train_and_plot(X, y, norm_x, norm_y, w, b, alpha):
    last_epoch_loss = 1000000000000001
    loss = 1000000000000000
    e = 0
    w_original = 0
    b_original = 0
    while (last_epoch_loss - loss) > 0.0001:
        w, b = update_w_and_b(norm_x, norm_y, w, b, alpha)
        # Desnormaliza w y b
        w_original = w * np.std(y) / np.std(X)
        b_original = b * np.std(y) + np.mean(y) - w_original * np.mean(X)
        
        last_epoch_loss = loss
        loss = avg_loss(X, y, w_original, b_original)
        #print("Epoch: {}, Loss: {}, w: {}, b: {}".format(e, loss, w_original, b_original))
        #plot_epoch(X, y, w_original, b_original, e, loss)
        e += 1

    print("Diferencia de costo entre epoch actual y pasado (buscando menor a 0.0001): {:.6f}".format(last_epoch_loss - loss))
    print("Epoch: {}, Loss: {:.2f}, w: {:.4f}, b: {:.4f}".format(e, loss, w_original, b_original))
    print()
    plot_epoch(X, y, w_original, b_original, e, loss)
    return w_original, b_original, loss

# Función para calcular el error cuadrático medio
def avg_loss(X, y, w, b):
    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error += (y[i] - (w*X[i] + b))**2
    return total_error / float(N)

# Función para predecir el precio de una casa con base en su tamaño
def predict(X, w, b):
    return w*X + b

# Función para imprimir los mensajes iniciales
def prints_iniciales():
    print("Implementación de una técnica de aprendizaje máquina sin el uso de un framework.")
    print("Diego Velázquez Saldaña - A01177877")
    print()
    print("El dataset utilizado es el de 'USA Housing' que contiene los precios de casas en USA con base en su tamaño. Se utilizará una regresión lineal para predecir el precio de una casa con base en su tamaño.")
    print()
    print("Se utilizará un 80% de los datos para entrenamiento y un 20% para prueba. Se hizo un shuffle de los datos para evitar sesgos y se almacenaron en archivos csv. De cualquier manera, se puede volver a randomizar los datos de entrenamiento/prueba si se desea. Esto suele dar un porcentaje de error de la predicción con base en el entrenamiento de alrededor de ±10%.")
    print()

    datos_precalculados = False

    print("Desea utilizar los datos precalculados (o randomizar los datos de entrenamiento/prueba)? (s/n)")
    respuesta = input()
    if respuesta == 's':
        datos_precalculados = True

    return datos_precalculados

# Función para imprimir las 5 predicciones más cercanas a la realidad y las 5 más alejadas
def prints_predicciones(y_test, y_pred):
    y_diff = y_test - y_pred
    y_diff = np.abs(y_diff)
    y_diff_sorted = np.argsort(y_diff)
    print("5 predicciones más cercanas a la realidad:")
    for i in range(5):
        print("Predicción: {:.2f}, Realidad: {:.2f}".format(y_pred[y_diff_sorted[i]], y_test[y_diff_sorted[i]]))
    
    print()
    print("5 predicciones más alejadas a la realidad:")
    for i in range(1, 6):
        print("Predicción: {:.2f}, Realidad: {:.2f}".format(y_pred[y_diff_sorted[-i]], y_test[y_diff_sorted[-i]]))

    print()

# Función para obtener los datos de entrenamiento y prueba con base en la respuesta del usuario
def datos_entrenamiento_prueba(datos_precalculados):
    if datos_precalculados:
        # Leer datos de entrenamiento
        train_data = pd.read_csv('./data/usa_housing_train.csv')
        X_train = train_data['X_train'].values
        y_train = train_data['y_train'].values

        # Leer datos de prueba
        test_data = pd.read_csv('./data/usa_housing_test.csv')
        X_test = test_data['X_test'].values
        y_test = test_data['y_test'].values
    else:
        # Leer los datos del archivo csv
        df = pd.read_csv('./data/usa_housing.csv')
        X, y = clean_data(df)

        # Separar los datos en entrenamiento y prueba (80% - 20%)
        sep = 0.8
        X_train = X[:int(len(X)*sep)]
        y_train = y[:int(len(y)*sep)]
        X_test = X[int(len(X)*sep):]
        y_test = y[int(len(y)*sep):]

    return X_train, y_train, X_test, y_test

# Función main
def main():
    datos_precalculados = prints_iniciales()
    
    X_train, y_train, X_test, y_test = datos_entrenamiento_prueba(datos_precalculados)

    # Almacenar datos de entrenamiento
    # train_data = pd.DataFrame({'X_train': X_train, 'y_train': y_train})
    # train_data.to_csv('./data/usa_housing_train.csv', index=False)

    # Almacenar datos de prueba
    # test_data = pd.DataFrame({'X_test': X_test, 'y_test': y_test})
    # test_data.to_csv('./data/usa_housing_test.csv', index=False)

    # Graficar los datos de entrenamiento y prueba
    plot_data(X_train, y_train, X_test, y_test)

    # Hacer la normalización de los datos
    norm_X = (X_train - np.mean(X_train)) / np.std(X_train)
    norm_Y = (y_train - np.mean(y_train)) / np.std(y_train)

    # Incializar w y b
    w = 0.0
    b = 0.0
    alpha = 0.01
    
    # Define epoch numbers to plot to visualize progress
    #epoch_plots = [0, 1, 2, 10, 50, 100, epochs]

    # epoch_plots = [i for i in range(0, 101, 10)]

    # Entrenar el modelo y graficar el progreso
    w_original, b_original, mse_t = train_and_plot(X_train, y_train, norm_X, norm_Y, w, b, alpha)

    # Hacer predicciones de los datos de prueba
    y_pred = predict(X_test, w_original, b_original)

    prints_predicciones(y_test, y_pred)

    # Calcular el error cuadrático medio de la predicción e imprimirlo junto con el error cuadrático medio del entrenamiento
    mse_p = avg_loss(X_test, y_test, w_original, b_original)

    porcentaje_error = ((mse_p / mse_t) * 100) - 100

    # Imprimir el porcentaje de error de la predicción con base en el entrenamiento
    print("Error cuadrático medio de la predicción: {:.2f}".format(mse_p))
    print("Error cuadrático medio del entrenamiento: {:.2f}".format(mse_t))
    print("Porcentaje de error de la predicción con base en el entrenamiento: {:.2f}%".format(porcentaje_error))

    # Graficar los datos de prueba y las predicciones
    plt.scatter(X_test, y_test, color='green', alpha=0.5)
    plt.plot(X_test, y_pred, color='red')
    plt.title("Predicciones | Diferencia % entre errores: {:.2f}%".format(porcentaje_error))
    plt.savefig('./img/predicciones.png')
    plt.show()

if __name__ == "__main__":
    main()
