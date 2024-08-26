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
    plt.show()
 
# Función para entrenar el modelo de regresión lineal
def update_w_and_b(X, y, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    N = len(X)
    for i in range(N):
        dl_dw += -2*X[i]*(y[i] - (w*X[i] + b))
        dl_db += -2*(y[i] - (w*X[i] + b))
    w = w - (1/float(N))*dl_dw*alpha
    b = b - (1/float(N))*dl_db*alpha
    return w, b

# Función para graficar el modelo de regresión lineal en determinado epoch
def plot_epoch(X, y, w, b, e, loss):
    plt.scatter(X, y, color='blue', alpha=0.5)
    plt.plot([min(X), max(X)], [min(X) * w + b, max(X) * w + b], color='red')
    plt.title("Epoch {} | Loss: {:.2f} | w:{:.4f}, b:{:.4f}".format(e, loss, w, b))
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
    return w_original, b_original

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

# Función main
def main():
    # Leer los datos del archivo csv
    df = pd.read_csv('./data/usa_housing.csv')
    X, y = clean_data(df)

    # Separar los datos en entrenamiento y prueba (80% - 20%)
    X_train = X[:int(len(X)*0.8)]
    y_train = y[:int(len(y)*0.8)]
    X_test = X[int(len(X)*0.8):]
    y_test = y[int(len(y)*0.8):]

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
    w_original, b_original = train_and_plot(X_train, y_train, norm_X, norm_Y, w, b, alpha)

    # Hacer predicciones de los datos de prueba
    y_pred = predict(X_test, w_original, b_original)

    # Calcular el error cuadrático medio
    mse = avg_loss(X_test, y_test, w_original, b_original)
    print("Error cuadrático medio de la predicción: {:.2f}".format(mse))
    print()
    
    # Imprimir las 5 predicciones más cercanas a la realidad y las 5 más alejadas
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

    # Graficar los datos de prueba y las predicciones
    plt.scatter(X_test, y_test, color='green', alpha=0.5)
    plt.plot(X_test, y_pred, color='red')
    plt.title("Predicciones")
    plt.show()

if __name__ == "__main__":
    main()
