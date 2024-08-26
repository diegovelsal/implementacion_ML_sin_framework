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
    # Imprimir head precios, street y city para verificar que se hayan revuelto
    print(df[['price', 'street', 'city']].head())
    df = df.sample(frac=1).reset_index(drop=True)
    print(df[['price', 'street', 'city']].head())

    # Definiendo variables X y Y
    X = df['sqft_living'].values
    y = df['price'].values

    return X, y  

# Imprime los datos en un gráfico de dispersión
def plot_data(X, Y):
    plt.scatter(X, Y, color='blue', alpha=0.5)
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    plt.title('USA Housing')
    plt.xlim(X.min(), X.max()+(X.max()*0.1))
    plt.ylim(Y.min(), Y.max()+(Y.max()*0.1))
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
    plt.title("Epoch {} | Loss: {} | w:{}, b:{}".format(e, round(loss,2), round(w, 4), round(b, 4)))
    plt.show()

# Función que realiza el entrenamiento del modelo y grafica el progreso
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
    print("Epoch: {}, Loss: {}, w: {}, b: {}".format(e, loss, w_original, b_original))
    return w, b

# Función para calcular el error cuadrático medio
def avg_loss(X, y, w, b):
    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error += (y[i] - (w*X[i] + b))**2
    return total_error / float(N)

# Función para predecir el precio de una casa
def predict(X, w, b):
    return w*X + b

def main():
    df = pd.read_csv('./data/usa_housing.csv')
    X, y= clean_data(df)

    # Hacer la normalización de los datos
    norm_X = (X - np.mean(X)) / np.std(X)
    norm_Y = (y - np.mean(y)) / np.std(y)

    w = 100
    b = -49
    alpha = 0.01
    
    # Define epoch numbers to plot to visualize progress
    #epoch_plots = [0, 1, 2, 10, 50, 100, epochs]

    # epoch_plots = [i for i in range(0, 101, 10)]
    w, b = train_and_plot(X, y, norm_X, norm_Y, w, b, alpha)

if __name__ == "__main__":
    main()
