import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []

        # Inicialización de pesos y sesgos con valores aleatorios para cada capa
        for i in range(len(layer_sizes) - 1):
            weight = np.random.rand(layer_sizes[i], layer_sizes[i + 1])/20
            self.weights.append(weight)

    def lossFn(self, y, h):
        return np.mean((y - h) ** 2)

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def logistic_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, inputs):
        activations = [inputs]  # Lista para almacenar las activaciones de cada capa
        #print(activations)
        # Calcula las activaciones pasando la entrada a través de las capas
        for i in range(len(self.weights)):
            inp = np.dot(activations[i], self.weights[i])
            activation = self.logistic(inp)
            activations.append(activation)
        return activations

    # Propagación hacia atrás: calcula los errores y los propaga de vuelta a través de la red
    def backward_propagation(self, activations, expected_output):
        # Calcula el error de la capa de salida
        mean_error = self.lossFn(expected_output, activations[-1])
        errors = [expected_output - activations[-1]]
        sys.stdout.write(f'\r{mean_error}')
        sys.stdout.flush()
        # Propaga el error hacia atrás y calcula el error para cada capa
        for i in range(len(activations) - 2, 0, -1):
            error = np.dot(errors[0], self.weights[i].T) * self.logistic_derivative(activations[i])
            errors.insert(0, error)
        return errors

    # Actualización de pesos y sesgos usando los errores calculados
    def update_weights(self, activations, errors, learning_rate):
        for i in range(len(self.weights)):
            # Actualiza los pesos con el gradiente del error y la tasa de aprendizaje
            self.weights[i] += np.dot(activations[i].T, errors[i]) * learning_rate

    # Función de entrenamiento: ejecuta la propagación hacia adelante y hacia atrás
    def train(self, inputs, expected_output, learning_rate, iterations):
        self.errors = []  # Inicializa una lista para almacenar los errores
        for i in range(iterations):
            activations = self.forward_propagation(inputs)
            errors = self.backward_propagation(activations, expected_output)
            self.update_weights(activations, errors, learning_rate)
            mean_error = self.lossFn(expected_output, activations[-1])
            self.errors.append(mean_error)  # Almacena el error medio en la lista

    # Función para graficar la curva de aprendizaje
    def plot_learning_curve(self):
        plt.plot(self.errors)
        plt.title('Curva de Aprendizaje')
        plt.xlabel('Iteraciones')
        plt.ylabel('Error Medio')
        plt.show()
            
    # Función de predicción: calcula las salidas de la red para las entradas dadas
    def predict(self, inputs):
        activations = self.forward_propagation(inputs)
        return activations[-1]

# Ejemplo de uso
# Configuración de tamaños de capa: 4 entradas, 5 neuronas en la primera capa oculta, y así sucesivamente hasta 3 salidas
layer_sizes = [4,245,300,3]
nn = NeuralNetwork(layer_sizes)

# Carga y preparación de datos de entrenamiento
df = pd.read_csv('../iris.csv')

# Extracción y normalización de las entradas
inputs = df.drop('species', axis=1)
normInputs = inputs.copy()
# Normaliza las características para que estén en una escala de 0 a 1
for column in inputs.columns[0:]:
    normInputs[column] = (inputs[column] - inputs[column].min()) / (inputs[column].max() - inputs[column].min())


# Convierte los datos normalizados y las salidas a arrays de numpy
normInputs = np.array(normInputs)
outputs = np.array(pd.get_dummies(df['species']))

# Entrenamiento de la red con los datos
nn.train(normInputs, outputs, learning_rate=0.01, iterations=5000)

# Uso de la red para hacer una predicción con un ejemplo de entrada
print(nn.predict(np.array([5.3,3.7,1.5,0.2])))
print(nn.predict(np.array([6.1,2.8,4,1.3])))
print(nn.predict(np.array([6.9,3.2,5.7,2.3])))

nn.plot_learning_curve()
