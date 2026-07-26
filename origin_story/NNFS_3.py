import numpy as np


inputs =[1.0, 2.0, 3.0, 2.5]
weights =[[0.2, 0.8, -0.5, 1.0], 
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]

bias =[2, 3, 0.5]

# Calcular el producto punto (dot product) y sumar el bias
outputs = np.dot(weights, inputs) + bias  # Esto debe definir 'outputs'

# Imprimir el resultado
print(outputs)