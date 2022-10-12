import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *


# modelo de regressão linear que avalia a relação entre peso e altura de estudantes
dt = pd.read_csv('weight-height.csv')
print(dt)
df = dt[['Height','Weight']]
df.columns = ['altura','peso']
print(df)

# equações de reta
x = df['altura']
y1 = -190 +5*x
y2 = -260 + 6.2*x
y3 = -350.74 + 7.72*x

# least square error
df['y1'] = y1
df['y2'] = y2

df['error01'] = (df['peso']-df['y1'])**2
df['error02'] = (df['peso']-df['y2'])**2

print(f"SS M01: {np.sum(df['error01'])}")
print(f"SS M02: {df['error02'].sum()}")

#criação do modelo linear
modelo = lm.LinearRegression()
X = np.array(df['altura']).reshape(-1,1)
y = np.array(df['peso']).reshape(-1,1)

modelo.fit(X,y)
pred = modelo.predict(X)

print(f'coeficiente da equação da reta (a): {modelo.coef_}')
print(f'coeficiente da equação da reta (b): {modelo.intercept_}')


# exibindo os dados da base altura x peso

plt.scatter(df['altura'],df['peso'])
plt.plot(x,y1,color='red',label='reta aproximada 1')
plt.plot(x,y2,color='green',label='reta aproximada 2')
plt.plot(x,y3,color='black',label='reta do modelo regressor')
plt.title('Relação entre altura e o peso')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.show()

# metrica mais importante de um modelo regressor Linear é o R²
# R² (coeficiente de determinação), quanto maior o valor de R² melhor é o desempenho do modelo regressor
# já um pequeno valor de R² indica um pior desempenho do modelo regressor

r2 = r2_score(y,pred)
print(f'O R² é {round(r2,3)}')