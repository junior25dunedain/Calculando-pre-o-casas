import pandas as pd
import sklearn.linear_model as lr
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import *

# Criação de um modelo que prever os preços de casas de acordo com atributos como tamanho do lote, localização da casa, numero de quartos e banheiros
df = pd.read_csv('kc_house_data.csv')
print(df.head())

# ciclo 1
# separando dados de treino
X_treino = df.drop(['price','date','id'],axis=1)
Y_treino = df['price'].copy()

# criando o modelo regressor linear
modelo = lr.LinearRegression()
# treinando o modelo
modelo.fit(X_treino,Y_treino)
# gerando os dados preditos do modelo
pred = modelo.predict(X_treino)

# metricas de performece
df1 = df.copy()

df1['predição'] = pred

# erro absoluto
df1['Erro'] = df1['price'] - df1['predição']

df1['Erro_abs'] = np.abs(df1['Erro'])

print(df1[['id','price','predição','Erro','Erro_abs']].head())
print('\n',f"O erro absoluto medio é {round(df1['Erro_abs'].mean(),2)}")

# erro relativo
df1['Erro_percent'] = (df1['price'] - df1['predição'])/ df1['price']

df1['Erro_percent_abs'] = np.abs(df1['Erro_percent'])

print(df1[['id','price','predição','Erro','Erro_abs','Erro_percent','Erro_percent_abs']].head())
print('\n',f"O erro percentual medio é {round(df1['Erro_percent_abs'].mean(),3)*100}%")

# ciclo2
# dados de entrada
X = df.drop(['price','date','id'],axis=1)
# dados alvo
Y = df['price'].copy()

# separando dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(X,Y,test_size=0.25,random_state=42)

# criando modelo2 regressor
modelo2 = lr.LinearRegression()

# treinando o modelo
modelo2.fit(x_treino,y_treino)
#gerando os dados de saida do modelo2
pred2_treino = modelo2.predict(x_treino)

pred2_teste = modelo2.predict(x_teste)

# metricas de avaliação do modelo regressor
mae_treino = mean_absolute_error(y_treino,pred2_treino)
mape_treino = mean_absolute_percentage_error(y_treino,pred2_treino)
print('\n',f"O erro absoluto medio dos dados de treino é {round(mae_treino,2)}")
print('\n',f"O erro percentual medio dos dados de treino é {round(mape_treino,2)*100}%")

mae_teste = mean_absolute_error(y_teste,pred2_teste)
mape_teste = mean_absolute_percentage_error(y_teste,pred2_teste)
print('\n',f"O erro absoluto medio dos dados de teste é {round(mae_teste,2)}")
print('\n',f"O erro percentual medio dos dados de teste é {round(mape_teste,3)*100}%")
print('\n',f"O R² é {round(r2_score(y_teste,pred2_teste),2)}")

