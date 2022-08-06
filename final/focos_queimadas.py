# Bibliotecas

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timezone, timedelta, time, date
from os.path import exists

from Funcoes import Funcoes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# importar dados

arquivo_csv = 'dados/focos_queimadas.csv'

if not exists(arquivo_csv):
    Funcoes.unir_arquivos()

df_focos_queimadas = pd.read_csv(arquivo_csv, sep=',', encoding='latin 1')

# Tratar dados

df_focos_queimadas.drop(['satelite', 'pais'], axis=1, inplace=True)
df_focos_queimadas.describe()

# Zerando os valores riscofogo, diasemchuva, precipitacao quando vazios

df_focos_queimadas.loc[df_focos_queimadas.diasemchuva.isna(), 'diasemchuva'] = 0
df_focos_queimadas.loc[df_focos_queimadas.precipitacao.isna(), 'precipitacao'] = 0
df_focos_queimadas.loc[df_focos_queimadas.riscofogo.isna(), 'riscofogo'] = 0

"""
Converter nome dos estados em UFs
"""

with open('dados/estados_map.json', encoding='utf-8') as json_file:
    estados_map = json.load(json_file)

df_focos_queimadas = df_focos_queimadas.replace({'estado': estados_map})

"""
Trabalhando com as datas
"""

# Converter a datahora para um objeto data

df_focos_queimadas['datahora'] = pd.to_datetime(df_focos_queimadas['datahora'], errors='coerce')

# Converter o objeto datahora para o fuso de SP (que é igual o de Brasilia)

df_focos_queimadas['datahora_tz'] = df_focos_queimadas.apply(Funcoes.converter_fuso, axis=1)

# Converter a datahora com fuso de Brasília, para um objeto data

df_focos_queimadas['datahora_tz'] = pd.to_datetime(df_focos_queimadas['datahora_tz'], errors='coerce')

# Separar a data e e hora em colunas diferentes

df_focos_queimadas['data'] = Funcoes.converter_data_hora(df_focos_queimadas, 'datahora_tz', 'data')
df_focos_queimadas['hora'] = Funcoes.converter_data_hora(df_focos_queimadas, 'datahora_tz', 'hora')

# Dia da semana
df_focos_queimadas['dia_semana'] = df_focos_queimadas['datahora_tz'].dt.weekday

# Mês do ano
df_focos_queimadas['mes'] = df_focos_queimadas['datahora_tz'].dt.month

"""
Risco fogo

Vamos adicionar a coluna riscofogo_nivel para classificar os níveis do risco de fogo, que o próprio instituto utiliza:

- Mínimo: abaixo de 0,15; 
- Baixo: de 0,15 a 0,4; 
- Médio: de 0,4 a 0,7; 
- Alto: de 0,7 a 0,95 ; 
- Crítico: acima de 0,95 até 1.
"""

df_focos_queimadas['riscofogo_nivel'] = df_focos_queimadas.apply(Funcoes.classificar_risco_fogo, axis=1)

"""
Vamos adicionar a coluna riscofogo_categoria para fazer uma classificação binária

- Abaixo de 0,7: 0 (Mínimo, Baixo, Médio)
- Acima de 0,7: 1 (Alto, Crítico)

"""

df_focos_queimadas['riscofogo_categoria'] = df_focos_queimadas.apply(Funcoes.categorizar_risco_fogo, axis=1)


"""
Tratamento de dados para os modelos de machine learning

Será criada uma cópia do dataframe anterior para preparar os dados para os modelos de machine learning. 
O dataframe original, será utilizado para a exploração de dados.
"""

df_focos_queimadas_ml = df_focos_queimadas.copy()

# Excluir colunas que não serão utilizadas
df_focos_queimadas_ml.drop(['datahora', 'datahora_tz', 'data', 'hora', 'latitude', 'longitude'], axis=1, inplace=True)

labelencoder = LabelEncoder()
df_focos_queimadas_ml['riscofogo_nivel_id'] = labelencoder.fit_transform(df_focos_queimadas_ml['riscofogo_nivel'])
df_focos_queimadas_ml['municipio_id'] = labelencoder.fit_transform(df_focos_queimadas_ml['municipio'])

# Transformar as colunas a seguir em colunas dummy, isso impede que elas sejam lidas pelo algoritmo como ordinais.

_df1 = pd.get_dummies(df_focos_queimadas_ml['bioma'], prefix="bioma")
_df2 = pd.get_dummies(df_focos_queimadas_ml['estado'], prefix="estado")

df_focos_queimadas_ml = pd.concat([df_focos_queimadas_ml, _df1], axis=1)
df_focos_queimadas_ml = pd.concat([df_focos_queimadas_ml, _df2], axis=1)

df_focos_queimadas_ml.drop(['riscofogo_nivel', 'municipio', 'bioma', 'estado'], axis=1, inplace=True)

print(df_focos_queimadas_ml.head())