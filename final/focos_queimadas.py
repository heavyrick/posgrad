# Bibliotecas

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timezone, timedelta, time, date

import locale

locale.setlocale(locale.LC_ALL, 'pt_BR')
import calendar

from os.path import exists

from Funcoes import Funcoes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
A análise exploratória de dados completa está contida no notebook. Por lá é possível visualizar os gráficos.
"""

# Excluindo o ano de 2022 da análise
df_focos_queimadas = df_focos_queimadas.loc[df_focos_queimadas['ano'] < 2022]

# Variável ano

df_ano = df_focos_queimadas.groupby(['ano'], as_index=True)['datahora'].count().reset_index()
df_ano.rename(columns={'datahora': 'qtde_focos'}, inplace=True)

Funcoes.plotar_grafico_linha(df_ano, 'ano', 'qtde_focos', 'Focos de queimadas por ano', 'Qtde de focos',
                             'Anos', 'queimadas_ano.png')

# Variável mes

df_mes = df_focos_queimadas.groupby(['mes'], as_index=True)['datahora'].count().reset_index()
df_mes.rename(columns={'datahora': 'qtde_focos'}, inplace=True)

# Converter o número do mês em nome
df_mes['mes'] = df_mes['mes'].apply(lambda x: calendar.month_name[x].capitalize())

df_mesano = df_focos_queimadas.groupby(['ano', 'mes'], as_index=True)['datahora'].count().reset_index()
df_mesano.rename(columns={'datahora': 'qtde_focos'}, inplace=True)

df_mesano['mes_nome'] = df_mesano['mes'].apply(lambda x: calendar.month_name[x].capitalize())

Funcoes.plotar_grafico_linha(df_mes, 'mes', 'qtde_focos', 'Focos de queimadas por mês (2015-2021)',
                             'Qtde de focos', 'Meses', 'queimadas_ano.png')


fig = plt.figure(figsize=(20, 25))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, j in zip(range(1, len(df_mesano['ano'].unique()) + 1), df_mesano['ano'].unique()):
    _df = df_mesano[df_mesano['ano'] == j]
    plt.subplot(6, 2, i)
    ax = sns.lineplot(data=_df, x='mes_nome', y='qtde_focos', marker='o')
    ax.set_ylim([0, max(df_mesano['qtde_focos']) + 5000])
    plt.title('Focos de queimadas por mês: ' + str(j), fontsize=20)
    plt.ylabel('Qtde de focos', fontsize=14)
    plt.xlabel('Meses', fontsize=14)
    plt.tight_layout();
    plt.savefig('dados/focos_meses_por_ano.png', dpi=300)

# Variável estado

df_estado = df_focos_queimadas.groupby(['estado'], as_index=True)['datahora'].count().reset_index()
df_estado.rename(columns={'datahora': 'qtde_focos'}, inplace=True)

# Ordenar pelo maior número de focos
df_estado.sort_values(by=['qtde_focos'], ascending=False, inplace=True, ignore_index=True)

# criar coluna com qtde de focos relativa
df_estado['qtde_focos_relativa'] = np.round(df_estado['qtde_focos'] / sum(df_estado['qtde_focos']), 4) * 100

plt.figure(figsize=(16, 8))
graph = plt.bar(df_estado.estado, df_estado.qtde_focos, color='darkorange')
plt.title('Focos de queimadas por estado (2015-2021)', fontsize=20)
plt.ylabel('Qtde de focos', fontsize=18)
plt.xlabel('Estado', fontsize=18)

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x + width / 2, y + height * 1.01, str(np.round(df_estado.qtde_focos_relativa[i], 1)) + '%', weight='bold',
             ha='center', va='bottom')
    i += 1
plt.tight_layout();
plt.savefig('dados/queimadas_estado.png', dpi=300)

# Variável bioma

df_bioma = df_focos_queimadas.groupby(['bioma'], as_index=True)['datahora'].count().reset_index()
df_bioma.rename(columns={'datahora': 'qtde_focos'}, inplace=True)

df_bioma.sort_values(by=['qtde_focos'], ascending=False, inplace=True, ignore_index=True)

# criar coluna com qtde de focos relativa
df_bioma['qtde_focos_relativa'] = np.round(df_bioma['qtde_focos'] / sum(df_bioma['qtde_focos']), 4) * 100

plt.figure(figsize=(12, 6))
graph = plt.bar(df_bioma.bioma, df_bioma.qtde_focos, color='darkblue')
plt.title('Focos de queimadas por bioma (2015-2021)', fontsize=20)
plt.ylabel('Qtde de focos', fontsize=18)
plt.xlabel('Bioma', fontsize=18)

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x + width / 2, y + height * 1.01, str(np.round(df_bioma.qtde_focos_relativa[i], 1)) + '%',
             weight='bold', ha='center', va='bottom')
    i += 1
plt.tight_layout()
plt.savefig('dados/queimadas_bioma.png', dpi=300)

# Variável risco-fogo

df_riscofogo_nivel = df_focos_queimadas.groupby(['riscofogo_nivel'], as_index=True)['datahora'].count().reset_index()
df_riscofogo_nivel.rename(columns={'datahora': 'qtde_focos'}, inplace=True)

df_riscofogo_nivel.sort_values(by=['qtde_focos'], ascending=False, inplace=True, ignore_index=True)

# criar coluna com qtde de focos relativa
df_riscofogo_nivel['qtde_focos_relativa'] = np.round(
    df_riscofogo_nivel['qtde_focos'] / sum(df_riscofogo_nivel['qtde_focos']), 4) * 100

plt.figure(figsize=(12, 6))
graph = plt.bar(df_riscofogo_nivel.riscofogo_nivel, df_riscofogo_nivel.qtde_focos, color='red')
plt.title('Focos de queimadas por risco-fogo nível (2015-2021)', fontsize=20)
plt.ylabel('Qtde de focos', fontsize=18)
plt.xlabel('Risco-fogo', fontsize=18)

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x + width / 2, y + height * 1.01, str(np.round(df_riscofogo_nivel.qtde_focos_relativa[i], 1)) + '%',
             weight='bold', ha='center', va='bottom')
    i += 1
plt.tight_layout()
plt.savefig('dados/queimadas_riscofogo.png', dpi=300)

# variável dias sem chuva

plt.hist(df_focos_queimadas['diasemchuva'], color='blue', edgecolor='black', bins=20)
plt.xlabel('Dias sem chuva', fontsize=18)
plt.ylabel('Qtde de focos', fontsize=18)
plt.title('Número de dias sem chuva até a detecção do foco', fontsize=20)
plt.tight_layout()
plt.savefig('dados/queimadas_diassemchuva.png', dpi=300)

# Variável precipitação

plt.hist(df_focos_queimadas['precipitacao'], color='blue', edgecolor='black', bins=20)
plt.xlabel('Precipitação', fontsize=18)
plt.ylabel('Qtde de focos', fontsize=18)
plt.title('Precipitação acumulada (mm/dia) até o momento do foco', fontsize=20)
plt.tight_layout()
plt.savefig('dados/queimadas_precipitacao.png', dpi=300)

# Variável FRP

plt.hist(df_focos_queimadas['frp'], color='blue', edgecolor='black', bins=20)
plt.xlabel('Precipitação', fontsize=18)
plt.ylabel('Qtde de focos', fontsize=18)
plt.title('Intensidade do fogo em MW no momento do foco', fontsize=20)
plt.tight_layout()
plt.savefig('dados/queimadas_frp.png', dpi=300)

# Correlações

df_focos_queimadas2 = df_focos_queimadas.drop(
    ['municipio', 'latitude', 'longitude', 'dia_semana', 'datahora_tz', 'data', 'hora', 'ano'], axis=1)

_df1 = pd.get_dummies(df_focos_queimadas2['bioma'], prefix="bioma")
df_focos_queimadas2 = pd.concat([df_focos_queimadas2, _df1], axis=1)
df_focos_queimadas2.drop(['bioma'], axis=1, inplace=True)

corr = df_focos_queimadas2.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r')
plt.title('Correlação entre variáveis', fontsize=20)
plt.tight_layout()
plt.savefig('dados/queimadas_correlacao.png', dpi=300)

# Medir importância de algumas variáveis

df_focos_queimadas_imp = df_focos_queimadas.drop(
    ['municipio', 'latitude', 'longitude', 'dia_semana', 'datahora_tz', 'data', 'hora', 'ano'], axis=1)
df_focos_queimadas_imp['bioma'] = labelencoder.fit_transform(df_focos_queimadas_imp['bioma'])

df_focos_queimadas_imp.drop(['frp', 'estado', 'riscofogo', 'riscofogo_nivel'], axis=1, inplace=True)

_X = df_focos_queimadas_imp.iloc[:, 1:-1]
_y = df_focos_queimadas_imp.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(_X, _y, random_state=123)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Importância das features
importances = pd.Series(data=model.feature_importances_, index=X_train.columns)
sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importância de cada feature')

"""
Tratamento de dados para os modelos de machine learning

Será criada uma cópia do dataframe anterior para preparar os dados para os modelos de machine learning. 
O dataframe original, será utilizado para a exploração de dados.
"""

df_focos_queimadas_ml = df_focos_queimadas.copy()

# Excluir colunas que não serão utilizadas
df_focos_queimadas_ml.drop(['datahora', 'datahora_tz', 'data', 'hora', 'latitude', 'longitude', 'municipio', 'estado'],
                           axis=1, inplace=True)

df_focos_queimadas_ml['riscofogo_nivel_id'] = labelencoder.fit_transform(df_focos_queimadas_ml['riscofogo_nivel'])

# Transformar as colunas a seguir em colunas dummy, isso impede que elas sejam lidas pelo algoritmo como ordinais.
_df1 = pd.get_dummies(df_focos_queimadas_ml['bioma'], prefix="bioma", drop_first=True)
df_focos_queimadas_ml = pd.concat([df_focos_queimadas_ml, _df1], axis=1)

df_focos_queimadas_ml.drop(['riscofogo_nivel', 'bioma'], axis=1, inplace=True)

print(df_focos_queimadas_ml.head())
