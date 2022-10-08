import pandas as pd
import glob
import os
from datetime import timezone
from pytz import timezone
import pytz
import matplotlib.pyplot as plt
import seaborn as sns


class Funcoes:

    @staticmethod
    def unir_arquivos():
        """
        Localiza os arquivos csv na pasta, os concatena num dataframe
        e salva num csv compilado.
        """
        arquivos = os.path.join('dados', '*.csv')
        arquivos = glob.glob(arquivos)
        df = pd.concat(map(pd.read_csv, arquivos), ignore_index=True)
        df.to_csv('dados/focos_queimadas.csv', index=False)
        return True

    @staticmethod
    def converter_fuso(row):
        """
        Recebe uma data, seta o timezone UTC
        e retorna a data com fuso horário -3h
        """
        # utc = pytz.utc
        dt = pytz.utc.localize(row.datahora)
        br_dt = dt.astimezone(timezone('America/Sao_Paulo'))
        return br_dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def converter_data_hora(df, coluna, tipo):
        """
        Recebe o dataframe, uma coluna de data e hora
        E retorna data ou hora
        """
        if tipo == 'data':
            return df[coluna].dt.date
        else:
            return df[coluna].dt.time

    @staticmethod
    def classificar_risco_fogo(row):
        """
        Recebe o valor de riscofogo, e retorna sua classificação
        """
        if 0 <= row.riscofogo < 0.15:
            return 'minimo'
        elif 0.15 <= row.riscofogo < 0.4:
            return 'baixo'
        elif 0.4 <= row.riscofogo < 0.7:
            return 'medio'
        elif 0.7 <= row.riscofogo < 0.95:
            return 'alto'
        elif 0.95 <= row.riscofogo <= 1:
            return 'critico'
        else:
            return 'na'

    @staticmethod
    def categorizar_risco_fogo(row):
        """
        Recebe o valor de riscofogo, e retorna uma categorização binária
        """
        if 0 <= row.riscofogo < 0.7:
            return 0
        elif 0.7 <= row.riscofogo <= 1:
            return 1
        else:
            return 0

    @staticmethod
    def plotar_grafico_linha(_df: any, _x: str, _y: str, titulo: str, y_label: str, x_label: str, imagem='padrao.png'):
        plt.subplots(figsize=(14, 6))
        ax = sns.lineplot(data=_df, x=_x, y=_y, marker='o')
        for x, y in zip(_df[_x], _df[_y]):
            plt.text(x=x, y=y, s='{:.0f}'.format(y), color='red', fontsize=10, horizontalalignment='right',
                     verticalalignment='center_baseline')
        plt.title(titulo, fontsize=16)
        plt.ylabel(y_label, fontsize=13)
        plt.xlabel(x_label, fontsize=13)
        plt.tight_layout()
        plt.savefig(f'dados/{imagem}', dpi=100)
