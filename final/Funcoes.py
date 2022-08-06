import pandas as pd
import glob
import os
from datetime import datetime, timezone, timedelta, time, date
from pytz import timezone
import pytz


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
        #utc = pytz.utc
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
        if row.riscofogo >= 0 and row.riscofogo < 0.15:
            return 'minimo'
        elif row.riscofogo >= 0.15 and row.riscofogo < 0.4:
            return 'baixo'
        elif row.riscofogo >= 0.4 and row.riscofogo < 0.7:
            return 'medio'
        elif row.riscofogo >= 0.7 and row.riscofogo < 0.95:
            return 'alto'
        elif row.riscofogo >= 0.95 and row.riscofogo <= 1:
            return 'critico'
        else:
            return 'na'
        
    @staticmethod
    def categorizar_risco_fogo(row):
        """
        Recebe o valor de riscofogo, e retorna uma categorização binária
        """
        if row.riscofogo >= 0 and row.riscofogo < 0.7:
            return 0
        elif row.riscofogo >= 0.7 and row.riscofogo <= 1:
            return 1
        else:
            return 0