# -*- coding: utf-8 -*-
"""
Streamlit App para análise de séries temporais.
Permite upload de arquivo, configuração de parâmetros e visualização de resultados.
"""

import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px



def formatar_colunas(colunas):
    import unicodedata
    return [
        ''.join(
            c for c in unicodedata.normalize('NFKD', coluna.lower())
            if c.isalnum() or c == '_'
        ).replace(' ', '_')
        for coluna in colunas
    ]

# Configuração inicial do Streamlit
st.title("Análise de Séries Temporais")
st.sidebar.header("Configurações")

# Upload do arquivo
dados = st.sidebar.file_uploader("Faça o upload do arquivo CSV", type=["csv","txt"])

if dados is not None:
    # Parâmetros de configuração do upload
    separador = st.sidebar.text_input("Separador de colunas", value=";")
    decimal = st.sidebar.text_input("Separador decimal", value=",")

    # Leitura do arquivo
    df = pd.read_csv(dados, sep=separador, decimal=decimal)

    # Função para formatar colunas


# Renomear colunas
    df.columns = formatar_colunas(df.columns)

    # Configurar colunas de datas
    try:
        df['data_interrupcao'] = pd.to_datetime(df['data_interrupcao'], format='%d/%m/%Y %H:%M:%S')
        df['inicio_ordem'] = pd.to_datetime(df['inicio_ordem'], format='%d/%m/%Y %H:%M:%S')
    except Exception as e:
        st.error("Erro ao converter colunas de datas: " + str(e))

    st.write("Visualização inicial dos dados:", df.head())

    # Filtro por descrição de causa
    descricao_causa = st.sidebar.text_input("Descrição da causa", value="180-QUEDA OU CRESCIMENTO DE ÁRVORE")
    df = df[df['descricao_causa'] == descricao_causa]

    # Criar novas colunas para períodos
    df['data_diaria'] = df['data_interrupcao'].dt.date
    df['data_semanal'] = df['data_interrupcao'].dt.to_period('W').apply(lambda r: r.start_time)
    df['data_mensal'] = df['data_interrupcao'].dt.to_period('M').apply(lambda r: r.start_time)

    # Agregações
    agregacao_diaria = df.groupby(['data_diaria']).size().reset_index(name='qtd_eventos_diarios')
    agregacao_diaria.columns = ['ds', 'y']

    agregacao_semanal = df.groupby(['data_semanal']).size().reset_index(name='qtd_eventos_semanais')
    agregacao_mensal = df.groupby(['data_mensal']).size().reset_index(name='qtd_eventos_mensais')

    # Exibir resultados
    st.write("Agregação Diária:", agregacao_diaria)
    st.write("Agregação Semanal:", agregacao_semanal)
    st.write("Agregação Mensal:", agregacao_mensal)

    # Decomposição da série temporal (diária)
    agregacao_diaria['ds'] = pd.to_datetime(agregacao_diaria['ds'])
    result = seasonal_decompose(agregacao_diaria['y'], model='additive', period=7)

    # Plotar decomposição
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    result.observed.plot(ax=axes[0], title="Série Observada")
    result.trend.plot(ax=axes[1], title="Tendência")
    result.seasonal.plot(ax=axes[2], title="Sazonalidade")
    result.resid.plot(ax=axes[3], title="Resíduos")
    plt.tight_layout()
    st.pyplot(fig)

    # Previsão com Prophet
    st.subheader("Modelo de Previsão Prophet")
    model = Prophet()
    model.fit(agregacao_diaria)

    # Configurar previsão futura
    periodos = st.sidebar.number_input("Dias para previsão", min_value=1, max_value=365, value=30)
    future = model.make_future_dataframe(periods=periodos)
    forecast = model.predict(future)

    # Visualizar previsão
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    st.write("Previsão:", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
else:
    st.info("Aguardando upload do arquivo...")
