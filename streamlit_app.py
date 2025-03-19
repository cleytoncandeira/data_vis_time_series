import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOCAL_PATHS = {
    "Produto": r"df_produto_proagro.csv",
    "Estado (UF)": r"df_UF_proagro.csv"
}

def load_data(groupby_option):
    file_path = LOCAL_PATHS[groupby_option]
    if os.path.exists(file_path):
        st.success(f"📂 Carregando dados de: {file_path}")
        df = pd.read_csv(file_path)
        if "ANO_MES" not in df.columns:
            st.error("❌ A coluna 'ANO_MES' não foi encontrada no arquivo.")
            return None
        df["ANO_MES"] = pd.to_datetime(df["ANO_MES"], errors="coerce", infer_datetime_format=True)
        df.set_index("ANO_MES", inplace=True)
        return df
    else:
        st.error(f"❌ Arquivo {file_path} não encontrado.")
        return None

class TimeSeriesAnalysis:
    def __init__(self, df, groupby_col, value_col, period=12):
        self.df = df
        self.groupby_col = groupby_col
        self.value_col = value_col
        self.period = period
        self.results = {}
        self.successful_groups = []

    def decompose_series(self):
        grouped = self.df.groupby(self.groupby_col)
        for group, data in grouped:
            data = data[[self.value_col]].copy()
            data.dropna(inplace=True)
            try:
                decomposition = seasonal_decompose(data[self.value_col], model='additive', period=self.period)
                self.results[group] = decomposition
                self.successful_groups.append(group)
            except ValueError:
                pass

    def plot_decomposition(self, group):
        if group not in self.results:
            st.warning(f"Nenhuma decomposição encontrada para {group}.")
            return

        decomposition = self.results[group]
        # Garante que cada parte da decomposição tenha índice DatetimeIndex
        for attr in ["observed", "trend", "seasonal", "resid"]:
            series = getattr(decomposition, attr)
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index, format="%Y-%m", errors="coerce")

        # Definição de cores e markers para cada componente
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['circle', 'square', 'diamond', 'x']
        titles = ["Série Original", "Tendência", "Sazonalidade", "Ruído"]

        # Cria subplots com 4 linhas e 1 coluna, compartilhando o eixo X
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=titles)

        fig.add_trace(go.Scatter(
            x=decomposition.observed.index,
            y=decomposition.observed,
            mode='lines+markers',
            name='Original',
            marker=dict(symbol=markers[0], size=6, color=colors[0]),
            line=dict(color=colors[0])
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=decomposition.trend.index,
            y=decomposition.trend,
            mode='lines+markers',
            name='Tendência',
            marker=dict(symbol=markers[1], size=6, color=colors[1]),
            line=dict(color=colors[1])
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=decomposition.seasonal.index,
            y=decomposition.seasonal,
            mode='lines+markers',
            name='Sazonalidade',
            marker=dict(symbol=markers[2], size=6, color=colors[2]),
            line=dict(color=colors[2])
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=decomposition.resid.index,
            y=decomposition.resid,
            mode='lines+markers',
            name='Ruído',
            marker=dict(symbol=markers[3], size=6, color=colors[3]),
            line=dict(color=colors[3])
        ), row=4, col=1)

        # Atualiza os eixos X e Y para exibir os valores em preto e o formato de ano para o eixo X
        for i in range(1, 5):
            fig.update_xaxes(tickformat="%Y", tickfont=dict(color="black"), row=i, col=1)
            fig.update_yaxes(tickfont=dict(color="black"), row=i, col=1)

        # Configura o layout com fundo branco e remove o título geral
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=800,
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            legend=dict(font=dict(color="black"))
        )

        # Atualiza os títulos dos subplots para que fiquem com fonte preta
        if "annotations" in fig.layout:
            for annotation in fig.layout.annotations:
                annotation.font.color = "black"

        st.plotly_chart(fig, use_container_width=True)

# --- Interface do Streamlit ---
st.title("📊 Análise de Séries Temporais das Variáveis do PROAGRO de Matrizes de Dados")

col_mapping = {
    "AREA_AMPARADA_ADESAO": "Área Amparada",
    "AREA_AMPARADA_ComCOP": "Área Amparada Sinistrada",
    "AREA_AMPARADA_ComCOP_DEFERIDA": "Área Amparada Sinistrada Deferida",
    "QTD_ADESAO": "Número de Contratos de Adesão",
    "QTD_ComCOP": "Número de Contratos Sinistrados",
    "QTD_ComCOP_DEFERIDA": "Número de Contratos Sinistrados Deferidos",
    "VL_AMPARADO_ComCOP_DEF": "Valor Amparado Sinistrado",
    "VL_COBERTURA_ComCOP_DEFERIDA_DEF": "Valor de Cobertura Sinistrado Deferido",
    "VL_ADICIONAL_ADESAO_DEF": "Valor Adicional"
}

uf_mapping = {
    "RO": "Rondônia",
    "AC": "Acre",
    "AM": "Amazonas",
    "RR": "Roraima",
    "PA": "Pará",
    "AP": "Amapá",
    "TO": "Tocantins",
    "MA": "Maranhão",
    "PI": "Piauí",
    "CE": "Ceará",
    "RN": "Rio Grande do Norte",
    "PB": "Paraíba",
    "PE": "Pernambuco",
    "AL": "Alagoas",
    "SE": "Sergipe",
    "BA": "Bahia",
    "MG": "Minas Gerais",
    "ES": "Espírito Santo",
    "RJ": "Rio de Janeiro",
    "SP": "São Paulo",
    "PR": "Paraná",
    "SC": "Santa Catarina",
    "RS": "Rio Grande do Sul",
    "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso",
    "GO": "Goiás",
    "DF": "Distrito Federal"
}

groupby_option = st.radio("📍 Escolha o agrupamento:", ["Produto", "Estado (UF)"])
groupby_col = "PRODUTO" if groupby_option == "Produto" else "UF"

df = load_data(groupby_option)
if df is not None:
    st.write("🔍 Visualização dos dados:", df.head())

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_columns:
        st.subheader("📈 Escolha a métrica a ser analisada:")
        options = {col_mapping.get(col, col): col for col in numeric_columns}
        selected_display = st.radio("Selecione a métrica:", list(options.keys()))
        value_col = options[selected_display]

        if st.button("🚀 Executar Análise"):
            st.session_state["analysis"] = TimeSeriesAnalysis(df, groupby_col, value_col)
            st.session_state["analysis"].decompose_series()
            st.session_state["successful_groups"] = st.session_state["analysis"].successful_groups

        if "successful_groups" in st.session_state and st.session_state["successful_groups"]:
            if groupby_option == "Estado (UF)":
                uf_options = {uf_mapping.get(uf, uf): uf for uf in st.session_state["successful_groups"]}
                selected_full = st.selectbox("Selecione um grupo para visualização:", list(uf_options.keys()))
                selected_group = uf_options[selected_full]
            else:
                selected_group = st.selectbox("Selecione um grupo para visualização:", st.session_state["successful_groups"])

            if st.button("📊 Plotar Decomposição"):
                if "analysis" in st.session_state:
                    st.session_state["analysis"].plot_decomposition(selected_group)
                else:
                    st.error("❌ Erro: Nenhuma análise foi encontrada. Execute a análise primeiro.")
    else:
        st.warning("Nenhuma métrica numérica encontrada no arquivo.")
