import streamlit as st
import pandas as pd

# --- CONFIGURAÇÕES INICIAIS ---
st.set_page_config(page_title="Previsão de Vacinação — ML", layout="wide")
st.title("📊 Previsão de Vacinação — Our World in Data")
st.caption("Fonte: Our World in Data — https://ourworldindata.org/covid-vaccinations")

# --- CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    return df

st.subheader("1️⃣ Dados Brutos")
dados = carregar_dados()
st.write("✅ Dados carregados:", dados.shape)
st.dataframe(dados.head())

# --- EXPLORAÇÃO INICIAL ---
st.subheader("2️⃣ Países disponíveis")
paises = sorted(dados["location"].unique())
st.write("Total de países:", len(paises))
st.write(paises[:15], "...")

# ============================================================
# ETAPA 2 — Seleção de país e visualização temporal
# ============================================================

st.divider()
st.subheader("2️⃣ Seleção de País e Visualização")

# --- Seleção de país ---
paises = sorted(dados["location"].unique())
pais = st.selectbox("Escolha o país", paises, index=paises.index("Brazil") if "Brazil" in paises else 0)

# --- Filtragem e limpeza ---
df_pais = dados[dados["location"] == pais].copy()
df_pais = df_pais[["date", "daily_vaccinations", "total_vaccinations", "people_vaccinated", "people_fully_vaccinated"]]
df_pais = df_pais.dropna(subset=["daily_vaccinations"])
df_pais = df_pais.sort_values("date")

st.write(f"**{pais}** — {len(df_pais)} registros disponíveis")

# --- Métricas rápidas ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💉 Total de Vacinas Aplicadas", f"{int(df_pais['total_vaccinations'].max()):,}".replace(",", "."))
with col2:
    st.metric("👥 Pessoas Vacinadas", f"{int(df_pais['people_vaccinated'].max()):,}".replace(",", "."))
with col3:
    st.metric("✅ Totalmente Vacinadas", f"{int(df_pais['people_fully_vaccinated'].max()):,}".replace(",", "."))

# --- Gráfico temporal ---
st.line_chart(df_pais.set_index("date")["daily_vaccinations"], height=300)

# --- Estatísticas descritivas ---
with st.expander("📈 Estatísticas do país selecionado"):
    st.dataframe(df_pais.describe())

# ============================================================
# ETAPA 3 — Previsão de Demanda de Vacinas por País
# ============================================================
from prophet import Prophet
import matplotlib.pyplot as plt

st.divider()
st.subheader("3️⃣ Previsão de Demanda de Vacinas (por país)")

# --- Selecionar países ---
paises = st.multiselect(
    "Selecione um ou mais países para prever:",
    sorted(df["location"].unique()),
    default=["Brazil"]
)

# --- Filtrar e prever para cada país ---
if not paises:
    st.warning("Selecione pelo menos um país para gerar previsões.")
else:
    for pais in paises:
        st.markdown(f"### 🌍 {pais}")

        df_pais = df[df["location"] == pais].copy()
        df_pais["date"] = pd.to_datetime(df_pais["date"])
        df_pais = df_pais[["date", "daily_vaccinations"]].dropna()

        if df_pais.empty or len(df_pais) < 10:
            st.warning(f"Dados insuficientes para {pais}.")
            continue

        # Preparar dados para Prophet
        df_forecast = df_pais.rename(columns={"date": "ds", "daily_vaccinations": "y"})
        df_forecast = df_forecast[df_forecast["y"] > 0]

        # Treinar modelo Prophet
        modelo = Prophet(daily_seasonality=True, yearly_seasonality=True)
        modelo.fit(df_forecast)

        # Criar horizonte de previsão (30 dias)
        futuro = modelo.make_future_dataframe(periods=30)
        previsao = modelo.predict(futuro)

        # Plotar gráfico
        fig1, ax1 = plt.subplots()
        modelo.plot(previsao, ax=ax1)
        plt.title(f"Previsão de vacinas — {pais}")
        st.pyplot(fig1)

        # Tabela formatada
        df_pretty = previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10).rename(columns={
            "ds": "Data",
            "yhat": "Vacinas previstas (média)",
            "yhat_lower": "Intervalo inferior",
            "yhat_upper": "Intervalo superior"
        })
        df_pretty["Vacinas previstas (média)"] = df_pretty["Vacinas previstas (média)"].round(0).astype(int)
        df_pretty["Intervalo inferior"] = df_pretty["Intervalo inferior"].round(0).astype(int)
        df_pretty["Intervalo superior"] = df_pretty["Intervalo superior"].round(0).astype(int)
        st.dataframe(df_pretty, use_container_width=True)

        # Cálculo total previsto
        proximo_mes = previsao.tail(30)
        estimativa_total = int(proximo_mes["yhat"].sum())
        st.success(f"💉 Estimativa para {pais} nos próximos 30 dias: **{estimativa_total:,} doses**")

        tendencia = proximo_mes["yhat"].mean() - df_forecast["y"].mean()
        if tendencia > 0:
            st.info("📈 Tendência de aumento na vacinação.")
        else:
            st.warning("📉 Tendência de redução na vacinação.")

        st.divider()

    df_pretty = previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10).rename(columns={
        "ds": "Data",
        "yhat": "Vacinas previstas (média)",
        "yhat_lower": "Intervalo inferior",
        "yhat_upper": "Intervalo superior"
    })

    # formatar números
    df_pretty["Vacinas previstas (média)"] = df_pretty["Vacinas previstas (média)"].round(0).astype(int)
    df_pretty["Intervalo inferior"] = df_pretty["Intervalo inferior"].round(0).astype(int)
    df_pretty["Intervalo superior"] = df_pretty["Intervalo superior"].round(0).astype(int)

    st.dataframe(df_pretty, use_container_width=True)

    # --- Cálculo total previsto para o próximo mês ---
    proximo_mes = previsao.tail(30)
    estimativa_total = int(proximo_mes["yhat"].sum())

    st.success(f"💉 Estimativa de vacinas necessárias nos próximos 30 dias: **{estimativa_total:,} doses**")

    # --- Tendência geral ---
    tendencia = proximo_mes["yhat"].mean() - df_forecast["y"].mean()
    if tendencia > 0:
        st.info("📈 A tendência indica **aumento** na aplicação diária de vacinas.")
    else:
        st.warning("📉 A tendência indica **redução** na aplicação diária de vacinas.")

