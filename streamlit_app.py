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
