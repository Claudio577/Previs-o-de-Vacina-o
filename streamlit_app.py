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
