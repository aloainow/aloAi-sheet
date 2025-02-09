import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler

# Configuração da página
st.set_page_config(page_title="BasketIA 🏀", page_icon="🏀", layout="wide")
st.title("BasketIA 🏀")

# Barra lateral
with st.sidebar:
    st.header("Configurações")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)

def load_data():
    """Carrega dados do CSV"""
    try:
        files = [f for f in os.listdir('files') if f.endswith('.csv')]
        if not files:
            st.error("Nenhum arquivo CSV encontrado na pasta 'files'")
            return None
        
        selected_file = files[0]
        df = pd.read_csv(os.path.join('files', selected_file))
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def create_agent(df):
    """Cria o agente do LangChain"""
    try:
        llm = ChatOpenAI(
            temperature=0.0,
            api_key=st.secrets["OPENAI_API_KEY"],
            model_name="gpt-3.5-turbo"
        )

        # Prompt personalizado para cálculos ofensivos
        prefix = """Você é um assistente que analisa estatísticas de basquete. Para análises ofensivas:

1. Use este código base:
python_repl_ast
# Calcular métrica ofensiva
df['Metrica_Ofensiva'] = df['PPG'] * 0.4 + df['APG'] * 0.3 + df['FG%'] * 0.3

# Selecionar top 10
result = df.nlargest(10, 'Metrica_Ofensiva')[['Player Name', 'Team Name', 'League', 'PPG', 'APG', 'FG%', 'Metrica_Ofensiva']]

# Formatar e mostrar
result = result.round(1)
st.table(result)

2. Sempre use python_repl_ast para executar código
3. Sempre mostre resultados com st.table()
4. Não adicione explicações, apenas execute o código"""

        return create_pandas_dataframe_agent(
            llm,
            df,
            prefix=prefix,
            verbose=True
        )
    except Exception as e:
        st.error(f"Erro ao criar agente: {str(e)}")
        return None

# Carregamento dos dados
df = load_data()

if df is not None:
    # Criar agente
    agent = create_agent(df)
    
    # Interface do chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá! Como posso ajudar com a análise dos dados?"}
        ]

    # Mostrar histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre os dados..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processar resposta
        with st.chat_message("assistant"):
            try:
                if "top" in prompt.lower() and ("ofensiv" in prompt.lower() or "estatistica" in prompt.lower()):
                    # Código direto para top 10 ofensivo
                    response = agent.run("""python_repl_ast
# Calcular métrica ofensiva
df['Metrica_Ofensiva'] = df['PPG'] * 0.4 + df['APG'] * 0.3 + df['FG%'] * 0.3

# Selecionar top 10
result = df.nlargest(10, 'Metrica_Ofensiva')[['Player Name', 'Team Name', 'League', 'PPG', 'APG', 'FG%', 'Metrica_Ofensiva']]

# Formatar e mostrar
result = result.round(1)
st.table(result)""")
                else:
                    response = agent.run(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                if isinstance(response, str):
                    st.markdown(response)
            except Exception as e:
                st.error(f"Erro na análise: {str(e)}")
                st.error("Tente reformular sua pergunta.")
else:
    st.error("Por favor, coloque arquivos CSV na pasta 'files'.")

# Estilo personalizado
st.markdown("""
<style>
    .stTable {
        width: 100%;
        margin: 1rem 0;
    }
    .stTable th {
        background-color: #f0f2f6;
        font-weight: bold;
        text-align: center;
        padding: 0.5rem;
    }
    .stTable td {
        text-align: right;
        padding: 0.5rem;
    }
    .stTable td:first-child {
        text-align: left;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
