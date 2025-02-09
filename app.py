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

def format_table(df):
    """Helper function to format tables consistently"""
    try:
        # Selecionar colunas padrão se disponíveis
        default_columns = ['Player Name', 'Team Name', 'League', 'Age', 'Height', 'Pos']
        available_columns = [col for col in default_columns if col in df.columns]
        
        if available_columns:
            df = df[available_columns]
        
        # Mostrar a tabela
        st.table(df)
    except Exception as e:
        st.error(f"Erro ao formatar tabela: {str(e)}")

def create_agent(df):
    try:
        llm = ChatOpenAI(
            temperature=0.0,
            api_key=st.secrets["OPENAI_API_KEY"],
            model_name="gpt-3.5-turbo"
        )

        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True
        )
    except Exception as e:
        st.error(f"Erro ao criar agente: {str(e)}")
        return None

def process_query(agent, query):
    try:
        if "jogadores" in query.lower() and "anos" in query.lower():
            # Extrair idade da query
            age = [int(s) for s in query.split() if s.isdigit()][0]
            
            # Filtrar dados
            filtered_df = df[df['Age'] == age]
            
            # Mostrar resultados
            format_table(filtered_df)
            
            return f"Mostrando jogadores com {age} anos."
        else:
            return agent.run(query)
    except Exception as e:
        st.error(f"Erro ao processar query: {str(e)}")
        return "Desculpe, não consegui processar sua solicitação."

# Carregar dados
df = load_data()
if df is not None:
    agent = create_agent(df)
    
    # Interface do chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá! Como posso ajudar com a análise dos dados?"}
        ]

    # Mostrar mensagens
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
            response = process_query(agent, prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            if isinstance(response, str):
                st.markdown(response)
else:
    st.error("Coloque arquivos CSV na pasta 'files'.")
