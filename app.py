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
        
        st.sidebar.write(f"Dataset: {selected_file}")
        st.sidebar.write(f"Registros: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def format_table(df):
    """Formata tabela com todas as estatísticas"""
    try:
        # Selecionar todas as colunas relevantes
        stats_columns = [
            'Player Name', 'Team Name', 'League', 'Age', 'Height', 'Pos',
            'GP', 'MPG', 'PPG', 'RPG', 'APG', 'SPG', 'BPG',
            'FG%', '3P%', 'FT%', 'EFF'
        ]
        
        # Filtrar apenas colunas disponíveis
        available_columns = [col for col in stats_columns if col in df.columns]
        result_df = df[available_columns]
        
        # Formatar números para 1 casa decimal
        numeric_columns = result_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            result_df[col] = result_df[col].round(1)
        
        # Mostrar a tabela
        st.table(result_df)
    except Exception as e:
        st.error(f"Erro ao formatar tabela: {str(e)}")

def create_agent(df):
    """Cria o agente do LangChain"""
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

def process_query(agent, df, query):
    """Processa queries e retorna resultados"""
    try:
        # Processar top 10 ofensivo
        if "top" in query.lower() and "ofensiv" in query.lower():
            # Calcular métrica ofensiva
            df['Métrica_Ofensiva'] = (
                df['PPG'] * 0.4 +  # Pontos por jogo
                df['APG'] * 0.3 +  # Assistências por jogo
                df['FG%'] * 0.3    # Porcentagem de arremessos
            )
            
            # Ordenar e pegar top 10
            top_10 = df.nlargest(10, 'Métrica_Ofensiva')
            
            # Selecionar colunas para exibição
            display_columns = [
                'Player Name', 'Team Name', 'League',
                'PPG', 'APG', 'FG%', '3P%', 'FT%',
                'MPG', 'EFF', 'Métrica_Ofensiva'
            ]
            
            # Formatar e mostrar tabela
            result_df = top_10[display_columns].copy()
            
            # Arredondar valores numéricos
            numeric_cols = result_df.select_dtypes(include=['float64', 'int64']).columns
            result_df[numeric_cols] = result_df[numeric_cols].round(1)
            
            st.table(result_df)
            return "Top 10 jogadores por métricas ofensivas."
            
        # Processar queries sobre idade
        elif "jogadores" in query.lower() and "anos" in query.lower():
            age = [int(s) for s in query.split() if s.isdigit()][0]
            filtered_df = df[df['Age'] == age]
            
            if not filtered_df.empty:
                format_table(filtered_df)
                return f"Mostrando jogadores com {age} anos e suas estatísticas."
            else:
                return f"Não encontrei jogadores com {age} anos."
        
        # Processar outras queries usando o agente
        else:
            return agent.run(query)
    except Exception as e:
        st.error(f"Erro ao processar query: {str(e)}")
        return "Desculpe, não consegui processar sua solicitação."

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
            response = process_query(agent, df, prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            if isinstance(response, str):
                st.markdown(response)

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
    .metric-highlight {
        font-weight: bold;
        color: #1f77b4;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
