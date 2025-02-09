import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from streamlit_chat import message

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Configuração da página
st.set_page_config(
    page_title="BasketIA 🏀",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("BasketIA 🏀")
st.markdown("### Análise Inteligente de Dados do Basquete")

# Configuração da barra lateral
with st.sidebar:
    st.header("Configurações")
    temperature = st.slider(
        "Temperatura da IA",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Controla a criatividade das respostas. Valores mais altos = mais criativo"
    )
    
    about = st.expander("🧠 Sobre")
    about.write("""
    BasketIA é uma ferramenta de análise de dados do basquete que utiliza IA.
    - Analise jogadores e times
    - Compare métricas e estatísticas
    - Visualize dados em gráficos
    - Descubra insights interessantes
    """)

def load_data():
    """Carrega e prepara os dados do CSV."""
    try:
        files = [f for f in os.listdir('files') if f.endswith('.csv')]
        if not files:
            st.error("Nenhum arquivo CSV encontrado na pasta 'files'")
            return None
        
        selected_file = st.sidebar.selectbox(
            "Selecione o arquivo para análise:",
            files
        ) if len(files) > 1 else files[0]
        
        df = pd.read_csv(os.path.join('files', selected_file))
        
        with st.sidebar.expander("📊 Informações do Dataset"):
            st.write(f"Dataset atual: {selected_file}")
            st.write(f"Total de registros: {len(df)}")
            st.write(f"Colunas disponíveis: {len(df.columns)}")
            st.write("Amostra dos dados:", df.head())
            
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def is_analytical_query(query):
    """Verifica se a query requer análise de dados."""
    analytical_keywords = [
        'mostre', 'mostra', 'analise', 'analisa', 'encontre', 'encontra',
        'compare', 'compara', 'liste', 'lista', 'plote', 'plota', 'gráfico',
        'grafico', 'calcule', 'calcula', 'estatísticas', 'estatisticas',
        'média', 'media', 'jogadores', 'time', 'liga', 'idade', 'altura',
        'pontos', 'melhor', 'pior', 'top', 'melhores', 'piores'
    ]
    return any(keyword in query.lower() for keyword in analytical_keywords)

def get_greeting_response(query):
    """Gerencia respostas para queries conversacionais."""
    greetings = {
        'olá': 'Olá! Como posso ajudar com a análise dos dados de basquete hoje?',
        'oi': 'Oi! Estou pronto para ajudar com suas análises. O que gostaria de saber?',
        'bom dia': 'Bom dia! Em que posso ajudar com os dados do basquete?',
        'boa tarde': 'Boa tarde! Como posso auxiliar em sua análise hoje?',
        'boa noite': 'Boa noite! Pronto para ajudar com suas análises de basquete.'
    }
    
    query_lower = query.lower()
    for greeting, response in greetings.items():
        if greeting in query_lower:
            return response
    
    return """Posso ajudar você a analisar dados de basquete. Por exemplo:
    - Mostrar os melhores jogadores por diferentes métricas
    - Comparar estatísticas entre jogadores ou times
    - Criar visualizações de dados
    - Analisar tendências e padrões
    
    Como posso ajudar?"""

def create_agent(df, openai_api_key, temperature=0.5):
    """Cria o agente de análise com prompt personalizado."""
    try:
        custom_prompt = """Você é um assistente especializado em análise de dados de basquete que gera código Python executável.

Instruções para análise:

1. Para qualquer consulta, você deve:
   - Gerar código Python executável
   - Calcular métricas relevantes
   - Exibir resultados em tabelas
   - Criar visualizações quando solicitado
   - Adicionar comentários explicativos

2. Calcule sempre estas métricas:
   - Métrica Ofensiva = PPG * 0.4 + APG * 0.3 + FG% * 0.3
   - Métrica Defensiva = RPG * 0.4 + BPG * 0.3 + SPG * 0.3
   - Métrica Combinada = (Métrica Ofensiva + Métrica Defensiva) / 2

3. Use estas ferramentas:
   - st.table() para exibir dados
   - matplotlib/seaborn para gráficos
   - round() para formatar números (2 decimais)

4. Formatação:
   - Tabelas com cabeçalhos em português
   - Gráficos com títulos e legendas em português
   - Números formatados com 2 casas decimais

5. Sempre inclua nas tabelas:
   - Nome, Time, Liga, Idade, Posição
   - Métricas calculadas relevantes
   - Ordenação apropriada dos resultados

Responda SEMPRE em português e execute código Python para todas as análises."""
        
        llm = ChatOpenAI(
            temperature=temperature,
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo"
        )
        
        return create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate",
            verbose=True,
            prefix=custom_prompt
        )
    except Exception as e:
        st.error(f"Erro ao criar agente: {str(e)}")
        return None

# Inicialização do estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Como posso ajudar com a análise dos dados de basquete hoje?"
        }
    ]

# Exibição das mensagens do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Carregamento dos dados e processamento das queries
df = load_data()
if df is not None:
    if "OPENAI_API_KEY" in st.secrets:
        agent = create_agent(df, st.secrets["OPENAI_API_KEY"], temperature)
        
        if prompt := st.chat_input("Faça uma pergunta sobre os dados de basquete..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                with st.chat_message("assistant"):
                    if is_analytical_query(prompt):
                        st_callback = StreamlitCallbackHandler(st.container())
                        response = agent.run(
                            f"Gere código Python para responder esta pergunta: {prompt}. "
                            "Inclua cálculo de métricas e apresentação em tabela.",
                            callbacks=[st_callback]
                        )
                        
                        # Tratamento de gráficos
                        if plt.get_fignums():
                            for fig_num in plt.get_fignums():
                                fig = plt.figure(fig_num)
                                st.pyplot(fig)
                                plt.close(fig)
                    else:
                        response = get_greeting_response(prompt)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)

            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")
                st.error("Por favor, tente reformular sua pergunta ou selecione diferentes parâmetros de análise.")
    else:
        st.error("Chave da API OpenAI não encontrada nos secrets.")
else:
    st.error("Certifique-se de que existem arquivos CSV na pasta 'files'.")

# Estilo personalizado para elementos do Streamlit
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
    }
    .stTable td {
        text-align: right;
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
