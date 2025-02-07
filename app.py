import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message
import statsmodels as sm
import seaborn as sns
import os
import sys
from io import BytesIO
from sklearn.linear_model import LinearRegression

# Configuração da página
st.set_page_config(page_title="BasketIA 🏀", page_icon="chart_with_upwards_trend")

st.title("BasketIA 🏀")

# Configuração do sidebar
about = st.sidebar.expander("🧠 About")
sections = [r"""
Encontre e compare jogadores, através da combinação entre estatísticas e todo o poder da Inteligência artificial.
Faça análises jogadores, recebendo insights. A database dessa versão possui todos os jogadores brasileiros que atuaram nas principais ligas da Europa, EUA (HS, Universitário e NBA), Brasil e principais ligas da AL.
As possibilidades são infinitas." 
    """]
for section in sections:
    about.write(section)

# Configuração da temperatura no sidebar
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.5

with st.sidebar.expander("🛠️Tools", expanded=False):
    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.01,
    )
    st.session_state["temperature"] = temperature

# Configuração do modelo Claude
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]

def load_csv_data():
    try:
        # Caminho direto para o arquivo na pasta files
        file_path = os.path.join('files', 'Jogadores Brasileiros FULL 23-24 Season - Página1-2.csv')
            
        if not os.path.exists(file_path):
            st.error(f"Erro: Arquivo não encontrado em {file_path}")
            print(f"Tentando acessar arquivo em: {file_path}")  # Para debugging
            return None, None, None, None, None
            
        # Carrega o arquivo
        df = pd.read_csv(file_path)
        
        # Mostrar informações básicas sobre o dataset
        st.sidebar.markdown("### Informações do Dataset")
        st.sidebar.write(f"Total de registros: {len(df)}")
        st.sidebar.write(f"Colunas: {len(df.columns)}")
        
        return df, df.head(), df.isnull().sum(), df.shape, df.columns
        
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        print(f"Erro detalhado: {e}")  # Para debugging
        return None, None, None, None, None

def generate_code(prompt, data_type, missing, shape):
    prompt_template = PromptTemplate(
        input_variables=['prompt','data_type', 'shape', 'missing'],
        template="""You are a basketball data analyst who understands portuguese. You will answer based only on the data that is on Basketball Data is loaded as 'df' is already loaded as 'df'
        column names and their types: {data_type}
        df.shape: {shape}
        missing values: {missing}
        Please provide short executeable python code, I knows python, include correct column names.
        query: {prompt}
        Answer: 
        """
    )
    
    llm = ChatAnthropic(
        api_key=anthropic_api_key,
        model="claude-3-sonnet-20240229",
        temperature=st.session_state["temperature"],
        max_tokens_to_sample=4096
    )
    
    about_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="about")
    chain = SequentialChain(chains=[about_chain], input_variables=["prompt","data_type", "shape", "missing"], output_variables=["about"])
    
    try:
        response = chain.run({'prompt': prompt, 'data_type': data_type, 'shape': shape, 'missing':missing})
        return response
    except Exception as e:
        st.error(f"Erro ao gerar código: {e}")
        return None

# Interface principal
if "messages" not in st.session_state or st.sidebar.button("Limpar histórico de conversa"):
    st.session_state["messages"] = [{"role": "assistant", "content": r"""Olá, através de prompts, compare jogadores, através da combinação entre estatísticas e todo o poder da Inteligência artificial.
Faça análises de times e jogadores, identificando insights para o seu time.
As possibilidades são infinitas."""}]
    st.session_state['history'] = []

# Carregar dados
df, data, missing, shape, columns = load_csv_data()

# Interface de chat
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message("assistant", avatar="https://raw.githubusercontent.com/aloainow/images/main/logo.png").write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Inicie aqui seu chat!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if df is not None:
        with st.chat_message("assistant", avatar="https://raw.githubusercontent.com/aloainow/images/main/logo.png"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            try:
                prompt_response = generate_code(prompt, missing, shape, columns)
                
                if prompt_response:
                    # Criar agente para análise
                    agent = create_pandas_dataframe_agent(
                        ChatAnthropic(
                            api_key=anthropic_api_key,
                            model="claude-3-sonnet-20240229",
                            temperature=st.session_state["temperature"]
                        ),
                        df,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                        number_of_head_rows=4
                    )
                    
                    # Executar análise
                    response = agent.run(prompt_response, callbacks=[st_cb])
                    
                    # Verificar se há gráfico para mostrar
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.set_size_inches(12, 6)
                        plt.tight_layout()
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        st.image(buf, caption=prompt, use_column_width=True)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)
                    
            except Exception as e:
                st.error("Problema na análise dos dados! Por favor, tente novamente com uma pergunta diferente.")
                st.stop()
    else:
        st.warning("Erro ao carregar os dados. Verifique se o arquivo está no local correto.")

# Esconder elementos do Streamlit
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
