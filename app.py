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
        # Lista todos os arquivos na pasta files
        arquivos = os.listdir('files')
        
        # Filtra apenas os arquivos CSV
        arquivos_csv = [f for f in arquivos if f.endswith('.csv')]
        
        if not arquivos_csv:
            st.error("Nenhum arquivo CSV encontrado na pasta 'files'")
            return None, None, None, None, None
            
        # Se houver mais de um arquivo, permite a seleção
        if len(arquivos_csv) > 1:
            arquivo_selecionado = st.sidebar.selectbox(
                "Selecione o arquivo para análise:",
                arquivos_csv
            )
        else:
            arquivo_selecionado = arquivos_csv[0]
            
        # Carrega o arquivo selecionado
        file_path = os.path.join('files', arquivo_selecionado)
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Debug logs
        print(f"Arquivo carregado: {arquivo_selecionado}")
        print("Colunas:", df.columns.tolist())
        st.sidebar.write("DataFrame carregado com sucesso")
        st.sidebar.write("Primeiras linhas:", df.head())
        
        # Mostrar informações básicas sobre o dataset
        st.sidebar.markdown("### Informações do Dataset")
        st.sidebar.write(f"Dataset atual: {arquivo_selecionado}")
        st.sidebar.write(f"Total de registros: {len(df)}")
        st.sidebar.write(f"Colunas: {len(df.columns)}")
        
        return df, df.head(), df.isnull().sum(), df.shape, df.columns
        
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        print(f"Erro detalhado: {str(e)}")
        print("Diretório atual:", os.getcwd())
        print("Conteúdo do diretório files:", os.listdir('files'))
        return None, None, None, None, None

def generate_code(prompt, columns, missing, shape):
    try:
        prompt_template = PromptTemplate(
            input_variables=['prompt', 'columns', 'shape', 'missing'],
            template="""You are a basketball data analyst who understands portuguese. You will answer based only on the data that is on Basketball Data is loaded as 'df' is already loaded as 'df'
            column names: {columns}
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
        chain = SequentialChain(
            chains=[about_chain], 
            input_variables=["prompt", "columns", "shape", "missing"],
            output_variables=["about"]
        )
        
        # Debug log
        print("Gerando resposta para prompt:", prompt)
        
        response = chain.run({
            'prompt': prompt, 
            'columns': columns, 
            'shape': shape, 
            'missing': missing
        })
        
        # Debug log
        print("Resposta gerada:", response)
        
        return response
        
    except Exception as e:
        print(f"Erro detalhado na geração de código: {str(e)}")
        st.error(f"Erro ao gerar código: {str(e)}")
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
                # Debug log
                print("Processando prompt:", prompt)
                
                prompt_response = generate_code(prompt, columns, missing, shape)
                
                if prompt_response:
                    # Debug log
                    print("Prompt response gerado:", prompt_response)
                    
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
                    
                    # Debug log
                    print("Resposta do agente:", response)
                    
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
                print(f"Erro detalhado: {str(e)}")
                st.error(f"Problema na análise dos dados: {str(e)}")
                st.stop()
    else:
        st.warning("Erro ao carregar os dados. Verifique se existem arquivos CSV na pasta 'files'.")

# Esconder elementos do Streamlit
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
