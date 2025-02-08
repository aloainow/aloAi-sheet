import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from streamlit_chat import message
import statsmodels as sm

# ─────────────────────────────────────────────
# Importações do LangChain
# ─────────────────────────────────────────────
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

# Importa o modelo da OpenAI
from langchain.chat_models import ChatOpenAI

# ─────────────────────────────────────────────
# Configuração da página e interface do Streamlit
# ─────────────────────────────────────────────
st.set_page_config(page_title="BasketIA 🏀", page_icon="chart_with_upwards_trend")
st.title("BasketIA 🏀")

# Sidebar About
about = st.sidebar.expander("🧠 About")
about.write(
    """Encontre e compare jogadores, através da combinação entre estatísticas e todo o poder da Inteligência artificial.
Faça análises de times e jogadores, identificando insights para o seu time.
As possibilidades são infinitas."""
)

# Configuração da temperatura no sidebar
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.5

with st.sidebar.expander("🛠️ Tools", expanded=False):
    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.01,
    )
    st.session_state["temperature"] = temperature

# Configuração do modelo OpenAI (use sua chave do OpenAI)
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ─────────────────────────────────────────────
# Função para carregar os dados CSV
# ─────────────────────────────────────────────
def load_csv_data():
    try:
        arquivos = os.listdir('files')
        arquivos_csv = [f for f in arquivos if f.endswith('.csv')]
        if not arquivos_csv:
            st.error("Nenhum arquivo CSV encontrado na pasta 'files'")
            return None, None, None, None, None
        if len(arquivos_csv) > 1:
            arquivo_selecionado = st.sidebar.selectbox("Selecione o arquivo para análise:", arquivos_csv)
        else:
            arquivo_selecionado = arquivos_csv[0]
        file_path = os.path.join('files', arquivo_selecionado)
        df = pd.read_csv(file_path, encoding='utf-8')
        st.sidebar.write("DataFrame carregado com sucesso")
        st.sidebar.write("Primeiras linhas:", df.head())
        st.sidebar.markdown("### Informações do Dataset")
        st.sidebar.write(f"Dataset atual: {arquivo_selecionado}")
        st.sidebar.write(f"Total de registros: {len(df)}")
        st.sidebar.write(f"Colunas: {len(df.columns)}")
        return df, df.head(), df.isnull().sum(), df.shape, df.columns
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None, None, None, None, None

# ─────────────────────────────────────────────
# Função para gerar o código com base no prompt
# ─────────────────────────────────────────────
def generate_code(prompt, columns, missing, shape):
    try:
        prompt_template = PromptTemplate(
            input_variables=['prompt', 'columns', 'shape', 'missing'],
            template="""You are a basketball data analyst who understands portuguese. You will answer based only on the data that is loaded in the variable 'df'.
Column names: {columns}
DataFrame shape: {shape}
Missing values: {missing}
Please provide short, executable Python code. I know Python, so include correct column names.
Query: {prompt}
Answer: 
"""
        )
        
        # Cria o modelo usando ChatOpenAI
        llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo",  # ou "gpt-4" se você tiver acesso
            temperature=st.session_state["temperature"]
        )
        
        about_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="about")
        chain = SequentialChain(
            chains=[about_chain],
            input_variables=["prompt", "columns", "shape", "missing"],
            output_variables=["about"]
        )
        
        response = chain.run({
            'prompt': prompt,
            'columns': columns,
            'shape': shape,
            'missing': missing
        })
        
        return response
    except Exception as e:
        st.error(f"Erro ao gerar código: {str(e)}")
        return None

# ─────────────────────────────────────────────
# Interface principal de chat
# ─────────────────────────────────────────────
if "messages" not in st.session_state or st.sidebar.button("Limpar histórico de conversa"):
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": """Olá, através de prompts, compare jogadores, através da combinação entre estatísticas e todo o poder da Inteligência artificial.
Faça análises de times e jogadores, identificando insights para o seu time.
As possibilidades são infinitas."""
    }]
    st.session_state['history'] = []

df, data, missing, shape, columns = load_csv_data()

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
                prompt_response = generate_code(prompt, columns, missing, shape)
                if prompt_response:
                    # Cria o agente usando o modelo ChatOpenAI
                    agent = create_pandas_dataframe_agent(
                        ChatOpenAI(
                            api_key=openai_api_key,
                            model_name="gpt-3.5-turbo",
                            temperature=st.session_state["temperature"]
                        ),
                        df,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        handle_parsing_errors=True,
                        number_of_head_rows=4,
                        allow_dangerous_code=True
                    )
                    
                    response = agent.run(prompt_response, callbacks=[st_cb])
                    
                    # Se houver gráfico, exibe-o
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
                st.error(f"Problema na análise dos dados: {str(e)}")
                st.stop()
    else:
        st.warning("Erro ao carregar os dados. Verifique se existem arquivos CSV na pasta 'files'.")

# Esconde elementos padrão do Streamlit
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
