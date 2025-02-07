import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_pandas_dataframe_agent
import streamlit as st
from streamlit_chat import message
import statsmodels as sm
import seaborn as sns
import os
import sys
from io import StringIO, BytesIO
from sklearn.linear_model import LinearRegression

# Configuração da página
st.set_page_config(page_title="BasketIA 🏀", page_icon="chart_with_upwards_trend")
st.title("BasketIA 🏀")

# Sobre section
about = st.sidebar.expander("🧠 About")
sections = [r"""
Encontre e compare jogadores, através da combinação entre estatísticas e todo o poder da Inteligência artificial.
Faça análises jogadores, recebendo insights. A database dessa versão possui todos os jogadores brasileiros que atuaram nas principais ligas da Europa, EUA( HS, Universitário e NBA), Brasil e principais ligas da AL.
As possibilidades são infinitas." 
    """]
for section in sections:
    about.write(section)

# Inicialização da temperatura no session_state
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.5

# Controles de temperatura na sidebar
with st.sidebar.expander("🛠️Tools", expanded=False):
    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.01,
    )
    st.session_state["temperature"] = temperature

# Configuração do modelo Anthropic
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=st.session_state["temperature"],
    max_tokens=4096
)

# Resto do seu código permanece igual...

def generate_code(prompt, data_type, missing, shape):
    prompt_template = PromptTemplate(
        input_variables=['prompt','data_type', 'shape', 'missing'],
        template="You are a basketball data analyst who understands portuguese. You will answer based only on the data that is on Basketball Data is loaded as 'df' is already loaded as 'df'\
        column names and their types: {data_type}\n\
        df.shape: {shape}\
        missing values: {missing}\
        Please provide short executeable python code, I knows python, include correct column names.\
        query: {prompt}\
        Answer: \
        "
    )
    about_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="about")
    chain = SequentialChain(chains=[about_chain], input_variables=["prompt","data_type", "shape", "missing"], output_variables=["about"])
    response = chain.run({'prompt': prompt, 'data_type': data_type, 'shape': shape, 'missing':missing})
    return response
