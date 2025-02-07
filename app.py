import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
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

# Inicialização da temperatura no session_state
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

# Configuração do modelo Anthropic
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]

# Inicialização do modelo
llm = ChatAnthropic(
    api_key=anthropic_api_key,
    model="claude-3-sonnet-20240229",
    temperature=st.session_state["temperature"],
    max_tokens_to_sample=4096
)
