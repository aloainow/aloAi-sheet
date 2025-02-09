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

# Page configuration
st.set_page_config(page_title="BasketIA 🏀", page_icon="🏀", layout="wide")
st.title("BasketIA 🏀")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
    
    about = st.expander("🧠 About")
    about.write("""
    Find and analyze players using combined aesthetics for possible call-ups.
    You can query players by age, country, league, etc., and request graphs showing attribute evolution over seasons.
    """)

def load_data():
    try:
        files = [f for f in os.listdir('files') if f.endswith('.csv')]
        if not files:
            st.error("No CSV files found in 'files' directory")
            return None
        
        selected_file = st.sidebar.selectbox("Select analysis file:", files) if len(files) > 1 else files[0]
        df = pd.read_csv(os.path.join('files', selected_file))
        
        with st.sidebar.expander("Dataset Info"):
            st.write(f"Current dataset: {selected_file}")
            st.write(f"Total records: {len(df)}")
            st.write(f"Columns: {len(df.columns)}")
            st.write("Sample data:", df.head())
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def is_analytical_query(query):
    """Check if the query requires data analysis."""
    analytical_keywords = [
        'show', 'analyze', 'find', 'compare', 'list', 'plot', 'graph', 'calculate',
        'stats', 'statistics', 'average', 'mean', 'players', 'team', 'league',
        'age', 'height', 'score', 'points', 'best', 'worst', 'top', 'bottom'
    ]
    return any(keyword in query.lower() for keyword in analytical_keywords)

def get_greeting_response(query):
    """Handle conversational queries."""
    greetings = {
        'olá': 'Olá! Como posso ajudar com a análise dos dados de basquete hoje? Você pode me perguntar sobre estatísticas dos jogadores, comparações entre times, ou solicitar gráficos de desempenho.',
        'oi': 'Oi! Estou aqui para ajudar com análises de basquete. Que tipo de informação você gostaria de ver?',
        'hello': 'Hello! How can I help you analyze basketball data today?',
        'hi': 'Hi! Ready to help you with basketball analytics. What would you like to know?'
    }
    
    query_lower = query.lower()
    for greeting, response in greetings.items():
        if greeting in query_lower:
            return response
    
    return "Como posso ajudar com sua análise de dados de basquete? Você pode perguntar sobre estatísticas específicas, comparar jogadores ou solicitar visualizações de dados."

def create_agent(df, openai_api_key, temperature=0.5):
    try:
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
            verbose=True
        )
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Como posso ajudar com a análise dos dados de basquete hoje?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load data and handle queries
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
                            f"Analyze this query and provide insights: {prompt}. If it involves visualization, create appropriate charts.",
                            callbacks=[st_callback]
                        )
                        
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
                st.error(f"Error during analysis: {str(e)}")
                st.error("Por favor, tente reformular sua pergunta ou selecione diferentes parâmetros de análise.")
    else:
        st.error("OpenAI API key not found in secrets.")
else:
    st.error("Please ensure CSV files are present in the 'files' directory.")

# Hide Streamlit default elements
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
