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

from langchain.agents import create_pandas_dataframe_agent, AgentExecutor
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

# Load data function
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

# Custom prompt template
AGENT_PROMPT = """You are a basketball data analyst expert. Analyze the DataFrame 'df' with these columns:
{columns}

For each query:
1. Create a Combined Metric = (Offensive Metric + Defensive Metric) / 2
2. Filter data based on query criteria
3. Sort by Combined Metric
4. Return top results as specified
5. Create visualizations if requested

Current query: {query}

Respond with executable Python code only. Use matplotlib or seaborn for visualizations.
"""

# Agent creation function with better error handling
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

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you analyze basketball data today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load data and create agent
df = load_data()
if df is not None:
    if "OPENAI_API_KEY" in st.secrets:
        agent = create_agent(df, st.secrets["OPENAI_API_KEY"], temperature)
        
        if prompt := st.chat_input("Ask about basketball data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                with st.chat_message("assistant"):
                    st_callback = StreamlitCallbackHandler(st.container())
                    response = agent.run(
                        f"Based on this query: {prompt}\nAnalyze the data and provide insights. Include visualizations if relevant.",
                        callbacks=[st_callback]
                    )
                    
                    # Handle matplotlib figures if present
                    if plt.get_fignums():
                        for fig_num in plt.get_fignums():
                            fig = plt.figure(fig_num)
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.error("Please try rephrasing your question or selecting different analysis parameters.")
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
