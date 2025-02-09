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
    """Encontre e analise jogadores utilizando as estéticas combinadas para possíveis convocações.
Você pode consultar jogadores por idade, país, liga, etc., e também solicitar gráficos que mostrem a evolução de atributos ao longo das temporadas."""
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

# Configuração do modelo OpenAI (utilize sua chave do OpenAI em st.secrets)
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ─────────────────────────────────────────────
# Definição do prompt para o agente (versão estrita)
# ─────────────────────────────────────────────
prompt_template = PromptTemplate(
    input_variables=['prompt', 'columns', 'shape', 'missing'],
    template="""You are a basketball data analyst with expertise in Python, pandas, matplotlib, and seaborn.
You have a DataFrame named "df" that contains basketball player data with the following columns:
'Player Name', 'Team Name', 'League', 'Nationality', 'Country', 'Age', 'Height', 'Pos', 'GP', 'EFF', 'MPG', 'PPG', 'RPG', 'ORB', 'DRB', 'APG', 'BPG', 'SPG', 'PF', 'FTA', 'FTM', 'FT%', '2PA', '2PM', '2P%', '3PA', '3PM', '3P%', 'TO', and 'TYPE'.

Additionally, the DataFrame already includes columns for offensive and defensive metrics.
Your task is to combine these metrics into a new column "Combined Metric" by computing:
    Combined Metric = (Offensive Metric + Defensive Metric) / 2

Based solely on the data in "df" and given a user's query, generate a short, executable Python code snippet that performs the following tasks:
1. Filter the DataFrame according to the query criteria (for example, age, country, league, etc.).
2. Create a new column "Combined Metric" as described.
3. Sort the filtered results based on "Combined Metric" in descending order.
4. Select the top players as specified in the query (the user may request a specific number of players, e.g., 5 players or only 1).
5. If the query requests a graphical visualization (for example, "show a line chart of the evolution of PPG over the seasons"), generate an appropriate plot using matplotlib or seaborn.
6. IMPORTANT: Your output must contain ONLY the Python code enclosed in triple backticks (```), with no additional text, commentary, instructions, or tool references.


User Query: {prompt}
Columns: {columns}
DataFrame shape: {shape}
Missing values: {missing}

Answer:
"""
)

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
        # Cria o modelo usando ChatOpenAI com o modelo desejado (por exemplo, "gpt-4o")
        llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-4o",  # Substitua "gpt-4o" pelo nome do modelo desejado
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
        "content": """Olá, através de prompts, encontre e analise jogadores utilizando as estéticas combinadas para possíveis convocações.
Você pode solicitar filtros específicos (como idade, país, liga) e também pedir para visualizar gráficos de evolução de atributos ao longo das temporadas."""
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
                    # Cria o agente usando o modelo ChatOpenAI para executar o código gerado
                    agent = create_pandas_dataframe_agent(
                        ChatOpenAI(
                            api_key=openai_api_key,
                            model_name="gpt-3.5-turbo",  # Ajuste conforme necessário para execução
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

# ─────────────────────────────────────────────
# Estilos para esconder elementos padrão do Streamlit
# ─────────────────────────────────────────────
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
