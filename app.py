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

def create_agent(df, openai_api_key, temperature=0.5):
    try:
        llm = ChatOpenAI(
            temperature=temperature,
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo"
        )

        # Prompt que força o uso correto do python_repl_ast
        prompt_prefix = """Você é um assistente que analisa dados usando Python. Importante:

1. Use APENAS o comando python_repl_ast para executar código
2. O código deve ser executável e completo
3. Para mostrar dados, use:
   st.write(df) ou st.table(df)
4. Sempre execute o código completo, não apenas partes

Exemplo de resposta correta:
python_repl_ast: |
    # Filtrar dados
    result_df = df[df['Age'] == 22]
    # Mostrar resultado
    st.write(result_df)

NÃO use outros comandos além do python_repl_ast."""

        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            prefix=prompt_prefix
        )
    except Exception as e:
        st.error(f"Erro ao criar agente: {str(e)}")
        return None

# Inicialização do chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Como posso ajudar com a análise dos dados?"}
    ]

# Exibição das mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Processamento principal
df = load_data()
if df is not None:
    if "OPENAI_API_KEY" in st.secrets:
        agent = create_agent(df, st.secrets["OPENAI_API_KEY"], temperature)
        
        if prompt := st.chat_input("Faça uma pergunta sobre os dados..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                with st.chat_message("assistant"):
                    st_callback = StreamlitCallbackHandler(st.container())
                    response = agent.run(
                        f"Execute este código Python usando python_repl_ast: {prompt}",
                        callbacks=[st_callback]
                    )
                    
                    if plt.get_fignums():
                        for fig_num in plt.get_fignums():
                            fig = plt.figure(fig_num)
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Erro na análise: {str(e)}")
                st.error("Tente reformular sua pergunta.")
    else:
        st.error("Chave da API OpenAI não encontrada nos secrets.")
else:
    st.error("Coloque arquivos CSV na pasta 'files'.")

# Estilo
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
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
</style>
""", unsafe_allow_html=True)
