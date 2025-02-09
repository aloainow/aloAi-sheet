import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent import AgentExecutor

# Configuração da página
st.set_page_config(page_title="BasketIA 🏀", page_icon="🏀", layout="wide")
st.title("BasketIA 🏀")

# Barra lateral com informações das colunas
with st.sidebar:
    st.header("Configurações e Ajuda 📊")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    
    # Expandable para mostrar estatísticas disponíveis
    with st.expander("📈 Estatísticas Disponíveis"):
        st.markdown("""
        ### Informações do Jogador
        - **Player Name**: Nome do jogador
        - **Team Name**: Nome do time
        - **League**: Liga
        - **Nationality**: Nacionalidade
        - **Country**: País
        - **Age**: Idade
        - **Height**: Altura
        - **Pos**: Posição
        - **GP**: Jogos disputados
        - **TYPE**: Tipo de jogador
        
        ### Estatísticas Principais
        - **EFF**: Eficiência
        - **MPG**: Minutos por jogo
        - **PPG**: Pontos por jogo
        - **RPG**: Rebotes por jogo
        - **APG**: Assistências por jogo
        - **BPG**: Bloqueios por jogo
        - **SPG**: Roubos de bola por jogo
        
        ### Estatísticas Detalhadas
        - **ORB**: Rebotes ofensivos
        - **DRB**: Rebotes defensivos
        - **PF**: Faltas pessoais
        - **TO**: Turnovers (perdas de bola)
        
        ### Arremessos
        - **FTA**: Tentativas de lance livre
        - **FTM**: Lances livres convertidos
        - **FT%**: Percentual de acerto em lances livres
        - **2PA**: Tentativas de 2 pontos
        - **2PM**: Arremessos de 2 pontos convertidos
        - **2P%**: Percentual de acerto em arremessos de 2 pontos
        - **3PA**: Tentativas de 3 pontos
        - **3PM**: Arremessos de 3 pontos convertidos
        - **3P%**: Percentual de acerto em arremessos de 3 pontos
        """)
    
    # Expandable para exemplos de perguntas
    with st.expander("❓ Exemplos de Perguntas"):
        st.markdown("""
        1. "Mostre os top 10 jogadores de 24 anos"
        2. "Quais são os jogadores com maior PPG?"
        3. "Liste os jogadores com melhor aproveitamento de 3 pontos"
        4. "Quem são os líderes em assistências?"
        5. "Mostre os jogadores mais eficientes (EFF) da liga"
        """)

def load_data():
    """Carrega dados do CSV"""
    try:
        files = [f for f in os.listdir('files') if f.endswith('.csv')]
        if not files:
            st.error("Nenhum arquivo CSV encontrado na pasta 'files'")
            return None
        
        selected_file = files[0]
        df = pd.read_csv(os.path.join('files', selected_file))
        
        # Converter colunas numéricas
        numeric_columns = ['Age', 'PPG', 'APG', 'FG%', '3P%', 'EFF', 'MPG', 'RPG', 'SPG', 'BPG']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def show_column_info(df):
    """Mostra informações sobre as colunas disponíveis"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Estatísticas Disponíveis")
        stats_cols = [col for col in df.columns if any(x in col for x in ['PPG', 'APG', 'RPG', 'FG%', '3P%', 'EFF'])]
        for col in stats_cols:
            st.write(f"- {col}")
    
    with col2:
        st.write("### Informações de Jogador")
        info_cols = [col for col in df.columns if col not in stats_cols]
        for col in info_cols[:10]:  # Limitando para não ficar muito grande
            st.write(f"- {col}")

def process_stats_query(df, age=None, stat_column=None):
    """Processa consulta de estatísticas"""
    try:
        # Filtrar por idade se especificado
        if age is not None:
            df = df[df['Age'] == age].copy()
        else:
            df = df.copy()

        # Se uma estatística específica foi solicitada
        if stat_column and stat_column in df.columns:
            columns = ['Player Name', 'Team Name', 'League', 'Age', stat_column]
            result = df.nlargest(10, stat_column)[columns].round(1)
        else:
            # Calcular métrica ofensiva
            df['Metrica_Ofensiva'] = (
                df['PPG'] * 0.4 + 
                df['APG'] * 0.3 + 
                df['FG%'] * 0.15 + 
                df['3P%'] * 0.15
            )
            columns = ['Player Name', 'Team Name', 'League', 'Age', 
                      'PPG', 'APG', 'FG%', '3P%', 'Metrica_Ofensiva']
            result = df.nlargest(10, 'Metrica_Ofensiva')[columns].round(1)
        
        return result
    except Exception as e:
        st.error(f"Erro ao processar estatísticas: {str(e)}")
        return None

def create_agent(df):
    """Cria o agente do LangChain"""
    try:
        llm = ChatOpenAI(
            temperature=temperature,
            api_key=st.secrets["OPENAI_API_KEY"],
            model_name="gpt-3.5-turbo"
        )

        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2
        )
    except Exception as e:
        st.error(f"Erro ao criar agente: {str(e)}")
        return None

# Carregamento dos dados
df = load_data()

if df is not None:
    # Criar agente
    agent = create_agent(df)
    
    # Mostrar ajuda inicial
    if "show_help" not in st.session_state:
        st.session_state.show_help = True
        
    if st.session_state.show_help:
        with st.expander("ℹ️ Como usar o BasketIA", expanded=True):
            st.write("Bem-vindo ao BasketIA! Aqui você pode:")
            st.write("1. Fazer perguntas sobre estatísticas dos jogadores")
            st.write("2. Filtrar por idade ou métricas específicas")
            st.write("3. Ver rankings e comparações")
            show_column_info(df)
            if st.button("Entendi! Não mostrar novamente"):
                st.session_state.show_help = False
                st.experimental_rerun()
    
    # Interface do chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá! Como posso ajudar com a análise dos dados? Você pode perguntar sobre estatísticas dos jogadores por idade, rankings e mais. Use a barra lateral para ver as estatísticas disponíveis e exemplos de perguntas!"}
        ]

    # Mostrar histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário com sugestões
    # Criar lista de sugestões baseada nas colunas
    suggestions = [
        "Quais são os jogadores com mais pontos por jogo (PPG)?",
        "Mostre os líderes em assistências (APG)",
        "Quem tem o melhor aproveitamento nos arremessos de 3 pontos (3P%)?",
        "Liste os jogadores com maior eficiência (EFF)",
        "Quais jogadores têm mais roubos de bola (SPG)?",
        "Mostre os melhores em bloqueios (BPG)",
        "Quem tem o melhor aproveitamento em lances livres (FT%)?",
        "Mostre os jogadores que mais jogam minutos por jogo (MPG)",
        "Quais são os líderes em rebotes (RPG)?",
        "Liste os jogadores com mais rebotes ofensivos (ORB)",
        "Mostre os top 10 jogadores de uma idade específica",
        "Quem são os melhores jogadores por posição (Pos)?",
        "Liste os jogadores por nacionalidade"
    ]

    # Input do usuário
    if prompt := st.chat_input(
        "Faça uma pergunta sobre os dados...",
        help="Digite sua pergunta ou use as sugestões da barra lateral"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processar resposta
        with st.chat_message("assistant"):
            try:
                # Verificar se é uma consulta de estatística específica
                stat_keywords = {
                    'PPG': ['ppg', 'pontos por jogo', 'pontos'],
                    'APG': ['apg', 'assistências', 'assistencias'],
                    'RPG': ['rpg', 'rebotes'],
                    'FG%': ['fg%', 'field goal', 'arremessos de 2'],
                    '3P%': ['3p%', 'three point', 'arremessos de 3'],
                    'EFF': ['eff', 'eficiência', 'eficiencia']
                }
                
                stat_column = None
                for col, keywords in stat_keywords.items():
                    if any(keyword in prompt.lower() for keyword in keywords):
                        stat_column = col
                        break

                # Verificar se é uma consulta de idade
                idade = None
                if "anos" in prompt.lower():
                    idade = int(''.join(filter(str.isdigit, prompt)))

                # Processar a consulta
                result = process_stats_query(df, age=idade, stat_column=stat_column)
                if result is not None and not result.empty:
                    if idade:
                        st.write(f"Aqui estão os top 10 jogadores com {idade} anos:")
                    else:
                        st.write("Aqui estão os resultados:")
                    st.table(result)
                else:
                    if idade:
                        st.write(f"Não encontrei jogadores com {idade} anos.")
                    else:
                        response = agent.run(prompt)
                        st.markdown(response)

            except Exception as e:
                st.error("Ocorreu um erro ao processar sua pergunta.")
                st.write("Tente usar uma das sugestões da barra lateral ou reformular sua pergunta.")
                show_column_info(df)
else:
    st.error("Por favor, coloque arquivos CSV na pasta 'files'.")

# Estilo personalizado
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
        padding: 0.5rem;
    }
    .stTable td {
        text-align: right;
        padding: 0.5rem;
    }
    .stTable td:first-child {
        text-align: left;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Estilo para as sugestões */
    .suggestion-box {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        background-color: #f0f2f6;
        cursor: pointer;
    }
    .suggestion-box:hover {
        background-color: #e0e2e6;
    }
</style>
""", unsafe_allow_html=True)
