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
        numeric_columns = ['Age', 'PPG', 'APG', '2P%', '3P%', 'EFF', 'MPG', 'RPG', 'SPG', 'BPG']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def show_column_info(df):
    """Mostra informações sobre as colunas disponíveis"""
    st.write("### Colunas Disponíveis")
    
    # Agrupar colunas por categoria
    categories = {
        "Informações Básicas": ["Player Name", "Team Name", "League", "Nationality", "Country", "Age", "Height", "Pos", "TYPE"],
        "Estatísticas por Jogo": ["PPG", "APG", "RPG", "BPG", "SPG", "MPG"],
        "Percentuais": ["2P%", "3P%", "FT%"],
        "Outras Estatísticas": ["EFF", "ORB", "DRB", "PF", "TO"]
    }
    
    for category, cols in categories.items():
        st.write(f"**{category}:**")
        available_cols = [col for col in cols if col in df.columns]
        st.write(", ".join(available_cols))

def process_stats_query(df, age=None, stat_column=None, height=None, multiple_stats=False):
    """Processa consulta de estatísticas com múltiplos critérios"""
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
                df['2P%'] * 0.15 + 
                df['3P%'] * 0.15
            )
            columns = ['Player Name', 'Team Name', 'League', 'Age', 
                      'PPG', 'APG', '2P%', '3P%', 'Metrica_Ofensiva']
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
            handle_parsing_errors=True
        )
    except Exception as e:
        st.error(f"Erro ao criar agente: {str(e)}")
        return None

# Carregamento dos dados
df = load_data()

if df is not None:
    # Criar agente
    agent = create_agent(df)
    
    # Interface principal
    st.write("### 🔍 Faça sua consulta")
    
    # Mostrar exemplos de perguntas
    with st.expander("Ver exemplos de perguntas"):
        st.markdown("""
        - "Quais são os jogadores com mais pontos por jogo (PPG)?"
        - "Mostre os líderes em assistências (APG)"
        - "Quem tem o melhor aproveitamento nos arremessos de 3 pontos (3P%)?"
        - "Liste os jogadores com maior eficiência (EFF)"
        - "Mostre os top 10 jogadores de 24 anos"
        """)
    
    # Input do usuário
    user_input = st.text_input(
        "Digite sua pergunta...",
        help="Use as sugestões acima como exemplos"
    )
    
    # Processar consulta quando o usuário pressionar Enter
    if user_input:
        try:
            # Verificar palavras-chave nas estatísticas
            stat_keywords = {
                'PPG': ['ppg', 'pontos por jogo', 'pontos'],
                'APG': ['apg', 'assistências', 'assistencias'],
                'RPG': ['rpg', 'rebotes'],
                '2P%': ['2p%', 'field goal', 'arremessos de 2'],
                '3P%': ['3p%', 'three point', 'arremessos de 3'],
                'EFF': ['eff', 'eficiência', 'eficiencia']
            }
            
            stat_column = None
            idade = None
            prompt_lower = user_input.lower()

            # Verificar múltiplas estatísticas
            multiple_stats = all(stat in prompt_lower for stat in ['eff', 'ppg', 'apg'])
            
            # Verificar estatística específica se não for múltipla
            stat_column = None
            if not multiple_stats:
                for col, keywords in stat_keywords.items():
                    if any(keyword in prompt_lower for keyword in keywords):
                        stat_column = col
                        break

            # Verificar idade
            idade = None
            if "age" in prompt_lower or "anos" in prompt_lower:
                try:
                    idade = int(''.join(filter(str.isdigit, user_input)))
                except ValueError:
                    pass

            # Verificar altura
            altura = None
            if "height" in prompt_lower or "altura" in prompt_lower:
                try:
                    # Procurar por números com vírgula ou ponto
                    import re
                    height_match = re.search(r'\d+[.,]\d+', user_input)
                    if height_match:
                        altura = float(height_match.group().replace(',', '.'))
                except ValueError:
                    pass

            # Processar a consulta
            result = process_stats_query(
                df, 
                age=idade, 
                stat_column=stat_column,
                height=altura,
                multiple_stats=multiple_stats)
            
            if result is not None and not result.empty:
                message = ""
                if idade:
                    message = f"📊 Lista de todos os jogadores com {idade} anos:"
                elif stat_column:
                    message = f"📊 Lista de jogadores ordenados por {stat_column}:"
                else:
                    message = "📊 Resultados da consulta:"
                
                # Adicionar contagem de resultados
                total_players = len(result)
                message += f" (Total: {total_players} jogadores)"
                st.write(message)
                
                # Adicionar opção de paginação
                items_per_page = st.slider('Jogadores por página', min_value=10, max_value=50, value=20)
                total_pages = (total_players + items_per_page - 1) // items_per_page
                
                if total_pages > 1:
                    page = st.number_input('Página', min_value=1, max_value=total_pages, value=1)
                    start_idx = (page - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, total_players)
                    
                    st.write(f"Mostrando jogadores {start_idx + 1} a {end_idx} de {total_players}")
                    st.table(result.iloc[start_idx:end_idx])
                else:
                    st.table(result)
                
                # Adicionar opção de download
                st.download_button(
                    label="Download dados completos (CSV)",
                    data=result.to_csv(index=False).encode('utf-8'),
                    file_name='jogadores.csv',
                    mime='text/csv'
                )
            else:
                st.warning("Não encontrei resultados para sua consulta. Tente reformular a pergunta.")
                st.write("Sugestões:")
                st.write("1. Use os nomes exatos das estatísticas (PPG, APG, etc.)")
                st.write("2. Especifique a idade se quiser filtrar por idade")
                st.write("3. Consulte as estatísticas disponíveis na barra lateral")

        except Exception as e:
            st.error("Ocorreu um erro ao processar sua pergunta.")
            st.write("Dicas para melhorar sua consulta:")
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
</style>
""", unsafe_allow_html=True)
