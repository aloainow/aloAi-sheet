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
            # Removido o nlargest para mostrar todos os resultados
            result = df[columns].round(1)
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
            # Removido o nlargest para mostrar todos os resultados
            result = df[columns].round(1)
        
        # Ordenar resultados de forma decrescente pela estatística relevante
        if stat_column:
            result = result.sort_values(by=stat_column, ascending=False)
        else:
            result = result.sort_values(by='Metrica_Ofensiva', ascending=False)
            
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

def main():
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
            - "Mostre todos os jogadores de 22 anos"
            """)
        
        # Input do usuário
        user_input = st.text_input(
            "Digite sua pergunta...",
            help="Use as sugestões acima como exemplos"
        )
        
        # Adicionar slider para número de resultados
        num_results = st.slider("Número de resultados a mostrar", 10, 100, 50)
        
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
                if not multiple_stats:
                    for col, keywords in stat_keywords.items():
                        if any(keyword in prompt_lower for keyword in keywords):
                            stat_column = col
                            break

                # Verificar idade
                if "age" in prompt_lower or "anos" in prompt_lower:
                    try:
                        idade = int(''.join(filter(str.isdigit, user_input)))
                    except ValueError:
                        pass

                # Processar a consulta
                result = process_stats_query(
                    df, 
                    age=idade, 
                    stat_column=stat_column,
                    height=None,
                    multiple_stats=multiple_stats
                )
                
                if result is not None and not result.empty:
                    total_players = len(result)
                    
                    # Permitir que o usuário veja todos os resultados ou limite pelo slider
                    result_displayed = result.head(num_results) if num_results < len(result) else result
                    
                    message = f"📊 Resultados encontrados:"
                    message += f" (Mostrando {len(result_displayed)} de {total_players} jogadores)"
                    st.write(message)
                    
                    # Mostrar todos os dados com scroll
                    st.dataframe(
                        result_displayed,
                        use_container_width=True,
                        height=500  # Altura fixa com scroll
                    )
                    
                    # Adicionar estatísticas resumidas
                    st.write("### Resumo das estatísticas")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total de Jogadores", total_players)
                    
                    with col2:
                        leagues = result['League'].nunique()
                        st.metric("Ligas Diferentes", leagues)
                    
                    with col3:
                        teams = result['Team Name'].nunique()
                        st.metric("Times Diferentes", teams)
                    
                    # Adicionar opção de download
                    st.download_button(
                        label="📥 Download lista completa (CSV)",
                        data=result.to_csv(index=False).encode('utf-8'),
                        file_name='jogadores_completo.csv',
                        mime='text/csv',
                        help="Clique para baixar a lista completa em formato CSV"
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

if __name__ == "__main__":
    main()
