import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="BasketIA 🏀", page_icon="🏀", layout="wide")

# Barra lateral com informações das colunas
with st.sidebar:
    st.header("Configurações e Ajuda 📊")
    
    # Expandable para mostrar estatísticas disponíveis
    with st.expander("📈 Estatísticas Disponíveis"):
        st.markdown("""
        ### Informações do Jogador
        - **Nome**: Nome do jogador
        - **ID**: Identificação
        - **Data de Nascimento**: Data de nascimento do jogador
        - **Altura**: Altura do jogador
        - **Nacionalidade**: País de origem
        - **Posição**: Posição em quadra
        - **Competição**: Liga/Campeonato
        - **Equipe**: Time atual
        
        ### Estatísticas por Jogo
        - **J**: Jogos disputados
        - **Mins**: Minutos totais
        - **MMIN**: Média de minutos por jogo
        - **PTS**: Pontos totais
        - **MPTS**: Média de pontos por jogo
        - **TREB**: Total de rebotes
        - **MTREB**: Média de rebotes por jogo
        - **3PTSC**: Arremessos de 3 pontos convertidos
        - **ASS**: Total de assistências
        - **MASS**: Média de assistências por jogo
        
        ### Estatísticas Defensivas/Ofensivas
        - **RB**: Rebotes
        - **MRB**: Média de rebotes
        - **T**: Tocos (bloqueios)
        - **MT**: Média de tocos
        - **REBD**: Rebotes defensivos
        - **REBO**: Rebotes ofensivos
        
        ### Outras Estatísticas
        - **LLT**: Lances livres tentados
        - **LLC**: Lances livres convertidos
        - **AT**: Arremessos tentados
        - **FR**: Faltas recebidas
        - **FP**: Faltas cometidas
        - **POP**: Posse de bola perdida
        - **MPOP**: Média de posse de bola perdida
        - **ERR**: Erros
        - **MERR**: Média de erros
        """)

def load_data():
    """Carrega dados do CSV com tratamento específico para a Planilha Piloto"""
    try:
        # Definir os tipos de dados para cada coluna
        dtype_dict = {
            'Nome': str,
            'ID': str,
            'Data de Nascimento': str,
            'Altura': str,
            'Nacionalidade': str,
            'Posição': str,
            'Competição': str,
            'Equipe': str,
            'J': 'Int64',  # Usando Int64 para suportar valores nulos
            'Mins': str,
            'MMIN': 'float64',
            'PTS': 'Int64',
            'MPTS': 'float64',
            'TREB': 'Int64',
            'MTREB': 'float64',
            '3PTSC': 'Int64',
            'ASS': 'Int64',
            'MASS': 'float64',
            'RB': 'Int64',
            'MRB': 'float64',
            'T': 'Int64',
            'MT': 'float64',
            'LLT': 'Int64',
            'LLC': 'Int64',
            'AT': 'Int64',
            'FR': 'Int64',
            'FP': 'Int64',
            'POP': 'Int64',
            'MPOP': 'Int64',
            'REBD': 'Int64',
            'REBO': 'Int64',
            'ERR': 'Int64',
            'MERR': 'float64'
        }

        files = [f for f in os.listdir('files') if f.endswith('.csv')]
        if not files:
            st.error("Nenhum arquivo CSV encontrado na pasta 'files'")
            return None
        
        selected_file = files[0]
        
        # Ler o CSV com os tipos de dados especificados
        df = pd.read_csv(
            os.path.join('files', selected_file),
            dtype=dtype_dict,
            na_values=['', 'NA', 'nan', 'NaN'],  # Valores a serem tratados como NA
            encoding='utf-8'
        )
        
        # Remover linhas onde Nome está vazio
        df = df[df['Nome'].notna()]
        
        # Converter colunas float para ter 2 casas decimais
        float_columns = ['MMIN', 'MPTS', 'MTREB', 'MASS', 'MRB', 'MT', 'MERR']
        for col in float_columns:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        # Substituir NaN por None para melhor visualização
        df = df.replace({pd.NA: None, pd.NaT: None, np.nan: None})
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def process_stats_query(df, stat_type=None):
    """Processa consulta de estatísticas adaptada para a Planilha Piloto"""
    try:
        # Colunas base sempre mostradas
        base_columns = ['Nome', 'Equipe', 'Competição', 'Posição', 'Nacionalidade']
        
        # Dicionário de tipos de estatísticas adaptado
        stat_types = {
            'pontos': ['PTS', 'MPTS', '3PTSC', 'LLT', 'LLC', 'AT'],
            'rebotes': ['TREB', 'MTREB', 'REBD', 'REBO', 'RB', 'MRB'],
            'assistencias': ['ASS', 'MASS'],
            'defesa': ['T', 'MT', 'REBD'],
            'geral': ['J', 'Mins', 'MMIN'],
            'erros': ['POP', 'MPOP', 'ERR', 'MERR', 'FP']
        }
        
        # Verificar colunas existentes
        available_columns = df.columns.tolist()
        base_columns = [col for col in base_columns if col in available_columns]
        
        if stat_type and stat_type in stat_types:
            stat_columns = [col for col in stat_types[stat_type] if col in available_columns]
            columns = base_columns + stat_columns
            result = df[columns].copy()
            
            # Ordenar baseado na estatística principal
            if stat_columns:
                main_stat = stat_columns[0]
                result = result.sort_values(by=main_stat, ascending=False)
        else:
            # Todas as estatísticas
            stat_columns = []
            for stats in stat_types.values():
                stat_columns.extend([col for col in stats if col in available_columns])
            
            columns = base_columns + list(dict.fromkeys(stat_columns))  # Remove duplicatas
            result = df[columns].copy()
            
            # Ordenar por MPTS
            if 'MPTS' in result.columns:
                result = result.sort_values(by='MPTS', ascending=False)
        
        return result
        
    except Exception as e:
        st.error(f"Erro ao processar estatísticas: {str(e)}")
        return pd.DataFrame()

def create_evolution_chart(df, player_name, attributes):
    """Cria gráfico de evolução dos atributos selecionados para um jogador"""
    try:
        # Filtrar dados do jogador
        player_data = df[df['Nome'] == player_name]
        
        if player_data.empty:
            st.error(f"Não foram encontrados dados para o jogador {player_name}")
            return None
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar uma linha para cada atributo
        for attr in attributes:
            if attr in player_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=player_data['Competição'],
                        y=player_data[attr],
                        name=attr,
                        mode='lines+markers',
                        hovertemplate=
                        "<b>%{x}</b><br>" +
                        f"{attr}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    )
                )
        
        # Atualizar layout
        fig.update_layout(
            title=f'Evolução de {player_name}',
            xaxis_title='Competição',
            yaxis_title='Valor',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Erro ao criar gráfico de evolução: {str(e)}")
        return None

def create_comparison_chart(df, players, attribute):
    """Cria gráfico de comparação de um atributo entre diferentes jogadores"""
    try:
        # Filtrar dados dos jogadores selecionados
        comparison_data = df[df['Nome'].isin(players)]
        
        if comparison_data.empty:
            st.error("Não foram encontrados dados para os jogadores selecionados")
            return None
        
        fig = px.bar(
            comparison_data,
            x='Nome',
            y=attribute,
            color='Competição',
            barmode='group',
            title=f'Comparação de {attribute}',
            labels={
                'Nome': 'Jogador',
                attribute: 'Valor'
            },
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Erro ao criar gráfico de comparação: {str(e)}")
        return None

def analytics_section():
    """Seção de análises e visualizações"""
    st.header("📊 Análise de Evolução")
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Criar tabs para diferentes tipos de análise
    tab1, tab2 = st.tabs(["Evolução Individual", "Comparação entre Jogadores"])
    
    with tab1:
        st.subheader("Evolução Individual do Jogador")
        
        # Selecionar jogador
        player_names = sorted(df['Nome'].unique())
        selected_player = st.selectbox(
            "Selecione um jogador",
            player_names
        )
        
        # Selecionar atributos para visualizar
        available_attributes = [
            'MPTS', 'MTREB', 'MASS', 'MRB', 'MT', 'MERR',
            'PTS', 'TREB', 'ASS', 'RB', 'T', '3PTSC'
        ]
        
        selected_attributes = st.multiselect(
            "Selecione os atributos para visualizar",
            available_attributes,
            default=['MPTS', 'MASS', 'MRB']
        )
        
        if selected_attributes:
            chart = create_evolution_chart(df, selected_player, selected_attributes)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Mostrar tabela com dados completos
            st.subheader("Dados Detalhados")
            player_data = df[df['Nome'] == selected_player]
            st.dataframe(player_data, use_container_width=True)
    
    with tab2:
        st.subheader("Comparação entre Jogadores")
        
        # Selecionar múltiplos jogadores
        selected_players = st.multiselect(
            "Selecione os jogadores para comparar",
            player_names,
            default=player_names[:2] if len(player_names) >= 2 else player_names
        )
        
        # Selecionar atributo para comparação
        selected_attribute = st.selectbox(
            "Selecione o atributo para comparar",
            available_attributes
        )
        
        if selected_players and selected_attribute:
            chart = create_comparison_chart(df, selected_players, selected_attribute)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Mostrar estatísticas resumidas
            st.subheader("Estatísticas Resumidas")
            comparison_data = df[df['Nome'].isin(selected_players)]
            summary = comparison_data.groupby('Nome')[selected_attribute].agg(['mean', 'min', 'max'])
            summary.columns = ['Média', 'Mínimo', 'Máximo']
            st.dataframe(summary.round(2), use_container_width=True)

def queries_section():
    """Seção de consultas com tratamento específico para a Planilha Piloto"""
    st.header("🔍 Consultas")
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
        
    # Opções de consulta
    query_type = st.selectbox(
        "Tipo de estatística",
        ["Todas", "Pontuação", "Rebotes", "Assistências", "Defesa", "Geral", "Erros"],
        format_func=lambda x: x.title()
    )
    
    query_map = {
        "Todas": None,
        "Pontuação": "pontos",
        "Rebotes": "rebotes",
        "Assistências": "assistencias",
        "Defesa": "defesa",
        "Geral": "geral",
        "Erros": "erros"
    }
    
    # Processar consulta
    result = process_stats_query(df, query_map[query_type])
    
    if result is not None and not result.empty:
        total_players = len(result)
        
        # Adicionar slider para número de resultados
        num_results = st.slider("Número de resultados a mostrar", 
                              min_value=1,
                              max_value=total_players,
                              value=min(50, total_players))
        
# Mostrar resultados
        result_displayed = result.head(num_results)
        
        message = f"📊 Resultados encontrados: (Mostrando {len(result_displayed)} de {total_players} jogadores)"
        st.write(message)
        
        try:
            # Tentar exibir o dataframe
            st.dataframe(
                data=result_displayed,
                use_container_width=True,
                height=500
            )
        except Exception as e:
            st.error(f"Erro ao exibir dados: {str(e)}")
            # Fallback para exibição HTML
            st.write(result_displayed.to_html(index=False), unsafe_allow_html=True)
        
        # Estatísticas resumidas
        st.write("### Resumo")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Jogadores", total_players)
        
        with col2:
            competicoes = result['Competição'].nunique()
            st.metric("Competições", competicoes)
        
        with col3:
            equipes = result['Equipe'].nunique()
            st.metric("Equipes", equipes)
        
        # Opção de download
        st.download_button(
            label="📥 Download lista completa (CSV)",
            data=result.to_csv(index=False, encoding='utf-8').encode('utf-8'),
            file_name='jogadores_estatisticas.csv',
            mime='text/csv',
            help="Clique para baixar a lista completa em formato CSV"
        )

def main():
    st.title("BasketIA 🏀")
    
    # Criar tabs principais
    tab1, tab2 = st.tabs(["Consultas", "Análise de Evolução"])
    
    with tab1:
        queries_section()
    
    with tab2:
        analytics_section()

if __name__ == "__main__":
    main()
