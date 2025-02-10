# ================ PARTE 1 - IMPORTAÇÕES E CONFIGURAÇÕES ================
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="CBB_IA 🏀", page_icon="🏀", layout="wide")  

# Barra lateral com informações das colunas
with st.sidebar:
    st.image("Logo CBB png.png", width=150)
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
        - **Gênero**: Masculino/Feminino
        
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

# Função centralizada para seleção de gênero
def get_gender_selection(key_suffix):
    """Função centralizada para seleção de gênero"""
    return st.radio(
        "Selecione o Gênero",
        ["Masculino", "Feminino"],
        horizontal=True,
        key=f"gender_select_{key_suffix}"
    )

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
            'Gênero': str,
            'J': pd.Int64Dtype(),  # Usando Int64Dtype para permitir valores nulos
            'Mins': str,
            'MMIN': 'float64',
            'PTS': pd.Int64Dtype(),
            'MPTS': 'float64',
            'TREB': pd.Int64Dtype(),
            'MTREB': 'float64',
            '3PTSC': pd.Int64Dtype(),
            'ASS': pd.Int64Dtype(),
            'MASS': 'float64',
            'RB': pd.Int64Dtype(),
            'MRB': 'float64',
            'T': pd.Int64Dtype(),
            'MT': 'float64',
            'LLT': pd.Int64Dtype(),
            'LLC': pd.Int64Dtype(),
            'AT': pd.Int64Dtype(),
            'FR': pd.Int64Dtype(),
            'FP': pd.Int64Dtype(),
            'POP': pd.Int64Dtype(),
            'MPOP': pd.Int64Dtype(),
            'REBD': pd.Int64Dtype(),
            'REBO': pd.Int64Dtype(),
            'ERR': pd.Int64Dtype(),
            'MERR': 'float64'
        }

        files = [f for f in os.listdir('files') if f.endswith('.csv')]
        if not files:
            st.error("Nenhum arquivo CSV encontrado na pasta 'files'")
            return None
        
        selected_file = files[0]
        
        # Ler o CSV com os tipos de dados especificados e tratamento de valores ausentes
        df = pd.read_csv(
            os.path.join('files', selected_file),
            dtype=dtype_dict,
            na_values=['', 'NA', 'nan', 'NaN', '#N/A', '#N/D', 'NULL'],
            encoding='utf-8'
        )
        
        # Verificar se a coluna Gênero existe e está preenchida
        if 'Gênero' not in df.columns or df['Gênero'].isna().any():
            # Se não existir ou tiver valores nulos, inferir baseado na competição
            df['Gênero'] = df['Competição'].apply(
                lambda x: 'Feminino' if isinstance(x, str) and 'Fem' in x else (
                    'Masculino' if isinstance(x, str) and 'Masc' in x else 'Não Especificado'
                )
            )
        
        # Remover linhas onde Nome está vazio
        df = df[df['Nome'].notna()]
        
        # Converter colunas float para ter 2 casas decimais
        float_columns = ['MMIN', 'MPTS', 'MTREB', 'MASS', 'MRB', 'MT', 'MERR']
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
        
        # Tratar valores ausentes
        for col in df.columns:
            if df[col].dtype in ['float64', 'Int64']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        st.write("Detalhes do erro para debug:", e)  # Adiciona mais detalhes do erro
        return None
# ================ PARTE 2 - FUNÇÕES DE PROCESSAMENTO ================

def process_text_query(df, query_text):
    """Processa consultas em texto livre"""
    query_text = query_text.lower()
    result = df.copy()
    
    try:
        # Dicionário de estatísticas ofensivas combinadas
        offensive_stats = ['PTS', 'MPTS', '3PTSC', 'ASS', 'MASS']
        defensive_stats = ['TREB', 'MTREB', 'RB', 'MRB', 'T', 'MT', 'REBD']
        
        # Processar diferentes tipos de consultas
        if "idade" in query_text or "anos" in query_text:
            # Converter data de nascimento para idade
            result['Idade'] = pd.to_datetime('today').year - pd.to_datetime(result['Data de Nascimento']).dt.year
            
            # Extrair número da consulta
            import re
            numbers = re.findall(r'\d+', query_text)
            if numbers:
                age = int(numbers[0])
                if "maior" in query_text or "acima" in query_text:
                    result = result[result['Idade'] > age]
                elif "menor" in query_text or "abaixo" in query_text:
                    result = result[result['Idade'] < age]
                else:
                    result = result[result['Idade'] == age]
        
        elif "ofensiv" in query_text:
            # Calcular score ofensivo combinado
            available_stats = [col for col in offensive_stats if col in result.columns]
            if available_stats:
                result['Score_Ofensivo'] = result[available_stats].mean(axis=1)
                result = result.sort_values('Score_Ofensivo', ascending=False)
        
        elif "defensiv" in query_text:
            # Calcular score defensivo combinado
            available_stats = [col for col in defensive_stats if col in result.columns]
            if available_stats:
                result['Score_Defensivo'] = result[available_stats].mean(axis=1)
                result = result.sort_values('Score_Defensivo', ascending=False)
        
        elif "top" in query_text or "melhores" in query_text:
            # Extrair número para top N
            import re
            numbers = re.findall(r'\d+', query_text)
            top_n = int(numbers[0]) if numbers else 5
            
            # Identificar estatística específica
            if "pont" in query_text:
                result = result.nlargest(top_n, 'MPTS')
            elif "rebote" in query_text:
                result = result.nlargest(top_n, 'MTREB')
            elif "assist" in query_text:
                result = result.nlargest(top_n, 'MASS')
            elif "block" in query_text or "toco" in query_text:
                result = result.nlargest(top_n, 'MT')
            else:
                # Score geral combinando principais estatísticas
                main_stats = ['MPTS', 'MTREB', 'MASS']
                available_stats = [col for col in main_stats if col in result.columns]
                if available_stats:
                    result['Score_Geral'] = result[available_stats].mean(axis=1)
                    result = result.nlargest(top_n, 'Score_Geral')
        
        return result.copy()
    
    except Exception as e:
        st.error(f"Erro ao processar consulta: {str(e)}")
        return df.copy()

def process_stats_query(df, gender, stat_types_selected=None, selected_stats=None):
    """Processa consulta de estatísticas com filtro de gênero e múltiplas estatísticas"""
    try:
        # Colunas base sempre mostradas
        base_columns = ['Nome', 'Equipe', 'Competição', 'Posição', 'Nacionalidade', 'Gênero']
        
        # Dicionário completo de tipos de estatísticas
        stat_types = {
            'pontos': ['PTS', 'MPTS', '3PTSC', 'LLT', 'LLC', 'AT'],
            'rebotes': ['TREB', 'MTREB', 'REBD', 'REBO', 'RB', 'MRB'],
            'assistencias': ['ASS', 'MASS'],
            'defesa': ['T', 'MT', 'REBD'],
            'geral': ['J', 'Mins', 'MMIN'],
            'erros': ['POP', 'MPOP', 'ERR', 'MERR', 'FP'],
            'eficiencia': ['AT', 'LLC', 'LLT', '3PTSC'],
            'produtividade': ['MPTS', 'MASS', 'MTREB', 'MT']
        }
        
        # Primeiro, filtrar por gênero
        result = df[df['Gênero'] == gender].copy()
        
        # Verificar colunas existentes
        available_columns = result.columns.tolist()
        base_columns = [col for col in base_columns if col in available_columns]
        
        if selected_stats:
            # Se há estatísticas específicas selecionadas, usar estas
            columns = base_columns + selected_stats
            result = result[columns].copy()
        elif stat_types_selected:
            # Se há tipos de estatísticas selecionados, pegar todas as estatísticas desses tipos
            stat_columns = []
            for stat_type in stat_types_selected:
                if stat_type in stat_types:
                    stat_columns.extend([col for col in stat_types[stat_type] if col in available_columns])
            columns = base_columns + list(dict.fromkeys(stat_columns))  # Remove duplicatas
            result = result[columns].copy()
        else:
            # Caso contrário, mostrar todas as estatísticas disponíveis
            all_stats = []
            for stats in stat_types.values():
                all_stats.extend([stat for stat in stats if stat in available_columns])
            columns = base_columns + list(dict.fromkeys(all_stats))
            result = result[columns].copy()
        
        # Ordenar por MPTS por padrão, se disponível
        if 'MPTS' in result.columns:
            result = result.sort_values(by='MPTS', ascending=False)
        
        return result
        
    except Exception as e:
        st.error(f"Erro ao processar estatísticas: {str(e)}")
        return pd.DataFrame()
# ================ PARTE 3 - FUNÇÕES DE VISUALIZAÇÃO E ANÁLISE ================

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

def text_query_section():
    """Seção de consultas por texto livre"""
    st.header("🔍 Consulta por Texto")
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Usar chave única para seleção de gênero nesta seção
    gender = get_gender_selection("text_query")
    
    # Filtrar dados por gênero antes de processar
    df = df[df['Gênero'] == gender]
    
    # Campo de texto para consulta
    query_text = st.text_input(
        "Digite sua consulta em texto livre",
        placeholder="Exemplo: 'top 5 jogadores em pontuação' ou 'jogadores com mais de 25 anos'"
    )
    
    # Exemplos de consultas
    with st.expander("📝 Exemplos de consultas"):
        st.markdown("""
        - Liste os top 5 jogadores em pontuação
        - Mostre jogadores com 20 anos de idade
        - Encontre os melhores jogadores em estatísticas ofensivas
        - Liste os armadores com melhor média de assistências
        - Mostre os jogadores brasileiros
        - Top 10 em rebotes
        - Melhores estatísticas defensivas
        """)
    
    if query_text:
        # Processar consulta
        result = process_text_query(df, query_text)
        
        if result is not None and not result.empty:
            total_results = len(result)
            
            # Limitar número de resultados mostrados
            num_results = st.slider(
                "Número de resultados a mostrar",
                min_value=1,
                max_value=total_results,
                value=min(50, total_results),
                key="text_query_slider"
            )
            
            result_displayed = result.head(num_results)
            
            # Mostrar resultados
            st.write(f"📊 Resultados encontrados: {total_results} jogadores")
            
            try:
                st.dataframe(
                    data=result_displayed,
                    use_container_width=True,
                    height=500
                )
            except Exception as e:
                st.error(f"Erro ao exibir dados: {str(e)}")
                st.write(result_displayed.to_html(index=False), unsafe_allow_html=True)
            
            # Opção de download
            st.download_button(
                label="📥 Download resultados (CSV)",
                data=result.to_csv(index=False, encoding='utf-8').encode('utf-8'),
                file_name='resultados_consulta.csv',
                mime='text/csv',
                key="text_query_download"
            )
# ================ PARTE 4 - SEÇÕES PRINCIPAIS E MAIN ================

def analytics_section():
    """Seção de análises e visualizações"""
    st.header("📊 Análise de Evolução")
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Usar chave única para seleção de gênero nesta seção
    gender = get_gender_selection("analytics")
    
    # Filtrar dados por gênero
    df = df[df['Gênero'] == gender]
    
    # Criar tabs para diferentes tipos de análise
    tab1, tab2 = st.tabs(["Evolução Individual", "Comparação entre Jogadores"])
    
    with tab1:
        st.subheader("Evolução Individual do Jogador")
        
        # Selecionar jogador
        player_names = sorted(df['Nome'].unique())
        selected_player = st.selectbox(
            "Selecione um jogador",
            player_names,
            key="player_select_evolution"
        )
        
        # Selecionar atributos para visualizar
        available_attributes = [
            'MPTS', 'MTREB', 'MASS', 'MRB', 'MT', 'MERR',
            'PTS', 'TREB', 'ASS', 'RB', 'T', '3PTSC'
        ]
        
        selected_attributes = st.multiselect(
            "Selecione os atributos para visualizar",
            available_attributes,
            default=['MPTS', 'MASS', 'MRB'],
            key="attributes_evolution"
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
            default=player_names[:2] if len(player_names) >= 2 else player_names,
            key="players_comparison"
        )
        
        # Selecionar atributo para comparação
        selected_attribute = st.selectbox(
            "Selecione o atributo para comparar",
            available_attributes,
            key="attribute_comparison"
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
    """Seção de consultas por categoria com filtro de gênero"""
    st.header("🔍 Consultas por Categoria")
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Usar chave única para seleção de gênero nesta seção
    gender = get_gender_selection("queries")
    
    # Criar duas colunas
    col1, col2 = st.columns([0.2, 0.8])
    
    with col1:
        # Seleção de categorias de estatísticas
        stat_categories = [
            "Pontuação", "Rebotes", "Assistências", "Defesa", 
            "Geral", "Erros", "Eficiência", "Produtividade"
        ]
        
        selected_categories = st.multiselect(
            "Selecione as Categorias de Estatísticas",
            stat_categories,
            default=["Pontuação"],
            key="stat_categories"
        )
    
    with col2:
        # Dicionário completo de todas as estatísticas disponíveis
        all_stats = {
            'Gerais': ['J', 'Mins', 'MMIN'],
            'Pontuação': ['PTS', 'MPTS', '3PTSC', 'AT'],
            'Lances Livres': ['LLT', 'LLC'],
            'Rebotes': ['TREB', 'MTREB', 'RB', 'MRB', 'REBD', 'REBO'],
            'Assistências': ['ASS', 'MASS'],
            'Defesa': ['T', 'MT'],
            'Erros': ['POP', 'MPOP', 'ERR', 'MERR', 'FP']
        }
        
        # Criar lista plana de todas as estatísticas
        all_stats_flat = []
        stats_descriptions = {
            'J': 'Jogos disputados',
            'Mins': 'Minutos totais',
            'MMIN': 'Média de minutos por jogo',
            'PTS': 'Pontos totais',
            'MPTS': 'Média de pontos por jogo',
            'TREB': 'Total de rebotes',
            'MTREB': 'Média de rebotes por jogo',
            '3PTSC': 'Arremessos de 3 pontos convertidos',
            'ASS': 'Total de assistências',
            'MASS': 'Média de assistências por jogo',
            'RB': 'Rebotes',
            'MRB': 'Média de rebotes',
            'T': 'Tocos (bloqueios)',
            'MT': 'Média de tocos',
            'LLT': 'Lances livres tentados',
            'LLC': 'Lances livres convertidos',
            'AT': 'Arremessos tentados',
            'REBD': 'Rebotes defensivos',
            'REBO': 'Rebotes ofensivos',
            'POP': 'Posse de bola perdida',
            'MPOP': 'Média de posse de bola perdida',
            'ERR': 'Erros',
            'MERR': 'Média de erros',
            'FP': 'Faltas cometidas'
        }
        
        for stats in all_stats.values():
            all_stats_flat.extend(stats)
        
        # Seleção de estatísticas específicas
        selected_stats = st.multiselect(
            "Selecione Estatísticas Específicas (opcional)",
            sorted(all_stats_flat),
            format_func=lambda x: f"{x} - {stats_descriptions.get(x, x)}",
            key="specific_stats"
        )
    
    # Converter categorias selecionadas para o formato do dicionário
    query_map = {
        "Pontuação": "pontos",
        "Rebotes": "rebotes",
        "Assistências": "assistencias",
        "Defesa": "defesa",
        "Geral": "geral",
        "Erros": "erros",
        "Eficiência": "eficiencia",
        "Produtividade": "produtividade"
    }
    
    selected_types = [query_map[cat] for cat in selected_categories if cat in query_map]
    
    # Processar consulta
    result = process_stats_query(df, gender, selected_types, selected_stats)
    
    if result is not None and not result.empty:
        total_players = len(result)
        
        # Adicionar slider para número de resultados
        num_results = st.slider(
            "Número de resultados a mostrar",
            min_value=1,
            max_value=total_players,
            value=min(50, total_players),
            key="query_results_slider"
        )
        
        # Mostrar resultados
        result_displayed = result.head(num_results)
        
        message = f"📊 Resultados encontrados: (Mostrando {len(result_displayed)} de {total_players} jogadores)"
        st.write(message)
        
        try:
            st.dataframe(
                data=result_displayed,
                use_container_width=True,
                height=500
            )
        except Exception as e:
            st.error(f"Erro ao exibir dados: {str(e)}")
            st.write(result_displayed.to_html(index=False), unsafe_allow_html=True)
        
        # Estatísticas resumidas
        st.write("### Resumo")
# Continuação da PARTE 4 - Final de queries_section() e main()

        # Continuação do queries_section()
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
            help="Clique para baixar a lista completa em formato CSV",
            key="query_download"
        )

def main():
    """Função principal da aplicação"""
    st.title("CBB_IA 🏀")
    
    # Criar tabs principais
    tab1, tab2, tab3 = st.tabs(["Consulta por Texto", "Consultas por Categoria", "Análise de Evolução"])
    
    with tab1:
        text_query_section()
    
    with tab2:
        queries_section()
    
    with tab3:
        analytics_section()

if __name__ == "__main__":
    main()

