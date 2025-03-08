# ================ PARTE 1 - IMPORTAÇÕES E CONFIGURAÇÕES ================
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Projeto RADAR_CBB 🏀", page_icon="🏀", layout="wide")  

# Barra lateral com informações das colunas
with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("Logo CBB png.png", width=150, use_column_width=True)
    st.header("Configurações e Ajuda 📊")    
    # Expandable para mostrar estatísticas disponíveis
    with st.expander("📈 Estatísticas Disponíveis"):
        st.markdown("""
        ### Informações do Jogador
        - **NOME**: Nome do jogador
        - **DATA DE NASCIMENTO**: Data de nascimento do jogador
        - **ALTURA**: Altura do jogador
        - **NACIONALIDADE**: País de origem
        - **POSIÇÃO**: Posição em quadra
        - **TEMPORADA**: Temporada/Ano
        - **EQUIPE**: Time 
        - **LIGA**: Liga/Campeonato
        - **Gênero**: Masculino/Feminino
        
        ### Estatísticas por Jogo
        - **J**: Jogos disputados
        - **MIN**: Minutos totais
        - **MMIN**: Média de minutos por jogo
        - **PTS**: Pontos totais
        - **MPTS**: Média de pontos por jogo
        - **RT**: Total de rebotes
        - **MTREB**: Média de rebotes por jogo
        - **3FGP**: Percentual de arremessos de 3 pontos
        - **AS**: Total de assistências
        - **MASS**: Média de assistências por jogo
        
        ### Estatísticas Defensivas/Ofensivas
        - **RO**: Rebotes ofensivos
        - **RD**: Rebotes defensivos
        - **RT**: Rebotes totais
        - **MRB**: Média de rebotes
        - **BS**: Tocos (bloqueios)
        - **MT**: Média de tocos
        
        ### Percentuais de Arremessos
        - **2FGP**: Percentual de arremessos de 2 pontos
        - **3FGP**: Percentual de arremessos de 3 pontos
        - **FT**: Percentual de lances livres
        
        ### Outras Estatísticas
        - **PF**: Faltas cometidas
        - **ST**: Roubadas de bola
        - **TO**: Turnovers (erros)
        - **MERR**: Média de erros
        - **RNK**: Ranking (eficiência)
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
    """Carrega dados de múltiplos arquivos CSV da estrutura de diretórios"""
    try:
        # Definir os tipos de dados para as colunas na nova estrutura
        dtype_dict = {
            'NOME': str,
            'DATA DE NASCIMENTO': str,
            'ALTURA': str,
            'NACIONALIDADE': str,
            'POSIÇÃO': str,
            'TEMPORADA': str,
            'EQUIPE': str,
            'LIGA': str,
            'J': pd.Int64Dtype(),  # Usando Int64Dtype para permitir valores nulos
            'MIN': 'float64',
            'PTS': 'float64',
            '2FGP': str,
            '3FGP': str,
            'FT': str,
            'RO': 'float64',
            'RD': 'float64',
            'RT': 'float64',
            'AS': 'float64',
            'PF': 'float64',
            'BS': 'float64',
            'ST': 'float64',
            'TO': 'float64',
            'RNK': 'float64'
        }
        
        # Listas de todas as pastas e seus respectivos gêneros
        folders = [
            ('files/Atletas Masculinos', 'Masculino'),
            ('files/Atletas Femininos', 'Feminino')
        ]
        
        # DataFrame para armazenar todos os dados
        all_data = []
        
        # Iterar sobre cada pasta
        for folder_path, gender in folders:
            try:
                if not os.path.exists(folder_path):
                    st.warning(f"Pasta não encontrada: {folder_path}")
                    continue
                
                # Encontrar todos os arquivos CSV na pasta
                csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
                
                if not csv_files:
                    st.warning(f"Nenhum arquivo CSV encontrado em: {folder_path}")
                    continue
                
                # Processar cada arquivo CSV
                for file_name in csv_files:
                    try:
                        file_path = os.path.join(folder_path, file_name)
                        
                        # Extrair o nome do jogador do nome do arquivo (removendo a extensão .csv)
                        player_name = os.path.splitext(file_name)[0]
                        
                        # Ler o arquivo CSV
                        df = pd.read_csv(
                            file_path,
                            dtype=dtype_dict,
                            na_values=['', 'NA', 'nan', 'NaN', '#N/A', '#N/D', 'NULL'],
                            encoding='utf-8'
                        )
                        
                        # Adicionar coluna de gênero
                        df['Gênero'] = gender
                        
                        # Verificar o formato do arquivo e ajustar se necessário
                        if df.shape[1] > 0 and df.columns[0] == 'Unnamed: 0':
                            # Arquivo está no formato esperado, não é necessário modificar o nome do jogador
                            # Já que NOME deve estar presente como coluna
                            pass
                        else:
                            # Adicionar coluna nome do jogador caso não exista
                            if 'NOME' not in df.columns:
                                df['NOME'] = player_name
                        
                        # Adicionar ao conjunto de dados geral
                        all_data.append(df)
                    except Exception as e:
                        st.error(f"Erro ao processar arquivo {file_name}: {str(e)}")
                        continue
            except Exception as e:
                st.error(f"Erro ao processar pasta {folder_path}: {str(e)}")
                continue
        
        if not all_data:
            st.error("Nenhum dado válido encontrado nas pastas especificadas")
            return None
        
        # Concatenar todos os DataFrames
        df = pd.concat(all_data, ignore_index=True)
        
        # Limpar dados
        # Remover linhas onde Nome está vazio
        df = df[df['NOME'].notna()]
        
        # Converter colunas de estatísticas para numérico e arredondar
        numeric_cols = ['MIN', 'PTS', 'RO', 'RD', 'RT', 'AS', 'PF', 'BS', 'ST', 'TO', 'RNK']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
        
        # Calcular campos derivados (médias por jogo)
        if 'J' in df.columns and 'J' in df.columns and df['J'].notna().any():
            # Minutos por jogo
            if 'MIN' in df.columns:
                df['MMIN'] = (df['MIN'] / df['J']).round(1)
            
            # Pontos por jogo
            if 'PTS' in df.columns:
                df['MPTS'] = (df['PTS'] / df['J']).round(1)
            
            # Rebotes por jogo
            if 'RT' in df.columns:
                df['MTREB'] = (df['RT'] / df['J']).round(1)
            
            # Assistências por jogo
            if 'AS' in df.columns:
                df['MASS'] = (df['AS'] / df['J']).round(1)
            
            # Rebotes por jogo
            if 'RT' in df.columns:
                df['MRB'] = (df['RT'] / df['J']).round(1)
            
            # Tocos por jogo
            if 'BS' in df.columns:
                df['MT'] = (df['BS'] / df['J']).round(1)
            
            # Erros por jogo
            if 'TO' in df.columns:
                df['MERR'] = (df['TO'] / df['J']).round(1)
        
        # Tratar valores ausentes
        for col in df.columns:
            if df[col].dtype in ['float64', 'Int64']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {str(e)}")
        st.write("Detalhes do erro para debug:", e)
        return None

def get_birth_year_filter(df, key_suffix):
    """
    Cria filtro por ano de nascimento com opções flexíveis
    """
    # Converter coluna de data de nascimento para datetime com tratamento de erros
    df['DATA DE NASCIMENTO'] = pd.to_datetime(df['DATA DE NASCIMENTO'], errors='coerce')
    
    # Extrair anos únicos de nascimento, removendo valores nulos
    anos_nascimento = sorted(df['DATA DE NASCIMENTO'].dt.year.dropna().unique(), reverse=True)
    
    # Se não houver anos válidos, retornar o DataFrame original
    if not anos_nascimento:
        st.warning("Não foram encontradas datas de nascimento válidas no conjunto de dados.")
        return df
    
    # Criar opções de filtro
    opcoes_filtro = ["Todos os anos"]
    
    # Adicionar opções para anos específicos
    for ano in anos_nascimento:
        opcoes_filtro.append(f"Nascidos em {ano}")
    
    # Adicionar opções para ranges
    ano_min, ano_max = min(anos_nascimento), max(anos_nascimento)
    opcoes_filtro.append(f"Nascidos até {ano_min}")
    opcoes_filtro.append(f"Nascidos entre {ano_min} e {ano_max}")
    
    # Criar selectbox
    filtro_selecionado = st.selectbox(
        "Filtrar por Ano de Nascimento",
        opcoes_filtro,
        key=f"birth_year_filter_{key_suffix}"
    )
    
    # Aplicar filtro selecionado
    if filtro_selecionado == "Todos os anos":
        return df
    elif "Nascidos em" in filtro_selecionado:
        ano = int(filtro_selecionado.split()[-1])
        return df[df['DATA DE NASCIMENTO'].dt.year == ano]
    elif "Nascidos até" in filtro_selecionado:
        ano = int(filtro_selecionado.split()[-1])
        return df[df['DATA DE NASCIMENTO'].dt.year <= ano]
    elif "Nascidos entre" in filtro_selecionado:
        anos = [int(x) for x in filtro_selecionado.split()[-3::2]]
        return df[df['DATA DE NASCIMENTO'].dt.year.between(anos[0], anos[1])]
    
    return df

# ================ PARTE 2 - FUNÇÕES DE PROCESSAMENTO ================

def process_text_query(df, query_text):
    """Processa consultas em texto livre"""
    query_text = query_text.lower()
    result = df.copy()
    
    try:
        # Dicionário de estatísticas ofensivas combinadas
        offensive_stats = ['PTS', 'MPTS', '3FGP', 'FT']
        defensive_stats = ['RT', 'MTREB', 'RD', 'BS', 'MT', 'ST']
        
        # Processar diferentes tipos de consultas
        if "idade" in query_text or "anos" in query_text:
            # Converter data de nascimento para idade
            result['Idade'] = pd.to_datetime('today').year - pd.to_datetime(result['DATA DE NASCIMENTO'], errors='coerce').dt.year
            
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
                # Converter percentuais para valores numéricos
                for col in available_stats:
                    if col in ['3FGP', 'FT', '2FGP']:
                        if result[col].dtype == 'object':
                            result[col] = result[col].str.rstrip('%').astype('float') / 100
                
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
            elif "eficiencia" in query_text or "ranking" in query_text:
                result = result.nlargest(top_n, 'RNK')
            else:
                # Score geral combinando principais estatísticas
                main_stats = ['MPTS', 'MTREB', 'MASS']
                available_stats = [col for col in main_stats if col in result.columns]
                if available_stats:
                    result['Score_Geral'] = result[available_stats].mean(axis=1)
                    result = result.nlargest(top_n, 'Score_Geral')
        
        # Filtrar por nacionalidade se mencionada
        for country in ["brasil", "brasileiro", "brasileira", "brazilian"]:
            if country in query_text:
                result = result[result['NACIONALIDADE'].str.contains('BRA', case=False)]
        
        # Filtrar por posição se mencionada
        positions = {
            "armador": ["PG", "G"],
            "ala": ["SF", "SG", "F", "F/G"],
            "pivô": ["C", "PF/C", "PF", "F"]
        }
        
        for pos_name, pos_codes in positions.items():
            if pos_name in query_text:
                result = result[result['POSIÇÃO'].str.contains('|'.join(pos_codes), case=False, regex=True)]
        
        return result.copy()
    
    except Exception as e:
        st.error(f"Erro ao processar consulta: {str(e)}")
        return df.copy()

def process_stats_query(df, gender, stat_types_selected=None, selected_stats=None):
    """Processa consulta de estatísticas com filtro de gênero e múltiplas estatísticas"""
    try:
        # Colunas base sempre mostradas
        base_columns = ['NOME', 'EQUIPE', 'LIGA', 'POSIÇÃO', 'NACIONALIDADE', 'Gênero']
        
        # Dicionário completo de tipos de estatísticas
        stat_types = {
            'pontos': ['PTS', 'MPTS', '3FGP', 'FT', '2FGP'],
            'rebotes': ['RT', 'MTREB', 'RD', 'RO', 'MRB'],
            'assistencias': ['AS', 'MASS'],
            'defesa': ['BS', 'MT', 'RD', 'ST'],
            'geral': ['J', 'MIN', 'MMIN'],
            'erros': ['TO', 'MERR', 'PF'],
            'eficiencia': ['2FGP', '3FGP', 'FT', 'RNK'],
            'produtividade': ['MPTS', 'MASS', 'MTREB', 'MT']
        }
        
        # Primeiro, filtrar por gênero
        result = df[df['Gênero'] == gender].copy()
        
        # Verificar colunas existentes
        available_columns = result.columns.tolist()
        base_columns = [col for col in base_columns if col in available_columns]
        
        if selected_stats:
            # Se há estatísticas específicas selecionadas, usar estas
            columns = base_columns + [col for col in selected_stats if col in available_columns]
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
        player_data = df[df['NOME'] == player_name]
        
        if player_data.empty:
            st.error(f"Não foram encontrados dados para o jogador {player_name}")
            return None
        
        # Ordenar por temporada para visualização adequada
        player_data = player_data.sort_values(by='TEMPORADA')
        
        # Tratar percentuais para plotagem
        for attr in attributes:
            if attr in ['2FGP', '3FGP', 'FT'] and attr in player_data.columns:
                if player_data[attr].dtype == 'object':
                    player_data[attr] = player_data[attr].str.rstrip('%').astype('float')
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar uma linha para cada atributo
        for attr in attributes:
            if attr in player_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=player_data['TEMPORADA'],
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
            xaxis_title='Temporada',
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
        comparison_data = df[df['NOME'].isin(players)]
        
        if comparison_data.empty:
            st.error("Não foram encontrados dados para os jogadores selecionados")
            return None
        
        # Tratar percentuais para plotagem
        if attribute in ['2FGP', '3FGP', 'FT'] and attribute in comparison_data.columns:
            if comparison_data[attribute].dtype == 'object':
                comparison_data[attribute] = comparison_data[attribute].str.rstrip('%').astype('float')
        
        fig = px.bar(
            comparison_data,
            x='NOME',
            y=attribute,
            color='LIGA',
            barmode='group',
            title=f'Comparação de {attribute}',
            labels={
                'NOME': 'Jogador',
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
    
    # Criar coluna para filtros
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Seleção de gênero
        gender = get_gender_selection("analytics")
        # Filtrar por gênero
        df = df[df['Gênero'] == gender]
    
    with filter_col2:
        # Filtro por ano de nascimento
        df = get_birth_year_filter(df, "analytics")
    
    # Criar tabs para diferentes tipos de análise
    tab1, tab2 = st.tabs(["Evolução Individual", "Comparação entre Jogadores"])
    
    with tab1:
        st.subheader("Evolução Individual do Jogador")
        
        # Selecionar jogador
        player_names = sorted(df['NOME'].unique())
        selected_player = st.selectbox(
            "Selecione um jogador",
            player_names,
            key="player_select_evolution"
        )
        
        # Selecionar atributos para visualizar
        available_attributes = [
            'MPTS', 'MTREB', 'MASS', 'MRB', 'MT', 'MERR',
            'PTS', 'RT', 'AS', 'RD', 'RO', 'BS', 'ST', 'RNK'
        ]
        
        # Filtrar atributos disponíveis pelo que está presente nos dados
        available_attributes = [attr for attr in available_attributes if attr in df.columns]
        
        selected_attributes = st.multiselect(
            "Selecione os atributos para visualizar",
            available_attributes,
            default=available_attributes[:3] if len(available_attributes) >= 3 else available_attributes,
            key="attributes_evolution"
        )
        
        if selected_attributes:
            chart = create_evolution_chart(df, selected_player, selected_attributes)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Mostrar tabela com dados completos
            st.subheader("Dados Detalhados")
            player_data = df[df['NOME'] == selected_player]
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
            comparison_data = df[df['NOME'].isin(selected_players)]
            summary = comparison_data.groupby('NOME')[selected_attribute].agg(['mean', 'min', 'max'])
            summary.columns = ['Média', 'Mínimo', 'Máximo']
            st.dataframe(summary.round(2), use_container_width=True)

def queries_section():
    """Seção de consultas por categoria com filtro de gênero"""
    st.header("🔍 Consultas por Categoria")
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Criar coluna para filtros
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Seleção de gênero
        gender = get_gender_selection("queries")
        # Filtrar por gênero
        df = df[df['Gênero'] == gender]
    
    with filter_col2:
        # Filtro por ano de nascimento
        df = get_birth_year_filter(df, "queries")
    
    # Criar duas colunas para as categorias
    col1, col2 = st.columns([0.4, 0.6])
    
    with col1:
        st.subheader("Categorias Principais")
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
        st.subheader("Estatísticas Detalhadas")
        # Dicionário completo de todas as estatísticas disponíveis
        all_stats = {
            'Gerais': ['J', 'MIN', 'MMIN'],
            'Pontuação': ['PTS', 'MPTS', '2FGP', '3FGP'],
            'Rebotes': ['RT', 'MTREB', 'RO', 'RD', 'MRB'],
            'Assistências': ['AS', 'MASS'],
            'Defesa': ['BS', 'MT', 'ST'],
            'Eficiência': ['RNK', 'FT'],
            'Erros': ['TO', 'MERR', 'PF']
        }
        
        # Criar lista plana de todas as estatísticas
        all_stats_flat = []
        stats_descriptions = {
            'J': 'Jogos disputados',
            'MIN': 'Minutos totais',
            'MMIN': 'Média de minutos por jogo',
            'PTS': 'Pontos totais',
            'MPTS': 'Média de pontos por jogo',
            'RT': 'Total de rebotes',
            'MTREB': 'Média de rebotes por jogo',
            '2FGP': 'Percentual de arremessos de 2 pontos',
            '3FGP': 'Percentual de arremessos de 3 pontos',
            'AS': 'Total de assistências',
            'MASS': 'Média de assistências por jogo',
            'RO': 'Rebotes ofensivos',
            'RD': 'Rebotes defensivos',
            'MRB': 'Média de rebotes',
            'BS': 'Tocos (bloqueios)',
            'MT': 'Média de tocos',
            'ST': 'Roubadas de bola',
            'FT': 'Percentual de lances livres',
            'PF': 'Faltas cometidas',
            'TO': 'Turnovers',
            'MERR': 'Média de erros',
            'RNK': 'Ranking (eficiência)'
        }
        
        # Filtrar apenas as estatísticas disponíveis nos dados
        available_stats = []
        for category, stats in all_stats.items():
            available_in_category = [stat for stat in stats if stat in df.columns]
            if available_in_category:
                all_stats_flat.extend(available_in_category)
        
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Jogadores", total_players)
        
        with col2:
            competicoes = result['LIGA'].nunique()
            st.metric("Competições", competicoes)
        
        with col3:
            equipes = result['EQUIPE'].nunique()
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
    st.title("Projeto RADAR_CBB 🏀")
    
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
