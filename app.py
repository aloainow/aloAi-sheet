# ================ PARTE 1 - IMPORTAÇÕES E CONFIGURAÇÕES ================
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Desabilitar o botão de fork
st.set_page_config(
    page_title="RADAR CBB",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.instagram.com/cbbasketeball/',
        'Report a bug': 'https://www.instagram.com/cbbasketeball/',
        'About': 'App desenvolvido para CBB por Igor Gomes'
    }
)

# Configuração adicional para ocultar o botão de fork
# Adicione isso logo após o st.set_page_config()
st.markdown("""
<style>
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

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
                        
                        # Ler o arquivo CSV - adicionando "-" aos valores NA
                        df = pd.read_csv(
                            file_path,
                            dtype=dtype_dict,
                            na_values=['', 'NA', 'nan', 'NaN', '#N/A', '#N/D', 'NULL', '-'],
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
        
        # Não calcular mais médias - usaremos apenas as estatísticas originais dos arquivos
        
        # Tratar valores ausentes
        for col in df.columns:
            # Usar pandas is_numeric_dtype para verificar qualquer tipo numérico
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {str(e)}")
        st.write("Detalhes do erro para debug:", e)
        return None

def aggregate_player_data(df):
    """
    Agrega os dados de jogadores por nome, criando uma visão única por jogador
    com estatísticas compiladas da carreira usando médias ponderadas pelo número de jogos.
    
    Parameters:
    df (pandas.DataFrame): DataFrame com os dados dos jogadores
    
    Returns:
    pandas.DataFrame: DataFrame com dados agregados (um registro por jogador)
    """
    if df.empty:
        return df
    
    # Cria uma cópia para não modificar o dataframe original
    result = df.copy()
    
    # Primeiro, vamos adicionar um ID temporário para cada jogador único
    player_ids = {}
    player_id = 1
    result['player_id'] = 0
    
    for idx, row in result.iterrows():
        player_name = row['NOME']
        if player_name not in player_ids:
            player_ids[player_name] = player_id
            player_id += 1
        result.at[idx, 'player_id'] = player_ids[player_name]
    
    # Identificar diferentes tipos de colunas
    # Utilizar apenas as colunas disponíveis no arquivo original
    
    # Colunas que devem ser somadas
    sum_cols = ['J']  # Jogos é a única coluna que deve ser realmente somada
    
    # Colunas de médias nos arquivos CSV - precisamos fazer média ponderada por jogos
    avg_cols = ['MIN', 'PTS', 'RO', 'RD', 'RT', 'AS', 'PF', 'BS', 'ST', 'TO', 'RNK']
    
    # Removendo colunas de médias derivadas que não queremos mais
    # Colunas que são percentuais - também precisam de média ponderada
    pct_cols = ['2FGP', '3FGP', 'FT']
    
    # Colunas que devem pegar o valor mais recente
    last_cols = ['POSIÇÃO', 'NACIONALIDADE', 'EQUIPE', 'LIGA', 'ALTURA', 'Gênero', 'DATA DE NASCIMENTO']
    
    # Criar dicionário de agregações
    agg_dict = {}
    
    # Para cada coluna no DataFrame
    for col in result.columns:
        # Ignoramos o player_id na agregação
        if col == 'player_id':
            continue
            
        # Nome do jogador - usamos 'first' para manter um dos valores
        elif col == 'NOME':
            agg_dict[col] = 'first'
            
        # Colunas que devem ser somadas
        elif col in sum_cols and col in result.columns:
            agg_dict[col] = 'sum'
            
        # Colunas de informação - pegamos o valor mais recente (última temporada)
        elif col in last_cols and col in result.columns:
            agg_dict[col] = 'last'
            
        # Temporada - pegamos a mais recente
        elif col == 'TEMPORADA':
            agg_dict[col] = 'last'
            
        # Para o resto das colunas, usamos 'first' como placeholder
        # As médias ponderadas serão calculadas separadamente
        else:
            agg_dict[col] = 'first'
    
    # Agrupar por ID do jogador
    grouped = result.groupby('player_id').agg(agg_dict)
    
    # Calcular médias ponderadas para estatísticas e percentuais
    all_weighted_cols = avg_cols + pct_cols
    
    for col in all_weighted_cols:
        if col in result.columns:
            try:
                # Manipular percentagens de forma segura
                if col in pct_cols:
                    # Remover o símbolo % e converter para float
                    # Primeiro remover valores NA/None
                    temp_col = result[col].copy()
                    # Substituir valores NA e None por "0.0%"
                    temp_col = temp_col.fillna("0.0%")
                    # Agora é seguro remover o % e converter para float
                    result[col] = pd.to_numeric(temp_col.str.rstrip('%'), errors='coerce') / 100
                
                # Calcular médias ponderadas pelo número de jogos
                weighted_values = []
                
                for player_id in grouped.index:
                    player_data = result[result['player_id'] == player_id]
                    
                    if 'J' not in player_data.columns or player_data['J'].sum() == 0:
                        # Se não tiver jogos, usamos média simples
                        weighted_avg = player_data[col].mean() if col in player_data.columns else 0
                    else:
                        # Média ponderada pelo número de jogos
                        if col in player_data.columns:
                            # Excluir valores NaN ao calcular a média ponderada
                            mask = ~player_data[col].isna()
                            if mask.any():
                                weighted_avg = (player_data.loc[mask, col] * player_data.loc[mask, 'J']).sum() / player_data.loc[mask, 'J'].sum()
                            else:
                                weighted_avg = 0
                        else:
                            weighted_avg = 0
                    
                    weighted_values.append(weighted_avg)
                
                # Atualizar o valor no DataFrame agrupado
                grouped[col] = weighted_values
                
                # Converter de volta para o formato percentual se necessário
                if col in pct_cols:
                    grouped[col] = (grouped[col] * 100).round(1).astype(str) + '%'
            except Exception as e:
                # Em caso de erro, apenas manter o valor original
                print(f"Erro ao processar coluna {col}: {e}")
                # Usar o primeiro valor disponível 
                grouped[col] = result.groupby('player_id')[col].first()
    
    # Não adicionamos mais as colunas de médias derivadas nem a coluna de Temporadas
    
    # Resetamos o índice para ter um DataFrame normal
    grouped = grouped.reset_index()
    
    # Removemos a coluna player_id pois não precisamos mais dela
    if 'player_id' in grouped.columns:
        grouped = grouped.drop('player_id', axis=1)
    
    return grouped

def get_birth_year_filter(df, key_suffix):
    """
    Cria filtro por ano de nascimento com opções flexíveis
    com tratamento de erros melhorado para datas no formato YYYY-MM-DD
    """
    try:
        # Criar uma cópia para não alterar o DataFrame original
        df_filtrado = df.copy()
        
        # Verificar se a coluna DATA DE NASCIMENTO existe
        if 'DATA DE NASCIMENTO' not in df_filtrado.columns:
            st.warning("Coluna 'DATA DE NASCIMENTO' não encontrada no conjunto de dados.")
            return df
            
        # Substituir valores problemáticos antes da conversão
        df_filtrado['DATA DE NASCIMENTO'] = df_filtrado['DATA DE NASCIMENTO'].replace(['-', '', 'x-x'], pd.NA)
        
        # Converter coluna de data de nascimento para datetime com tratamento de erros
        df_filtrado['DATA DE NASCIMENTO'] = pd.to_datetime(df_filtrado['DATA DE NASCIMENTO'], errors='coerce')
        
        # Extrair anos únicos de nascimento, removendo valores nulos
        anos_validos = df_filtrado['DATA DE NASCIMENTO'].dt.year.dropna()
        
        # Se não houver anos válidos, retornar o DataFrame original
        if len(anos_validos) == 0:
            st.warning("Não foram encontradas datas de nascimento válidas no conjunto de dados.")
            return df
            
        anos_nascimento = sorted(anos_validos.unique(), reverse=True)
        
        # Criar opções de filtro
        opcoes_filtro = ["Todos os anos"]
        
        # Adicionar opções para anos específicos
        for ano in anos_nascimento:
            if pd.notna(ano):  # Verificar se o ano não é NaN
                opcoes_filtro.append(f"Nascidos em {int(ano)}")
        
        # Adicionar opções para ranges se tivermos anos válidos
        if len(anos_nascimento) > 0:
            ano_min, ano_max = int(min(anos_nascimento)), int(max(anos_nascimento))
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
            try:
                ano = int(filtro_selecionado.split()[-1])
                # Filtrando apenas registros com data válida e ano igual ao selecionado
                return df_filtrado[df_filtrado['DATA DE NASCIMENTO'].dt.year == ano]
            except (ValueError, IndexError, TypeError) as e:
                st.warning(f"Erro ao aplicar filtro '{filtro_selecionado}': {str(e)}. Mostrando todos os dados.")
                return df
        elif "Nascidos até" in filtro_selecionado:
            try:
                ano = int(filtro_selecionado.split()[-1])
                # Filtrando apenas registros com data válida e ano menor ou igual ao selecionado
                return df_filtrado[df_filtrado['DATA DE NASCIMENTO'].dt.year <= ano]
            except (ValueError, IndexError, TypeError) as e:
                st.warning(f"Erro ao aplicar filtro '{filtro_selecionado}': {str(e)}. Mostrando todos os dados.")
                return df
        elif "Nascidos entre" in filtro_selecionado:
            try:
                # Extrai os dois anos do formato "Nascidos entre X e Y"
                partes = filtro_selecionado.split()
                ano_inicio = int(partes[-3])
                ano_fim = int(partes[-1])
                # Filtrando apenas registros com data válida e ano entre os selecionados
                return df_filtrado[df_filtrado['DATA DE NASCIMENTO'].dt.year.between(ano_inicio, ano_fim)]
            except (ValueError, IndexError, TypeError) as e:
                st.warning(f"Erro ao aplicar filtro '{filtro_selecionado}': {str(e)}. Mostrando todos os dados.")
                return df
        
        return df
    except Exception as e:
        st.warning(f"Erro ao processar filtro por ano de nascimento: {str(e)}. Mostrando todos os dados.")
        return df

# ================ PARTE 2 - FUNÇÕES DE PROCESSAMENTO ================

def process_text_query_with_aggregation(df, query_text, aggregate=True):
    """Processa consultas em texto livre com opção de agregação por jogador"""
    query_text = query_text.lower()
    result = df.copy()
    
    try:
        # Verificar as colunas disponíveis no DataFrame
        available_columns = result.columns.tolist()
        
        # Dicionário de estatísticas ofensivas combinadas (apenas as disponíveis)
        offensive_stats = [col for col in ['PTS', 'MPTS', '3FGP', 'FT'] if col in available_columns]
        defensive_stats = [col for col in ['RT', 'MTREB', 'RD', 'BS', 'MT', 'ST'] if col in available_columns]
        
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
            if offensive_stats:
                # Converter percentuais para valores numéricos
                for col in offensive_stats:
                    if col in ['3FGP', 'FT', '2FGP']:
                        if result[col].dtype == 'object':
                            result[col] = result[col].str.rstrip('%').astype('float') / 100
                
                result['Score_Ofensivo'] = result[offensive_stats].mean(axis=1)
                result = result.sort_values('Score_Ofensivo', ascending=False)
        
        elif "defensiv" in query_text:
            # Calcular score defensivo combinado
            if defensive_stats:
                result['Score_Defensivo'] = result[defensive_stats].mean(axis=1)
                result = result.sort_values('Score_Defensivo', ascending=False)
        
        elif "top" in query_text or "melhores" in query_text:
            # Extrair número para top N
            import re
            numbers = re.findall(r'\d+', query_text)
            top_n = int(numbers[0]) if numbers else 5
            
            # Identificar estatística específica
            if "pont" in query_text and 'MPTS' in available_columns:
                result = result.sort_values('MPTS', ascending=False)
            elif "rebote" in query_text and 'MTREB' in available_columns:
                result = result.sort_values('MTREB', ascending=False)
            elif "assist" in query_text and 'MASS' in available_columns:
                result = result.sort_values('MASS', ascending=False)
            elif "block" in query_text or "toco" in query_text and 'MT' in available_columns:
                result = result.sort_values('MT', ascending=False)
            elif "eficiencia" in query_text or "ranking" in query_text and 'RNK' in available_columns:
                result = result.sort_values('RNK', ascending=False)
            else:
                # Score geral combinando principais estatísticas
                main_stats = [col for col in ['MPTS', 'MTREB', 'MASS'] if col in available_columns]
                if main_stats:
                    result['Score_Geral'] = result[main_stats].mean(axis=1)
                    result = result.sort_values('Score_Geral', ascending=False)
        
        # Filtrar por nacionalidade se mencionada
        for country in ["brasil", "brasileiro", "brasileira", "brazilian"]:
            if country in query_text and 'NACIONALIDADE' in available_columns:
                result = result[result['NACIONALIDADE'].str.contains('BRA', case=False)]
        
        # Filtrar por posição se mencionada
        if 'POSIÇÃO' in available_columns:
            positions = {
                "armador": ["PG", "G"],
                "ala": ["SF", "SG", "F", "F/G"],
                "pivô": ["C", "PF/C", "PF", "F"]
            }
            
            for pos_name, pos_codes in positions.items():
                if pos_name in query_text:
                    result = result[result['POSIÇÃO'].str.contains('|'.join(pos_codes), case=False, regex=True)]
        
        # Agregar dados por jogador se solicitado
        if aggregate:
            result = aggregate_player_data(result)
            
            # Limitar para top_n se for uma consulta de top
            if "top" in query_text or "melhores" in query_text:
                result = result.head(top_n)
        
        return result.copy()
    
    except Exception as e:
        st.error(f"Erro ao processar consulta: {str(e)}")
        return df.copy()

def process_stats_query_with_aggregation(df, gender, stat_types_selected=None, selected_stats=None, aggregate=True):
    """Processa consulta de estatísticas com filtro de gênero, múltiplas estatísticas e agregação por jogador"""
    try:
        # Verificar as colunas disponíveis no DataFrame
        available_columns = df.columns.tolist()
        
        # Colunas base sempre mostradas (apenas as disponíveis)
        # Adicionando 'DATA DE NASCIMENTO' à lista de colunas base
        base_columns = [col for col in ['NOME', 'DATA DE NASCIMENTO', 'EQUIPE', 'LIGA', 'POSIÇÃO', 'NACIONALIDADE', 'Gênero'] if col in available_columns]
        
        # Dicionário completo de tipos de estatísticas (apenas as disponíveis e sem médias derivadas)
        stat_types = {
            'pontos': [col for col in ['PTS', '3FGP', 'FT', '2FGP'] if col in available_columns],
            'rebotes': [col for col in ['RT', 'RD', 'RO'] if col in available_columns],
            'assistencias': [col for col in ['AS'] if col in available_columns],
            'defesa': [col for col in ['BS', 'RD', 'ST'] if col in available_columns],
            'geral': [col for col in ['J', 'MIN'] if col in available_columns],
            'erros': [col for col in ['TO', 'PF'] if col in available_columns],
            'eficiencia': [col for col in ['2FGP', '3FGP', 'FT', 'RNK'] if col in available_columns],
            'produtividade': [col for col in ['PTS', 'AS', 'RT', 'BS'] if col in available_columns]
        }
        
        # Remover tipos de estatísticas vazios
        stat_types = {k: v for k, v in stat_types.items() if v}
        
        # Primeiro, filtrar por gênero
        result = df[df['Gênero'] == gender].copy()
        
        # Pré-processamento para limpar valores vazios
        for col in result.columns:
            if col not in base_columns:  # Não precisamos converter colunas de texto como NOME, EQUIPE, etc.
                if pd.api.types.is_numeric_dtype(result[col]):
                    # Para colunas já em formato numérico, apenas substituir NaN por 0
                    result[col] = result[col].fillna(0)
                else:
                    # Para colunas em formato string, primeiro substituir strings vazias por NaN
                    result[col] = result[col].replace(['', '-', 'nan', 'NA', '#N/A', '#N/D', 'NULL'], pd.NA)
                    
                    # Para colunas que deveriam ser numéricas, converter para numérico com tratamento de erros
                    if col in ['PTS', 'MIN', 'RO', 'RD', 'RT', 'AS', 'PF', 'BS', 'ST', 'TO', 'RNK', 'J']:
                        result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
                    # Para colunas de percentuais (que contêm '%'), tratar especialmente
                    elif col in ['2FGP', '3FGP', 'FT']:
                        # Manter como string por enquanto, será tratado durante a agregação
                        pass
        
        if selected_stats:
            # Se há estatísticas específicas selecionadas, usar estas (se estiverem disponíveis)
            valid_selected_stats = [col for col in selected_stats if col in available_columns]
            columns = base_columns + valid_selected_stats
            if valid_selected_stats:  # Apenas filtrar se houver estatísticas válidas selecionadas
                result = result[columns].copy()
        elif stat_types_selected:
            # Se há tipos de estatísticas selecionados, pegar todas as estatísticas desses tipos
            stat_columns = []
            for stat_type in stat_types_selected:
                if stat_type in stat_types:
                    stat_columns.extend(stat_types[stat_type])
            columns = base_columns + list(dict.fromkeys(stat_columns))  # Remove duplicatas
            result = result[columns].copy()
        else:
            # Caso contrário, mostrar todas as estatísticas disponíveis
            all_stats = []
            for stats in stat_types.values():
                all_stats.extend(stats)
            columns = base_columns + list(dict.fromkeys(all_stats))
            result = result[columns].copy()
        
        # Agregar dados por jogador se solicitado
        if aggregate:
            result = aggregate_player_data(result)
        
        # Ordenar por PTS por padrão, se disponível
        if 'PTS' in result.columns:
            result = result.sort_values(by='PTS', ascending=False)
        
        return result
        
    except Exception as e:
        st.error(f"Erro ao processar estatísticas: {str(e)}")
        st.write("Detalhes do problema:", e)
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
        
        # Verificar quais atributos estão realmente disponíveis
        available_attributes = [attr for attr in attributes if attr in player_data.columns]
        
        if not available_attributes:
            st.error(f"Nenhum dos atributos selecionados está disponível para o jogador {player_name}")
            return None
            
        # Ordenar por temporada para visualização adequada
        player_data = player_data.sort_values(by='TEMPORADA')
        
        # Tratar percentuais para plotagem
        for attr in available_attributes:
            if attr in ['2FGP', '3FGP', 'FT'] and attr in player_data.columns:
                if player_data[attr].dtype == 'object':
                    player_data[attr] = player_data[attr].str.rstrip('%').astype('float')
        
        # Converter para valores numéricos e tratar valores nulos
        for attr in available_attributes:
            player_data[attr] = pd.to_numeric(player_data[attr], errors='coerce').fillna(0)
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar uma linha para cada atributo
        for attr in available_attributes:
            fig.add_trace(
                go.Scatter(
                    x=player_data['TEMPORADA'].astype(str),  # Converter para string para garantir serialização
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

def create_comparison_chart(df, players, attributes):
    """Cria gráfico de comparação de atributos entre diferentes jogadores"""
    try:
        # Se apenas um atributo for fornecido e não for uma lista, transforme-o em lista
        if not isinstance(attributes, list):
            attributes = [attributes]
            
        # Verificar se os atributos estão disponíveis
        for attr in attributes:
            if attr not in df.columns:
                st.error(f"O atributo '{attr}' não está disponível nos dados")
                return None
                
        # Filtrar dados dos jogadores selecionados
        comparison_data = df[df['NOME'].isin(players)]
        
        if comparison_data.empty:
            st.error("Não foram encontrados dados para os jogadores selecionados")
            return None
        
        # Tratar percentuais para plotagem
        percentage_cols = ['2FGP', '3FGP', 'FT']
        for attr in attributes:
            if attr in percentage_cols and attr in comparison_data.columns:
                if comparison_data[attr].dtype == 'object':
                    comparison_data[attr] = comparison_data[attr].str.rstrip('%').astype('float')
        
        # Se for apenas um jogador ou um atributo, usar gráfico de barras
        if len(players) <= 1 or len(attributes) == 1:
            main_attr = attributes[0]  # Usamos o primeiro atributo para o gráfico de barras
            fig = px.bar(
                comparison_data,
                x='NOME',
                y=main_attr,
                color='LIGA',
                barmode='group',
                title=f'Comparação de {main_attr}',
                labels={
                    'NOME': 'Jogador',
                    main_attr: 'Valor'
                },
                height=500
            )
            return fig
            
        # Usar radar chart para múltiplos atributos
        # Garantir que todos os atributos são numéricos
        for attr in attributes:
            comparison_data[attr] = pd.to_numeric(comparison_data[attr], errors='coerce').fillna(0)
        
        # Criar figura para o radar
        fig = go.Figure()
        
        # Calcular valores máximos para normalização
        max_values = {}
        for attr in attributes:
            max_val = comparison_data[attr].max()
            max_values[attr] = max_val if max_val > 0 else 1
        
        # Adicionar dados de cada jogador
        for player in players:
            player_data = comparison_data[comparison_data['NOME'] == player]
            if not player_data.empty:
                values = []
                for attr in attributes:
                    # Garantir que o valor é numérico
                    val = float(player_data[attr].iloc[0])
                    # Normalizar para escala 0-100
                    val_norm = (val / max_values[attr]) * 100
                    values.append(val_norm)
                
                # Adicionar ao gráfico
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=attributes,
                    fill='toself',
                    name=player
                ))
        
        # Atualizar layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=f'Comparação de Estatísticas',
            showlegend=True,
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Erro ao criar gráfico de comparação: {str(e)}")
        # Em caso de erro, tentar um gráfico de barras simples
        try:
            main_attr = attributes[0] if isinstance(attributes, list) else attributes
            fig = px.bar(
                comparison_data,
                x='NOME',
                y=main_attr,
                color='LIGA',
                barmode='group',
                title=f'Comparação de {main_attr}',
                labels={
                    'NOME': 'Jogador',
                    main_attr: 'Valor'
                },
                height=500
            )
            return fig
        except:
            return None
def text_query_section():
    """Seção de consultas por texto livre"""
    st.header("🔍 Consulta por Texto")
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Criar coluna para filtros
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Seleção de gênero
        gender = get_gender_selection("text_query")
        # Filtrar por gênero
        df = df[df['Gênero'] == gender]
    
    with filter_col2:
        # Seleção de país
        country = get_country_selection(df, "text_query")
        # Filtrar por país
        df = filter_by_country(df, country)
    
    with filter_col3:
        # Adicionar opção para agregar por jogador ou mostrar por temporada
        aggregation_option = st.radio(
            "Modo de exibição",
            ["Compilado da carreira", "Por temporada"],
            horizontal=True,
            key="text_query_aggregation"
        )
    
    # Determinar se deve agregar com base na opção selecionada
    aggregate = (aggregation_option == "Compilado da carreira")
    
    # Campo de texto para consulta
    query_text = st.text_input(
        "Digite sua consulta em texto livre",
        placeholder="Exemplo: 'top 5 jogadores em pontuação' ou 'jogadores com mais de 25 anos'",
        key="text_query_input"
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
        # Processar consulta com a nova função que suporta agregação
        result = process_text_query_with_aggregation(df, query_text, aggregate=aggregate)
        
        if result is not None and not result.empty:
            # Remover a coluna Temporadas se existir
            if 'Temporadas' in result.columns:
                result = result.drop('Temporadas', axis=1)
                
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
            st.write(f"📊 Resultados encontrados: {total_results} {'jogadores' if aggregate else 'registros'}")
            
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
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Seleção de gênero
        gender = get_gender_selection("analytics")
        # Filtrar por gênero
        df = df[df['Gênero'] == gender]
    
    with filter_col2:
        # Filtro por ano de nascimento
        df = get_birth_year_filter(df, "analytics")
    
    with filter_col3:
        # Seleção de país
        country = get_country_selection(df, "analytics")
        # Filtrar por país
        df = filter_by_country(df, country)
    
    # Criar tabs para diferentes tipos de análise
    tab1, tab2 = st.tabs(["Evolução Individual", "Comparação entre Jogadores"])
    
    with tab1:
        # ... código existente para tab1 ...
    
    with tab2:
        st.subheader("Comparação entre Jogadores")
        
        # Selecionar múltiplos jogadores
        player_names = sorted(df['NOME'].unique())
        selected_players = st.multiselect(
            "Selecione os jogadores para comparar",
            player_names,
            default=player_names[:2] if len(player_names) >= 2 else player_names,
            key="players_comparison"
        )
        
        # Obter atributos disponíveis para os jogadores selecionados
        if selected_players:
            players_data = df[df['NOME'].isin(selected_players)]
            available_comparison_attrs = []
            
            # Lista de atributos potenciais
            potential_attrs = ['PTS', 'RT', 'AS', 'RD', 'RO', 'BS', 'ST', 'RNK']
            
            # Verificar quais atributos existem e têm dados para todos os jogadores selecionados
            for attr in potential_attrs:
                if attr in players_data.columns and not players_data[attr].isnull().all():
                    available_comparison_attrs.append(attr)
            
            # Selecionar múltiplos atributos para comparação
            if available_comparison_attrs:
                selected_attributes = st.multiselect(
                    "Selecione os atributos para comparar",
                    available_comparison_attrs,
                    default=available_comparison_attrs[:5] if len(available_comparison_attrs) >= 5 else available_comparison_attrs,
                    key="attributes_comparison"
                )
                
                # NOVO CÓDIGO:
                if selected_attributes:
                    chart = create_comparison_chart(df, selected_players, selected_attributes)
                    if chart is not None:
                        try:
                            st.plotly_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erro ao renderizar o gráfico: {str(e)}")
                            st.write("Tente selecionar atributos diferentes ou outros jogadores.")
                
                    
                    # Mostrar estatísticas resumidas
                    st.subheader("Estatísticas Resumidas")
                    comparison_data = df[df['NOME'].isin(selected_players)]
                    summary_table = pd.DataFrame(index=selected_players)
                    
                    for attr in selected_attributes:
                        if attr in comparison_data.columns:
                            for player in selected_players:
                                player_data = comparison_data[comparison_data['NOME'] == player]
                                if not player_data.empty:
                                    summary_table.loc[player, attr] = player_data[attr].iloc[0]
                    
                    st.dataframe(summary_table.round(2), use_container_width=True)
                else:
                    st.info("Selecione ao menos um atributo para comparação.")
            else:
                st.warning("Não há atributos numéricos disponíveis para comparação entre os jogadores selecionados.")
        else:
            st.info("Selecione ao menos um jogador para comparação.")
def queries_section():
    """Seção de consultas por categoria com filtro de gênero, ano de nascimento e país"""
    st.header("🔍 Consultas por Categoria")
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Criar coluna para filtros
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Seleção de gênero
        gender = get_gender_selection("queries")
        # Filtrar por gênero
        df = df[df['Gênero'] == gender]
    
    with filter_col2:
        # Filtro por ano de nascimento
        df = get_birth_year_filter(df, "queries")
    
    with filter_col3:
        # Seleção de país
        country = get_country_selection(df, "queries")
        # Filtrar por país
        df = filter_by_country(df, country)
    
    # Adicionar opção para agregar por jogador ou mostrar por temporada
    aggregation_option = st.radio(
        "Modo de exibição",
        ["Compilado da carreira", "Por temporada"],
        horizontal=True,
        key="queries_aggregation"
    )
    
    # Determinar se deve agregar com base na opção selecionada
    aggregate = (aggregation_option == "Compilado da carreira")
    
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
        
        # Verificar colunas disponíveis
        available_columns = df.columns.tolist()
        
        # Dicionário completo de estatísticas disponíveis (sem médias derivadas)
        all_stats = {
            'Gerais': [col for col in ['J', 'MIN'] if col in available_columns],
            'Pontuação': [col for col in ['PTS', '2FGP', '3FGP'] if col in available_columns],
            'Rebotes': [col for col in ['RT', 'RO', 'RD'] if col in available_columns],
            'Assistências': [col for col in ['AS'] if col in available_columns],
            'Defesa': [col for col in ['BS', 'ST'] if col in available_columns],
            'Eficiência': [col for col in ['RNK', 'FT'] if col in available_columns],
            'Erros': [col for col in ['TO', 'PF'] if col in available_columns]
        }
        
        # Remover categorias vazias
        all_stats = {k: v for k, v in all_stats.items() if v}
        
        # Criar lista plana de todas as estatísticas disponíveis
        all_stats_flat = []
        for category_stats in all_stats.values():
            all_stats_flat.extend(category_stats)
        
        stats_descriptions = {
            'J': 'Jogos disputados',
            'MIN': 'Minutos totais',
            'PTS': 'Pontos totais',
            'RT': 'Total de rebotes',
            '2FGP': 'Percentual de arremessos de 2 pontos',
            '3FGP': 'Percentual de arremessos de 3 pontos',
            'AS': 'Total de assistências',
            'RO': 'Rebotes ofensivos',
            'RD': 'Rebotes defensivos',
            'BS': 'Tocos (bloqueios)',
            'ST': 'Roubadas de bola',
            'FT': 'Percentual de lances livres',
            'PF': 'Faltas cometidas',
            'TO': 'Turnovers (erros)',
            'RNK': 'Ranking (eficiência)'
        }
        
        # Seleção de estatísticas específicas
        if all_stats_flat:
            selected_stats = st.multiselect(
                "Selecione Estatísticas Específicas (opcional)",
                sorted(all_stats_flat),
                format_func=lambda x: f"{x} - {stats_descriptions.get(x, x)}",
                key="specific_stats"
            )
        else:
            st.warning("Não há estatísticas disponíveis para seleção")
            selected_stats = []
    
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
    
    # Processar consulta com a nova função que suporta agregação
    result = process_stats_query_with_aggregation(df, gender, selected_types, selected_stats, aggregate=aggregate)
    
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
        
        message = f"📊 Resultados encontrados: (Mostrando {len(result_displayed)} de {total_players} {'jogadores' if aggregate else 'registros'})"
        st.write(message)
        
        # Remover a coluna Temporadas dos resultados de exibição
        if 'Temporadas' in result.columns:
            result = result.drop('Temporadas', axis=1)
        
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
            st.metric("Total de Jogadores" if aggregate else "Total de Registros", total_players)
        
        if 'LIGA' in result.columns:
            with col2:
                competicoes = result['LIGA'].nunique()
                st.metric("Competições", competicoes)
        
        if 'EQUIPE' in result.columns:
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

# Adicione este dicionário para mapear os códigos de liga para países
PAIS_MAPPING = {
    'ARG': 'Argentina',
    'AUS': 'Austrália',
    'BEL': 'Bélgica',
    'BIH': 'Bósnia e Herzegovina',
    'BOL': 'Bolívia', 
    'BRA': 'Brasil',
    'BUL': 'Bulgária',
    'CHI': 'Chile',
    'CHN': 'China',
    'COL': 'Colômbia',
    'CRO': 'Croácia',
    'CZE': 'República Tcheca',
    'DOM': 'República Dominicana',
    'ESP': 'Espanha',
    'FRA': 'França',
    'GER': 'Alemanha',
    'GRE': 'Grécia',
    'HUN': 'Hungria',
    'IDN': 'Indonésia',
    'ISL': 'Islândia',
    'ISR': 'Israel',
    'ITA': 'Itália',
    'JPN': 'Japão',
    'KOS': 'Kosovo',
    'LAT': 'Letônia',
    'LTU': 'Lituânia',
    'MEX': 'México',
    'MKD': 'Macedônia do Norte',
    'NCAA': 'Estados Unidos (NCAA)',
    'NIC': 'Nicarágua',
    'PAR': 'Paraguai',
    'POL': 'Polônia',
    'POR': 'Portugal',
    'PUR': 'Porto Rico',
    'QAT': 'Catar',
    'ROM': 'Romênia',
    'RUS': 'Rússia',
    'SLO': 'Eslovênia',
    'SRB': 'Sérvia',
    'SUI': 'Suíça',
    'TUR': 'Turquia',
    'URU': 'Uruguai',
    'USA': 'Estados Unidos',
    'VEN': 'Venezuela',
    'NAIA': 'Estados Unidos (NAIA)',
    'JUCO': 'Estados Unidos (JUCO)',
    'G League': 'Estados Unidos (G League)',
    'NBA': 'Estados Unidos (NBA)',
    'WNBA': 'Estados Unidos (WNBA)',
    'EUROL': 'Euroliga',
    'MZRKL': 'Liga Adriática',
    'LAM': 'Liga das Américas',
    'LSA': 'Liga Sul-Americana',
    'BCL': 'Basketball Champions League',
    'Eurocup': 'EuroCup',
    'GOOD': 'Competição Internacional',
    'ANGT': 'Adidas Next Generation Tournament',
    'EYBL': 'European Youth Basketball League',
    'CEBL': 'Canadian Elite Basketball League',
    'USPO': 'Universidade Canadense',
    'FEL': 'FIBA Europe League',
    'IntCup': 'Copa Internacional',
    'AlKo': 'Liga Adriático-Kosovo',
    'ADR': 'Liga Adriática'
}

def get_country_from_league(league_code):
    """
    Extrai o código do país do código da liga.
    
    Args:
        league_code (str): Código da liga (ex: "ESP-1", "NCAA1")
        
    Returns:
        str: Código do país ou da liga principal
    """
    if not league_code or pd.isna(league_code):
        return "Desconhecido"
        
    # Para ligas com formato PAÍS-NÍVEL
    if "-" in league_code:
        return league_code.split("-")[0]
    
    # Para ligas específicas
    if any(code in league_code for code in ["NCAA", "JUCO", "NAIA", "NBA", "WNBA"]):
        return league_code.split("1")[0] if "1" in league_code else league_code
    
    # Para competições continentais e outras ligas
    for code in ["EUROL", "MZRKL", "LAM", "LSA", "BCL", "Eurocup", "GOOD", "ANGT", "EYBL", "CEBL", "USPO", "FEL", "IntCup", "AlKo", "ADR"]:
        if code in league_code:
            return code
    
    # Se não conseguir identificar, retorna o próprio código
    return league_code

def get_latest_league(df, group_by='NOME'):
    """
    Obtém a liga da última temporada de cada jogador.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados dos jogadores
        group_by (str): Coluna para agrupar os dados (normalmente 'NOME')
        
    Returns:
        dict: Dicionário com o nome do jogador e o código da liga da última temporada
    """
    latest_leagues = {}
    
    for name, group in df.groupby(group_by):
        # Ordenar por temporada
        sorted_group = group.sort_values(by='TEMPORADA')
        if not sorted_group.empty:
            # Obter a liga da última temporada
            latest_league = sorted_group.iloc[-1]['LIGA']
            latest_leagues[name] = latest_league
    
    return latest_leagues

def get_countries_from_data(df):
    """
    Obtém todos os países únicos presentes nos dados.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados dos jogadores
        
    Returns:
        list: Lista de países únicos
    """
    latest_leagues = get_latest_league(df)
    
    countries = []
    for league in latest_leagues.values():
        country_code = get_country_from_league(league)
        country_name = PAIS_MAPPING.get(country_code, country_code)
        if country_name not in countries:
            countries.append(country_name)
    
    # Ordenar países alfabeticamente
    countries.sort()
    
    # Adicionar opção "Todos os países" no início
    countries = ["Todos os países"] + countries
    
    return countries

def get_country_selection(df, key_suffix):
    """
    Função centralizada para seleção de país.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados dos jogadores
        key_suffix (str): Sufixo para a chave do componente
        
    Returns:
        str: País selecionado
    """
    countries = get_countries_from_data(df)
    return st.selectbox(
        "Selecione o País",
        countries,
        key=f"country_select_{key_suffix}"
    )

def filter_by_country(df, country):
    """
    Filtra os dados pelo país selecionado.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados dos jogadores
        country (str): País selecionado
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    if country == "Todos os países":
        return df
    
    # Obtém a liga da última temporada de cada jogador
    latest_leagues = get_latest_league(df)
    
    # Cria uma lista para armazenar os jogadores que estão no país selecionado
    players_in_country = []
    
    for player_name, league in latest_leagues.items():
        country_code = get_country_from_league(league)
        country_name = PAIS_MAPPING.get(country_code, country_code)
        if country_name == country:
            players_in_country.append(player_name)
    
    # Filtrar apenas jogadores no país selecionado
    return df[df['NOME'].isin(players_in_country)]

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
