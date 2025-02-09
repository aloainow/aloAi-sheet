def process_stats_query(df, age=None, stat_column=None, height=None, multiple_stats=False):
    """Processa consulta de estatísticas com múltiplos critérios"""
    try:
        # Filtrar por idade se especificado
        if age is not None:
            df = df[df['Age'] == age].copy()
        else:
            df = df.copy()

        # Lista completa de colunas para exibição
        base_columns = ['Player Name', 'Team Name', 'League', 'Age', 'Height', 'Pos', 'TYPE']
        
        # Estatísticas ofensivas
        offensive_stats = ['PPG', 'APG', '2PM', '2PA', '2P%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%']
        
        # Estatísticas defensivas
        defensive_stats = ['RPG', 'BPG', 'SPG', 'ORB', 'DRB']
        
        # Outras estatísticas
        other_stats = ['EFF', 'MPG', 'GP', 'PF', 'TO']

        # Se são múltiplas estatísticas específicas (EFF, PPG, APG)
        if multiple_stats:
            # Normalizar cada estatística para ter peso igual
            df['EFF_norm'] = (df['EFF'] - df['EFF'].min()) / (df['EFF'].max() - df['EFF'].min())
            df['PPG_norm'] = (df['PPG'] - df['PPG'].min()) / (df['PPG'].max() - df['PPG'].min())
            df['APG_norm'] = (df['APG'] - df['APG'].min()) / (df['APG'].max() - df['APG'].min())
            
            # Calcular pontuação combinada
            df['Combined_Score'] = (df['EFF_norm'] + df['PPG_norm'] + df['APG_norm']) / 3
            
            display_columns = base_columns + ['EFF', 'PPG', 'APG', 'Combined_Score']
            result = df[display_columns].sort_values(by='Combined_Score', ascending=False)
            
        # Se uma estatística específica foi solicitada
        elif stat_column and stat_column in df.columns:
            display_columns = base_columns + [stat_column]
            result = df[display_columns].sort_values(by=stat_column, ascending=False)
        else:
            # Mostrar todas as estatísticas disponíveis
            all_stats = base_columns + offensive_stats + defensive_stats + other_stats
            available_columns = [col for col in all_stats if col in df.columns]
            
            # Calcular métrica ofensiva e defensiva
            df['Metrica_Ofensiva'] = (
                df['PPG'] * 0.4 + 
                df['APG'] * 0.3 + 
                df['2P%'] * 0.15 + 
                df['3P%'] * 0.15
            )
            
            df['Metrica_Defensiva'] = (
                df['RPG'] * 0.4 + 
                df['BPG'] * 0.3 + 
                df['SPG'] * 0.3
            )
            
            available_columns.extend(['Metrica_Ofensiva', 'Metrica_Defensiva'])
            result = df[available_columns].sort_values(by=['Metrica_Ofensiva', 'Metrica_Defensiva'], ascending=[False, False])
        
        # Arredondar valores numéricos
        numeric_columns = result.select_dtypes(include=['float64', 'float32']).columns
        result[numeric_columns] = result[numeric_columns].round(2)
        
        return result
    except Exception as e:
        st.error(f"Erro ao processar estatísticas: {str(e)}")
        return None

# Na parte do processamento da consulta, atualize a verificação de palavras-chave:
stat_keywords = {
    'PPG': ['ppg', 'pontos por jogo', 'pontos'],
    'APG': ['apg', 'assistências', 'assistencias'],
    'RPG': ['rpg', 'rebotes'],
    'BPG': ['bpg', 'bloqueios'],
    'SPG': ['spg', 'roubos', 'roubos de bola'],
    '2P%': ['2p%', 'field goal', 'arremessos de 2'],
    '3P%': ['3p%', 'three point', 'arremessos de 3'],
    'EFF': ['eff', 'eficiência', 'eficiencia'],
    'ORB': ['orb', 'rebotes ofensivos'],
    'DRB': ['drb', 'rebotes defensivos']
}
