def format_table(df):
    """Helper function to format tables with all relevant statistics"""
    try:
        # Selecionar todas as colunas relevantes
        stats_columns = [
            'Player Name', 'Team Name', 'League', 'Age', 'Height', 'Pos',
            'GP', 'MPG', 'PPG', 'RPG', 'APG', 'SPG', 'BPG',
            'FG%', '3P%', 'FT%', 'EFF'
        ]
        
        # Filtrar apenas colunas disponíveis
        available_columns = [col for col in stats_columns if col in df.columns]
        result_df = df[available_columns]
        
        # Formatar números para 1 casa decimal
        numeric_columns = result_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            result_df[col] = result_df[col].round(1)
        
        # Mostrar a tabela
        st.table(result_df)
    except Exception as e:
        st.error(f"Erro ao formatar tabela: {str(e)}")
