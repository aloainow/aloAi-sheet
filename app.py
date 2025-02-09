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

# Interface principal atualizada
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
            # [resto do código permanece igual até a parte do resultado]
            
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
