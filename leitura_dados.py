import pandas as pd

arquivos = [
    '2025_LoL_esports_match_data_from_OraclesElixir.csv',
    '2026_LoL_esports_match_data_from_OraclesElixir.csv'
]

print("⏳ Carregando as bases de 2025 e 2026...")
lista_tabelas = [pd.read_csv(arq, low_memory=False) for arq in arquivos]
df_lol = pd.concat(lista_tabelas, ignore_index=True)

print("✅ Base bruta carregada!")
print("-" * 40)

# --- A MÁGICA DA LIMPEZA COMEÇA AQUI ---
print("🧹 Filtrando os dados inúteis...")

# 1. Filtrar apenas jogos da LCK
df_lck = df_lol[df_lol['league'] == 'LCK'].copy()

# 2. Filtrar apenas as linhas de 'Time' (ignorando os dados individuais dos jogadores)
df_times = df_lck[df_lck['position'] == 'team'].copy()

# 3. Selecionar apenas as colunas vitais para a nossa IA
colunas_importantes = [
    'gameid',       # ID único da partida (para sabermos quem jogou contra quem)
    'teamname',     # Nome do time
    'side',         # Lado do mapa (Blue ou Red)
    'result',       # Resultado (1 para Vitória, 0 para Derrota)
    'gamelength',   # Tempo de jogo em segundos
    'firstblood',   # Fez o First Blood? (1 para Sim, 0 para Não)
    'firstdragon',  # Fez o Primeiro Dragão? (1 para Sim, 0 para Não)
    'golddiffat15'  # Diferença de ouro aos 15 minutos
]

df_limpo = df_times[colunas_importantes]

print("✅ Limpeza concluída!")
print(f"📊 Novo tamanho da base (Apenas LCK consolidada): {df_limpo.shape[0]} linhas e {df_limpo.shape[1]} colunas.\n")

print("Olha a cara dos nossos dados limpos agora:")
print(df_limpo.head())