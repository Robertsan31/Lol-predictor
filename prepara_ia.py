import pandas as pd

arquivos = ['2025_LoL_esports_match_data_from_OraclesElixir.csv', '2026_LoL_esports_match_data_from_OraclesElixir.csv']

print("⏳ Processando dados...")
df_lol = pd.concat([pd.read_csv(arq, low_memory=False) for arq in arquivos], ignore_index=True)

df_lck = df_lol[df_lol['league'] == 'LCK'].copy()
df_times = df_lck[df_lck['position'] == 'team'].copy()

colunas_importantes = ['gameid', 'teamname', 'side', 'result', 'firstblood', 'firstdragon', 'golddiffat15']
df_limpo = df_times[colunas_importantes]

print("🥊 Preparando o ringue (Juntando Lado Azul vs Lado Vermelho)...")

# 1. Separamos quem jogou de Azul e quem jogou de Vermelho
df_azul = df_limpo[df_limpo['side'] == 'Blue'].copy()
df_vermelho = df_limpo[df_limpo['side'] == 'Red'].copy()

# 2. Renomeamos as colunas para sabermos de quem é cada estatística
df_azul = df_azul.add_suffix('_blue')
df_vermelho = df_vermelho.add_suffix('_red')

# 3. Arrumamos o nome do ID do jogo para podermos juntar as duas tabelas
df_azul = df_azul.rename(columns={'gameid_blue': 'gameid'})
df_vermelho = df_vermelho.rename(columns={'gameid_red': 'gameid'})

# 4. Juntamos as duas partes! Onde o ID do jogo for igual, ele vira uma linha só.
df_partidas = pd.merge(df_azul, df_vermelho, on='gameid')

# Como o resultado final é óbvio (se o azul ganhou, o vermelho perdeu), 
# deixamos só a coluna que diz se o Azul ganhou ou não (1 = Vitória do Azul, 0 = Vitória do Vermelho)
df_partidas = df_partidas.rename(columns={'result_blue': 'blue_win'})
df_partidas = df_partidas.drop(columns=['result_red', 'side_blue', 'side_red'])

print("✅ Confrontos criados com sucesso!")
print(f"📊 Total de Partidas Únicas para treinar a IA: {df_partidas.shape[0]}")
print("\nOlha como a IA vai ler o jogo agora:")
print(df_partidas[['teamname_blue', 'teamname_red', 'firstblood_blue', 'firstdragon_blue', 'blue_win']].head())