import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("⏳ Preparando a Máquina do Tempo dos Dados...")

# 1. Leitura padrão
arquivos = ['2025_LoL_esports_match_data_from_OraclesElixir.csv', '2026_LoL_esports_match_data_from_OraclesElixir.csv']
df_lol = pd.concat([pd.read_csv(arq, low_memory=False) for arq in arquivos], ignore_index=True)

df_lck = df_lol[df_lol['league'] == 'LCK'].copy()
df_times = df_lck[df_lck['position'] == 'team'].copy()

# 2. ADICIONAMOS A COLUNA 'date' E TIRAMOS AS TRAPAÇAS (FB, Dragão, etc)
colunas_importantes = ['gameid', 'date', 'teamname', 'side', 'result']
df_limpo = df_times[colunas_importantes].copy()

# 3. A MÁGICA DO TEMPO: Ordenamos os jogos do mais antigo para o mais novo
df_limpo['date'] = pd.to_datetime(df_limpo['date'], utc=True)
df_limpo = df_limpo.sort_values('date')

# 4. Calculando o "Win Rate" acumulado ANTES do jogo começar
print("🧮 Calculando o Win Rate histórico de cada time...")
# O Python agrupa por time, pega os resultados, 'empurra' um jogo pra trás (shift) 
# para não usar o resultado do jogo atual, e calcula a média de vitórias até aquele dia.
df_limpo['win_rate_historico'] = df_limpo.groupby('teamname')['result'].transform(lambda x: x.shift(1).expanding().mean())

# Preenchemos os times que estão no primeiro jogo (sem histórico) com 50% (0.5)
df_limpo['win_rate_historico'] = df_limpo['win_rate_historico'].fillna(0.5)

# 5. O Ringue: Separando Lado Azul e Vermelho
df_azul = df_limpo[df_limpo['side'] == 'Blue'].copy().add_suffix('_blue').rename(columns={'gameid_blue': 'gameid'})
df_vermelho = df_limpo[df_limpo['side'] == 'Red'].copy().add_suffix('_red').rename(columns={'gameid_red': 'gameid'})

df_partidas = pd.merge(df_azul, df_vermelho, on='gameid')
df_partidas = df_partidas.rename(columns={'result_blue': 'blue_win'})
df_partidas = df_partidas.dropna()

# --- TREINAMENTO DO NOVO CÉREBRO ---
print("🤖 Treinando a IA com dados REAIS de Pré-Jogo...\n")

# Nossas pistas reais agora são a força de cada time baseada no histórico!
X = df_partidas[['win_rate_historico_blue', 'win_rate_historico_red']]
y = df_partidas['blue_win']

# Não podemos pegar jogos aleatórios para testar, temos que respeitar o tempo.
# Vamos treinar com os 80% jogos mais antigos e testar (apostar) nos 20% jogos mais recentes!
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=False)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_teste)

precisao = accuracy_score(y_teste, previsoes)

print("-" * 40)
print(f"🎯 TAXA DE ACERTO REAL (Pré-Jogo): {precisao * 100:.2f}%")
print("-" * 40)