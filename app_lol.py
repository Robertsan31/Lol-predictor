import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="LoL Predictor PRO", layout="wide")

@st.cache_data
def treinar_modelo_avancado():
    arquivos = ['2025_LoL_esports_match_data_from_OraclesElixir.csv', '2026_LoL_esports_match_data_from_OraclesElixir.csv']
    df_lol = pd.concat([pd.read_csv(arq, low_memory=False) for arq in arquivos], ignore_index=True)
    
    df_lck = df_lol[df_lol['league'] == 'LCK'].copy()
    df_times = df_lck[df_lck['position'] == 'team'].copy()
    
    df_limpo = df_times[['gameid', 'date', 'teamname', 'side', 'result']].copy()
    df_limpo['date'] = pd.to_datetime(df_limpo['date'], utc=True)
    df_limpo = df_limpo.sort_values('date')
    
    # --- A NOVA INTELIGÊNCIA (MÚLTIPLAS VARIÁVEIS) ---
    # 1. Win Rate Geral da História
    df_limpo['wr_geral'] = df_limpo.groupby('teamname')['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    
    # 2. O Momento (Win Rate apenas dos últimos 5 jogos)
    df_limpo['wr_5_jogos'] = df_limpo.groupby('teamname')['result'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0.5)
    
    # 3. Especialidade do Lado (Win Rate jogando especificamente de Azul ou Vermelho)
    df_limpo['wr_lado'] = df_limpo.groupby(['teamname', 'side'])['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    
    # O Ringue
    df_azul = df_limpo[df_limpo['side'] == 'Blue'].copy().add_suffix('_blue').rename(columns={'gameid_blue': 'gameid'})
    df_vermelho = df_limpo[df_limpo['side'] == 'Red'].copy().add_suffix('_red').rename(columns={'gameid_red': 'gameid'})
    
    df_partidas = pd.merge(df_azul, df_vermelho, on='gameid').rename(columns={'result_blue': 'blue_win'}).dropna()
    
    # Agora a IA tem 6 pistas para olhar, não apenas 2!
    X = df_partidas[['wr_geral_blue', 'wr_5_jogos_blue', 'wr_lado_blue', 
                     'wr_geral_red', 'wr_5_jogos_red', 'wr_lado_red']]
    y = df_partidas['blue_win']
    
    # Separamos um pedaço para calcular a acurácia para mostrar no App
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    modelo = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    modelo.fit(X_treino, y_treino)
    
    acerto = accuracy_score(y_teste, modelo.predict(X_teste)) * 100
    
    # Treina com TUDO para a previsão final ficar perfeita
    modelo.fit(X, y)
    
    lista_times = sorted(df_limpo['teamname'].unique())
    
    # Salva as últimas estatísticas de cada time para a previsão
    ultimos_dados = df_limpo.groupby(['teamname', 'side']).last().reset_index()
    
    return modelo, lista_times, ultimos_dados, acerto

with st.spinner("Atualizando Redes Neurais com Lado do Mapa e Histórico Recente..."):
    modelo_ia, times_disponiveis, dados_recentes, acuracia_modelo = treinar_modelo_avancado()

# --- FRONT-END ---
st.title("🏆 LoL Predictor PRO (v2.0)")
st.caption(f"🧠 Acurácia atual da IA (Backtesting): **{acuracia_modelo:.1f}%**")
st.markdown("---")

aba1, aba2 = st.tabs(["🤖 Previsão de Confronto", "💰 Gestão de Banca"])

with aba1:
    col1, col2 = st.columns(2)
    with col1:
        time_azul = st.selectbox("🟦 Lado Azul", times_disponiveis, index=0)
    with col2:
        time_vermelho = st.selectbox("🟥 Lado Vermelho", times_disponiveis, index=1)
        
    if st.button("Analisar Confronto", type="primary"):
        if time_azul == time_vermelho:
            st.error("Selecione times diferentes.")
        else:
            # Resgatando a ficha criminal atualizada do Time Azul (Lado Azul)
            stats_azul = dados_recentes[(dados_recentes['teamname'] == time_azul) & (dados_recentes['side'] == 'Blue')]
            # Resgatando a ficha criminal atualizada do Time Vermelho (Lado Vermelho)
            stats_verm = dados_recentes[(dados_recentes['teamname'] == time_vermelho) & (dados_recentes['side'] == 'Red')]
            
            # Montando a configuração exata para a IA prever
            X_previsao = [[
                stats_azul['wr_geral'].values[0], stats_azul['wr_5_jogos'].values[0], stats_azul['wr_lado'].values[0],
                stats_verm['wr_geral'].values[0], stats_verm['wr_5_jogos'].values[0], stats_verm['wr_lado'].values[0]
            ]]
            
            probabilidade = modelo_ia.predict_proba(X_previsao)[0]
            
            st.success("Análise tática concluída!")
            col_a, col_b = st.columns(2)
            col_a.metric(label=f"Vitória - {time_azul}", value=f"{probabilidade[1]*100:.1f}%")
            col_b.metric(label=f"Vitória - {time_vermelho}", value=f"{probabilidade[0]*100:.1f}%")

with aba2:
    st.subheader("Calculadora de Stake (Kelly Fracionado)")
    banca = st.number_input("Banca Atual (R$)", value=1000.0)
    chance_ia = st.number_input("Probabilidade da IA (%)", min_value=1.0, max_value=99.0, value=55.0)
    odd_casa = st.number_input("Odd da Casa de Apostas", min_value=1.01, value=1.85)
    
    p = chance_ia / 100
    b = odd_casa - 1
    
    if b > 0:
        f = (b * p - (1 - p)) / b
        if f > 0:
            st.success("📈 Aposta de Valor Encontrada!")
            st.write(f"Risco recomendado (1/4 Kelly): **R$ {(banca * f) / 4:.2f}**")
        else:
            st.error("🛑 EV Negativo. Não aposte nesta partida.")