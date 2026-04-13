import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="LoL Predictor PRO", layout="wide")

@st.cache_data
def treinar_modelo_super_cerebro():
    arquivos = ['2025_LoL_esports_match_data_from_OraclesElixir.csv', '2026_LoL_esports_match_data_from_OraclesElixir.csv']
    df_lol = pd.concat([pd.read_csv(arq, low_memory=False) for arq in arquivos], ignore_index=True)
    
    df_lck = df_lol[df_lol['league'] == 'LCK'].copy()
    df_times = df_lck[df_lck['position'] == 'team'].copy()
    
    # ADICIONAMOS A COLUNA 'patch' AQUI!
    df_limpo = df_times[['gameid', 'date', 'teamname', 'side', 'result', 'patch']].copy()
    df_limpo['date'] = pd.to_datetime(df_limpo['date'], utc=True)
    df_limpo = df_limpo.sort_values('date')
    
    # Limpando o nome do patch (as vezes vem quebrado na planilha)
    df_limpo['patch'] = df_limpo['patch'].astype(str).str.extract(r'(\d+\.\d+)')[0]
    
    # --- AS 4 CAMADAS DE INTELIGÊNCIA ---
    # 1. Histórico Geral
    df_limpo['wr_geral'] = df_limpo.groupby('teamname')['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    # 2. Momento (5 Jogos)
    df_limpo['wr_5_jogos'] = df_limpo.groupby('teamname')['result'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0.5)
    # 3. Lado do Mapa
    df_limpo['wr_lado'] = df_limpo.groupby(['teamname', 'side'])['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    
    # 4. SUPER CÉREBRO: O META (Taxa de vitória do time neste Patch específico)
    df_limpo['wr_patch'] = df_limpo.groupby(['teamname', 'patch'])['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    
    # O Ringue
    df_azul = df_limpo[df_limpo['side'] == 'Blue'].copy().add_suffix('_blue').rename(columns={'gameid_blue': 'gameid'})
    df_vermelho = df_limpo[df_limpo['side'] == 'Red'].copy().add_suffix('_red').rename(columns={'gameid_red': 'gameid'})
    
    df_partidas = pd.merge(df_azul, df_vermelho, on='gameid').rename(columns={'result_blue': 'blue_win'}).dropna()
    
    # Agora a IA analisa 8 pistas diferentes!
    X = df_partidas[['wr_geral_blue', 'wr_5_jogos_blue', 'wr_lado_blue', 'wr_patch_blue',
                     'wr_geral_red', 'wr_5_jogos_red', 'wr_lado_red', 'wr_patch_red']]
    y = df_partidas['blue_win']
    
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Aumentamos o tamanho do cérebro (n_estimators de 200 para 300) para processar o patch
    modelo = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    modelo.fit(X_treino, y_treino)
    
    acerto = accuracy_score(y_teste, modelo.predict(X_teste)) * 100
    modelo.fit(X, y)
    
    lista_times = sorted(df_limpo['teamname'].unique())
    lista_patches = sorted(df_limpo['patch'].dropna().unique(), reverse=True)
    
    # Guardamos o último registro de cada time em cada patch
    ultimos_dados = df_limpo.groupby(['teamname', 'side', 'patch']).last().reset_index()
    
    return modelo, lista_times, lista_patches, ultimos_dados, acerto

with st.spinner("Conectando o Super Cérebro e analisando os Patches da Riot Games..."):
    modelo_ia, times_disponiveis, patches_disponiveis, dados_recentes, acuracia_modelo = treinar_modelo_super_cerebro()

# --- FRONT-END ---
st.title("🏆 LoL Predictor PRO (v3.0 - Super Cérebro)")
st.caption(f"🧠 Acurácia atual da IA: **{acuracia_modelo:.1f}%**")
st.markdown("---")

aba1, aba2 = st.tabs(["🤖 Previsão de Confronto", "💰 Gestão de Banca"])

with aba1:
    # Novo seletor de Patch!
    patch_atual = st.selectbox("🛠️ Em qual Patch será o jogo?", patches_disponiveis)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        time_azul = st.selectbox("🟦 Lado Azul", times_disponiveis, index=0)
    with col2:
        time_vermelho = st.selectbox("🟥 Lado Vermelho", times_disponiveis, index=1)
        
    if st.button("Analisar Confronto no Meta Atual", type="primary"):
        if time_azul == time_vermelho:
            st.error("Selecione times diferentes.")
        else:
            # Buscando os dados do time Azul pro Patch selecionado
            stats_azul = dados_recentes[(dados_recentes['teamname'] == time_azul) & 
                                        (dados_recentes['side'] == 'Blue') & 
                                        (dados_recentes['patch'] == patch_atual)]
            
            # Buscando os dados do time Vermelho pro Patch selecionado
            stats_verm = dados_recentes[(dados_recentes['teamname'] == time_vermelho) & 
                                        (dados_recentes['side'] == 'Red') & 
                                        (dados_recentes['patch'] == patch_atual)]
            
            # Trava de segurança: se o time nunca jogou nesse patch, o Python avisa
            if stats_azul.empty or stats_verm.empty:
                st.warning("⚠️ Um dos times ainda não jogou partidas oficiais neste Patch. A IA usará a taxa base de 50% para a adaptação ao meta.")
                wr_patch_a = 0.5 if stats_azul.empty else stats_azul['wr_patch'].values[0]
                wr_patch_v = 0.5 if stats_verm.empty else stats_verm['wr_patch'].values[0]
                
                # Pega os outros dados recentes (ignorando o patch) pra não quebrar o app
                dados_gerais_a = dados_recentes[(dados_recentes['teamname'] == time_azul) & (dados_recentes['side'] == 'Blue')].iloc[-1]
                dados_gerais_v = dados_recentes[(dados_recentes['teamname'] == time_vermelho) & (dados_recentes['side'] == 'Red')].iloc[-1]
            else:
                wr_patch_a = stats_azul['wr_patch'].values[0]
                wr_patch_v = stats_verm['wr_patch'].values[0]
                dados_gerais_a = stats_azul.iloc[0]
                dados_gerais_v = stats_verm.iloc[0]
                
            X_previsao = [[
                dados_gerais_a['wr_geral'], dados_gerais_a['wr_5_jogos'], dados_gerais_a['wr_lado'], wr_patch_a,
                dados_gerais_v['wr_geral'], dados_gerais_v['wr_5_jogos'], dados_gerais_v['wr_lado'], wr_patch_v
            ]]
            
            probabilidade = modelo_ia.predict_proba(X_previsao)[0]
            
            st.success("Análise tática concluída!")
            col_a, col_b = st.columns(2)
            col_a.metric(label=f"Vitória - {time_azul}", value=f"{probabilidade[1]*100:.1f}%")
            col_b.metric(label=f"Vitória - {time_vermelho}", value=f"{probabilidade[0]*100:.1f}%")

with aba2:
    st.subheader("Calculadora Avançada de Stake (Critério de Kelly)")
    
    col1, col2 = st.columns(2)
    with col1:
        banca = st.number_input("Banca Atual (R$)", min_value=10.0, value=1000.0, step=50.0)
        chance_ia = st.number_input("Probabilidade da IA (%)", min_value=1.0, max_value=99.0, value=55.0)
        odd_casa = st.number_input("Odd da Casa de Apostas", min_value=1.01, value=1.85, step=0.05)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True) # Espaçamento
        perfil_risco = st.selectbox("Seu Perfil de Risco (Fração de Kelly)", [
            "Conservador (1/8 - Muito Seguro)",
            "Recomendado (1/4 - Equilibrado)",
            "Agressivo (1/2 - Maior Risco)",
            "Kamikaze (Kelly Cheio - NÃO RECOMENDADO)"
        ], index=1)

    # Dicionário para mapear a escolha do usuário para a matemática
    fracoes = {
        "Conservador (1/8 - Muito Seguro)": 8,
        "Recomendado (1/4 - Equilibrado)": 4,
        "Agressivo (1/2 - Maior Risco)": 2,
        "Kamikaze (Kelly Cheio - NÃO RECOMENDADO)": 1
    }
    divisor_kelly = fracoes[perfil_risco]

    # --- MATEMÁTICA ---
    p = chance_ia / 100
    b = odd_casa - 1
    
    # Cálculo do Valor Esperado (EV)
    ev_percentual = ((p * odd_casa) - 1) * 100

    st.markdown("---")
    
    if b > 0:
        f_star = (b * p - (1 - p)) / b
        
        if f_star > 0:
            aposta_sugerida = (banca * f_star) / divisor_kelly
            porcentagem_banca = (aposta_sugerida / banca) * 100
            unidades = porcentagem_banca # Considerando 1 Unidade = 1% da Banca

            st.success("✅ **Aposta de Valor (+EV) Encontrada!**")
            
            # Métricas visuais bacanas
            metrica1, metrica2, metrica3 = st.columns(3)
            metrica1.metric(label="Vantagem Matemática (EV)", value=f"+{ev_percentual:.2f}%")
            metrica2.metric(label="Tamanho da Aposta (Unidades)", value=f"{unidades:.2f} U")
            metrica3.metric(label="Valor em Dinheiro", value=f"R$ {aposta_sugerida:.2f}")
            
            st.info(f"💡 **Leitura:** O modelo sugere arriscar **{porcentagem_banca:.2f}%** da sua banca nesta entrada de acordo com o seu perfil de risco.")
            
        else:
            st.error("🛑 **EV Negativo (Esperança Matemática Ruim).**")
            st.write(f"A longo prazo, apostar nessa Odd te fará perder dinheiro (EV: **{ev_percentual:.2f}%**). Fique de fora desse jogo ou busque uma Odd maior ao vivo.")