import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(page_title="LoL Predictor PRO", layout="wide")

# --- MEMÓRIA DO DIÁRIO DE APOSTAS ---
if 'diario_apostas' not in st.session_state:
    st.session_state['diario_apostas'] = pd.DataFrame(columns=[
        'Data', 'Mercado', 'Confronto', 'Odd', 'Stake (R$)', 'Status', 'Retorno (R$)'
    ])

@st.cache_data
def treinar_super_modelo_multimercados():
    arquivos = ['2025_LoL_esports_match_data_from_OraclesElixir.csv', '2026_LoL_esports_match_data_from_OraclesElixir.csv']
    df_lol = pd.concat([pd.read_csv(arq, low_memory=False) for arq in arquivos], ignore_index=True)
    
    df_lck = df_lol[df_lol['league'] == 'LCK'].copy()
    df_times = df_lck[df_lck['position'] == 'team'].copy()
    
    # 1. Usando APENAS colunas oficiais que sabemos que existem!
    cols = ['gameid', 'date', 'teamname', 'side', 'result', 'patch', 'playoffs', 'dragons', 'gamelength']
    df_limpo = df_times[cols].copy()
    
    # Garantindo que não tenha espaço vazio atrapalhando o cálculo
    df_limpo['dragons'] = pd.to_numeric(df_limpo['dragons'], errors='coerce').fillna(0)
    
    # A Mágica do Pandas: Descobrindo os dragões do adversário matematicamente
    df_limpo['total_dragons_partida'] = df_limpo.groupby('gameid')['dragons'].transform('sum')
    df_limpo['opp_dragons'] = df_limpo['total_dragons_partida'] - df_limpo['dragons']
    
    # 2. Criando os alvos dos novos mercados
    df_limpo['over_4_dragons'] = (df_limpo['total_dragons_partida'] > 4).astype(int)
    df_limpo['mais_dragons'] = (df_limpo['dragons'] > df_limpo['opp_dragons']).astype(int)
    df_limpo['jogo_longo'] = (df_limpo['gamelength'] > 1920).astype(int) # Mais de 32 minutos
    
    df_limpo['date'] = pd.to_datetime(df_limpo['date'], utc=True)
    df_limpo = df_limpo.sort_values('date')
    
    # 3. Consertando o Patch
    df_limpo['patch'] = df_limpo['patch'].astype(str).str.extract(r'(\d+\.\d+)')[0]
    df_limpo['patch'] = df_limpo['patch'].str.replace('^16\.', '26.', regex=True)
    
    # 4. Criando histórico dinâmico (Win Rate e Dragon Rate)
    for col in ['result', 'mais_dragons', 'dragons']:
        df_limpo[f'media_{col}'] = df_limpo.groupby('teamname')[col].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)

    df_azul = df_limpo[df_limpo['side'] == 'Blue'].copy().add_suffix('_blue').rename(columns={'gameid_blue': 'gameid'})
    df_vermelho = df_limpo[df_limpo['side'] == 'Red'].copy().add_suffix('_red').rename(columns={'gameid_red': 'gameid'})
    df_p = pd.merge(df_azul, df_vermelho, on='gameid').dropna()

    # 5. Novas pistas que a IA vai olhar (incluindo se é MD5/Playoff)
    features = ['media_result_blue', 'media_mais_dragons_blue', 'media_dragons_blue',
                'media_result_red', 'media_mais_dragons_red', 'media_dragons_red', 'playoffs_blue']
    X = df_p[features]
    
    # 6. Treinando os 4 Cérebros
    m_vit = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df_p['result_blue'])
    m_dra = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df_p['mais_dragons_blue'])
    m_tot_dra = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df_p['over_4_dragons_blue'])
    m_tempo = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df_p['jogo_longo_blue'])
    
    # 7. Raio-X do modelo Principal
    importancias = m_vit.feature_importances_
    nomes_variaveis = ['Blue - Win Rate', 'Blue - +Dragões', 'Blue - Média Dragões',
                       'Red - Win Rate', 'Red - +Dragões', 'Red - Média Dragões', 'Fator MD5/Playoff']
    df_peso_ia = pd.DataFrame({'Peso na Decisão (%)': importancias * 100}, index=nomes_variaveis)
    
    lista_times = sorted(df_limpo['teamname'].unique())
    lista_patches = sorted(df_limpo['patch'].dropna().unique(), reverse=True)
    ultimos_dados = df_limpo.groupby(['teamname', 'patch']).last().reset_index()
    
    return lista_times, lista_patches, ultimos_dados, m_vit, m_dra, m_tot_dra, m_tempo, df_peso_ia

with st.spinner("Conectando os 4 Cérebros da IA e gerando Diário..."):
    times, patches, dados, m_vit, m_dra, m_tot_dra, m_tempo, df_peso_ia = treinar_super_modelo_multimercados()

# --- FRONT-END ---
st.title("🏆 LoL Predictor PRO (v5.0 - Multimercados)")
st.markdown("---")

aba1, aba2, aba3 = st.tabs(["🤖 Previsões & Raio-X", "💰 Gestão de Banca", "📊 Diário de Apostas"])

with aba1:
    col_p, col_m = st.columns(2)
    p_atual = col_p.selectbox("🛠️ Em qual Patch será o jogo?", patches)
    is_playoff = col_m.checkbox("⚠️ É uma MD5 (Playoffs)?")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    t_azul = c1.selectbox("🟦 Lado Azul", times, index=0)
    t_red = c2.selectbox("🟥 Lado Vermelho", times, index=1)
    
    if st.button("Analisar Todos os Mercados", type="primary"):
        if t_azul == t_red:
            st.error("Selecione times diferentes.")
        else:
            s_a = dados[(dados['teamname'] == t_azul) & (dados['patch'] == p_atual)]
            s_r = dados[(dados['teamname'] == t_red) & (dados['patch'] == p_atual)]
            
            if s_a.empty: s_a = dados[dados['teamname'] == t_azul].iloc[[-1]]
            if s_r.empty: s_r = dados[dados['teamname'] == t_red].iloc[[-1]]
            
            input_ia = [[s_a['media_result'].values[0], s_a['media_mais_dragons'].values[0], s_a['media_dragons'].values[0],
                         s_r['media_result'].values[0], s_r['media_mais_dragons'].values[0], s_r['media_dragons'].values[0],
                         1 if is_playoff else 0]]
            
            prob_v = m_vit.predict_proba(input_ia)[0]
            prob_d = m_dra.predict_proba(input_ia)[0]
            prob_td = m_tot_dra.predict_proba(input_ia)[0]
            prob_t = m_tempo.predict_proba(input_ia)[0]
            
            st.success("Análise tática e de objetivos concluída!")
            
            st.subheader("🎯 Mercado Principal")
            col_a, col_b = st.columns(2)
            col_a.metric(label=f"Vitória - {t_azul}", value=f"{prob_v[1]*100:.1f}%")
            col_b.metric(label=f"Vitória - {t_red}", value=f"{prob_v[0]*100:.1f}%")
            
            st.subheader("🐲 Mercados Secundários (Objetivos e Tempo)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Mais Dragões (Azul)", f"{prob_d[1]*100:.1f}%")
            m2.metric("Over 4.5 Dragões", f"{prob_td[1]*100:.1f}%")
            m3.metric("Jogo Longo (+32min)", f"{prob_t[1]*100:.1f}%")

            st.markdown("---")
            st.subheader("🔍 Raio-X do Vencedor")
            st.write("Veja como as novas estatísticas de dragões e MD5 impactam o cérebro da IA para escolher o vencedor.")
            st.bar_chart(df_peso_ia)

with aba2:
    st.subheader("Calculadora Avançada de Stake")
    col1, col2 = st.columns(2)
    with col1:
        banca = st.number_input("Banca Atual (R$)", min_value=10.0, value=1000.0, step=50.0)
        chance_ia = st.number_input("Probabilidade da IA (%)", min_value=1.0, max_value=99.0, value=55.0)
        odd_casa = st.number_input("Odd da Casa de Apostas", min_value=1.01, value=1.85, step=0.05)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        perfil_risco = st.selectbox("Seu Perfil de Risco (Fração de Kelly)", [
            "Conservador (1/8 - Muito Seguro)", "Recomendado (1/4 - Equilibrado)", 
            "Agressivo (1/2 - Maior Risco)", "Kamikaze (1/1 - NÃO RECOMENDADO)"
        ], index=1)

    fracoes = {"Conservador (1/8 - Muito Seguro)": 8, "Recomendado (1/4 - Equilibrado)": 4, "Agressivo (1/2 - Maior Risco)": 2, "Kamikaze (1/1 - NÃO RECOMENDADO)": 1}
    divisor_kelly = fracoes[perfil_risco]

    p = chance_ia / 100
    b = odd_casa - 1
    ev_percentual = ((p * odd_casa) - 1) * 100

    st.markdown("---")
    if b > 0:
        f_star = (b * p - (1 - p)) / b
        if f_star > 0:
            aposta_sugerida = (banca * f_star) / divisor_kelly
            st.success("✅ **Aposta de Valor (+EV) Encontrada!**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Vantagem (EV)", f"+{ev_percentual:.2f}%")
            m2.metric("Tamanho (Unidades)", f"{(aposta_sugerida / banca) * 100:.2f} U")
            m3.metric("Apostar em Dinheiro", f"R$ {aposta_sugerida:.2f}")
            
            st.markdown("---")
            st.write("📝 **Salvar no Diário**")
            c_merc, c_nome = st.columns(2)
            mercado_nome = c_merc.text_input("Mercado (Ex: Over 4.5 Drags)")
            confronto_nome = c_nome.text_input("Confronto (Ex: T1 vs GenG)")
            
            if st.button("Salvar no Diário de Apostas"):
                nova_aposta = pd.DataFrame([{
                    'Data': datetime.now().strftime("%d/%m/%Y"),
                    'Mercado': mercado_nome,
                    'Confronto': confronto_nome,
                    'Odd': odd_casa,
                    'Stake (R$)': round(aposta_sugerida, 2),
                    'Status': 'Pendente',
                    'Retorno (R$)': 0.0
                }])
                st.session_state['diario_apostas'] = pd.concat([st.session_state['diario_apostas'], nova_aposta], ignore_index=True)
                st.toast("Aposta salva com sucesso no Diário!")
        else:
            st.error(f"🛑 **EV Negativo ({ev_percentual:.2f}%).** A longo prazo, apostar nessa Odd te fará perder dinheiro.")

with aba3:
    st.subheader("📊 Seu Diário de Apostas")
    st.write("Dê dois cliques na coluna **Status** para mudar de 'Pendente' para 'Ganha' ou 'Perdida'.")
    
    df_editado = st.data_editor(
        st.session_state['diario_apostas'],
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status da Aposta",
                options=["Pendente", "Ganha", "Perdida", "Reembolso"],
                required=True,
            ),
            "Retorno (R$)": st.column_config.NumberColumn(
                "Retorno (R$)",
                format="R$ %.2f"
            )
        },
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True
    )
    
    st.session_state['diario_apostas'] = df_editado
    lucro_total = df_editado['Retorno (R$)'].sum()
    st.markdown("---")
    if lucro_total > 0:
        st.metric("Lucro Total (PNL)", f"R$ {lucro_total:.2f}", delta="Positivo")
    else:
        st.metric("Lucro Total (PNL)", f"R$ {lucro_total:.2f}", delta="Negativo", delta_color="inverse")