import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
from PIL import Image
import requests
import io

# --- IMPORTAÇÕES NOVAS PARA A IA LER IMAGENS ---
import google.generativeai as genai

st.set_page_config(page_title="LoL Predictor PRO v12", layout="wide")

# --- CONECTANDO O CÉREBRO DO GEMINI (SCANNER) E PANDASCORE ---
api_ativa = False
if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        api_ativa = True
    except Exception as e:
        st.error(f"Erro ao configurar IA de visão: {e}")
else:
    print("Atenção: GEMINI_API_KEY não encontrada nos Secrets do Streamlit.")

# Chave secreta da PandaScore (Tenta pegar PANDASCORE_API_KEY, se não achar, tenta a antiga)
pandascore_key = st.secrets.get("PANDASCORE_API_KEY", st.secrets.get("ODDS_API_KEY", None))

# --- MEMÓRIA DO DIÁRIO DE APOSTAS E DA ANÁLISE ---
if 'diario_apostas' not in st.session_state:
    st.session_state['diario_apostas'] = pd.DataFrame(columns=[
        'Data', 'Mercado', 'Confronto', 'Odd', 'Stake (R$)', 'Status', 'Retorno (R$)'
    ])
if 'analise_salva' not in st.session_state:
    st.session_state['analise_salva'] = False

# --- O CÃO FAREJADOR (BUSCA DE JOGOS PANDASCORE) ---
@st.cache_data(ttl=1800) # Guarda o resultado por 30 minutos
def buscar_jogos_pandascore(api_key):
    if not api_key:
        return None
    
    url = "https://api.pandascore.co/lol/matches/upcoming"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
        else:
            return {"erro": res.status_code, "mensagem": res.text}
    except Exception as e:
        return {"erro": "Exception", "mensagem": str(e)}

# --- MOTOR DA INTELIGÊNCIA ARTIFICIAL (V12 - MERCADO DE KILLS) ---
@st.cache_resource(show_spinner=False)
def treinar_motor_dinamico_v12():
    filenames = [
        "2025_LoL_esports_match_data_from_OraclesElixir.csv",
        "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    ]
    
    urls = {
        "2025_LoL_esports_match_data_from_OraclesElixir.csv": "https://oracleselixir-downloadable-files.s3-us-west-2.amazonaws.com/2025_LoL_esports_match_data_from_OraclesElixir.csv",
        "2026_LoL_esports_match_data_from_OraclesElixir.csv": "https://oracleselixir-downloadable-files.s3-us-west-2.amazonaws.com/2026_LoL_esports_match_data_from_OraclesElixir.csv"
    }
    
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://oracleselixir.com/"}
    dfs = []

    for nome_arq in filenames:
        try:
            # 1. TENTA LER O ARQUIVO LOCAL PRIMEIRO
            df_temp = pd.read_csv(nome_arq, low_memory=False)
            dfs.append(df_temp)
        except Exception:
            # 2. SE NÃO ACHAR LOCAL, TENTA BAIXAR
            try:
                res = requests.get(urls[nome_arq], headers=headers, timeout=10)
                if res.status_code == 200:
                    dfs.append(pd.read_csv(io.StringIO(res.text), low_memory=False))
            except:
                pass

    if not dfs:
        return [], [], pd.DataFrame(), None, None, None, None, pd.DataFrame(), pd.DataFrame(), "Erro de Base"

    df_lol = pd.concat(dfs, ignore_index=True)
    df_lck = df_lol[df_lol['league'] == 'LCK'].copy()
    df_times = df_lck[df_lck['position'] == 'team'].copy()
    
    # NOVAS MÉTRICAS DE EARLY GAME, SANGUE E KILLS (V12)
    cols = ['gameid', 'date', 'teamname', 'side', 'result', 'patch', 'playoffs', 'dragons', 'gamelength', 'game', 'firstblood', 'ckpm', 'kills']
    
    tem_gd15 = 'golddiffat15' in df_times.columns
    if tem_gd15: cols.append('golddiffat15')

    df_limpo = df_times[cols].copy()
    
    # Limpeza de dados numéricos
    df_limpo['dragons'] = pd.to_numeric(df_limpo['dragons'], errors='coerce').fillna(0)
    df_limpo['kills'] = pd.to_numeric(df_limpo['kills'], errors='coerce').fillna(0)
    df_limpo['game'] = pd.to_numeric(df_limpo['game'], errors='coerce').fillna(1)
    df_limpo['firstblood'] = pd.to_numeric(df_limpo['firstblood'], errors='coerce').fillna(0)
    df_limpo['ckpm'] = pd.to_numeric(df_limpo['ckpm'], errors='coerce').fillna(0.7)
    if tem_gd15:
        df_limpo['golddiffat15'] = pd.to_numeric(df_limpo['golddiffat15'], errors='coerce').fillna(0)
    
    df_limpo['date'] = pd.to_datetime(df_limpo['date'], utc=True)
    
    # --- A MÁGICA DO MOMENTUM ---
    df_limpo['data_curta'] = df_limpo['date'].dt.date
    df_limpo = df_limpo.sort_values(['teamname', 'date'])
    df_limpo['momentum'] = df_limpo.groupby(['teamname', 'data_curta'])['result'].shift(1).fillna(0.5)
    
    # Dragões, Tempo e Kills
    df_limpo['total_dragons_partida'] = df_limpo.groupby('gameid')['dragons'].transform('sum')
    df_limpo['total_kills_partida'] = df_limpo.groupby('gameid')['kills'].transform('sum')
    
    df_limpo['opp_dragons'] = df_limpo['total_dragons_partida'] - df_limpo['dragons']
    df_limpo['over_4_dragons'] = (df_limpo['total_dragons_partida'] > 4).astype(int)
    df_limpo['mais_dragons'] = (df_limpo['dragons'] > df_limpo['opp_dragons']).astype(int)
    
    # Reordenar por data
    df_limpo = df_limpo.sort_values('date')
    
    df_limpo['patch'] = df_limpo['patch'].astype(str).str.extract(r'(\d+\.\d+)')[0]
    df_limpo['patch'] = df_limpo['patch'].str.replace('^16\.', '26.', regex=True)
    
    # MÉDIAS HISTÓRICAS (O ADN DAS EQUIPAS COM RECENCY BIAS)
    metricas_medias = ['result', 'mais_dragons', 'dragons', 'firstblood', 'ckpm']
    if tem_gd15: metricas_medias.append('golddiffat15')

    for col in metricas_medias:
        # Usando a média das últimas 10 partidas para focar no momento atual
        df_limpo[f'media_{col}'] = df_limpo.groupby('teamname')[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0.5)

    df_azul = df_limpo[df_limpo['side'] == 'Blue'].copy().add_suffix('_blue').rename(columns={'gameid_blue': 'gameid'})
    df_vermelho = df_limpo[df_limpo['side'] == 'Red'].copy().add_suffix('_red').rename(columns={'gameid_red': 'gameid'})
    df_p = pd.merge(df_azul, df_vermelho, on='gameid').dropna()

    features = ['media_result_blue', 'media_mais_dragons_blue', 'media_dragons_blue', 'media_firstblood_blue', 'media_ckpm_blue',
                'media_result_red', 'media_mais_dragons_red', 'media_dragons_red', 'media_firstblood_red', 'media_ckpm_red', 
                'playoffs_blue', 'game_blue', 'momentum_blue']
    
    X = df_p[features]
    
    # Treino dos Modelos Fixos
    m_vit = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(X, df_p['result_blue'])
    m_dra = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(X, df_p['mais_dragons_blue'])
    m_tot_dra = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(X, df_p['over_4_dragons_blue'])
    
    importancias = m_vit.feature_importances_
    nomes_variaveis = ['BL - WinRate', 'BL - +Dragões', 'BL - Média Drags', 'BL - FirstBlood%', 'BL - Banho de Sangue',
                       'RED - WinRate', 'RED - +Dragões', 'RED - Média Drags', 'RED - FirstBlood%', 'RED - Banho de Sangue', 
                       'MD5', 'Nº Mapa', 'Momentum']
    df_peso_ia = pd.DataFrame({'Peso na Decisão (%)': importancias * 100}, index=nomes_variaveis)
    
    lista_times = sorted(df_limpo['teamname'].unique())
    lista_patches = sorted(df_limpo['patch'].dropna().unique(), reverse=True)
    ultimos_dados = df_limpo.groupby(['teamname', 'patch']).last().reset_index()
    ultima_data = df_limpo['date'].max().strftime("%d/%m/%Y")
    
    return lista_times, lista_patches, ultimos_dados, m_vit, m_dra, m_tot_dra, df_peso_ia, X, df_p, ultima_data

# --- BOTÃO DE ATUALIZAÇÃO MANUAL ---
col_t, col_b = st.columns([3, 1])
with col_t:
    st.title("🏆 LoL Predictor PRO (v12 - Mercado de Kills)")
with col_b:
    st.write("")
    if st.button("🔄 Atualizar Cache de Dados e Radar", type="primary"):
        st.cache_resource.clear()
        st.cache_data.clear() # Limpa a memória do radar
        st.rerun()

with st.spinner("Conectando ao Oracle's Elixir e carregando Motor V12..."):
    times, patches, dados, m_vit, m_dra, m_tot_dra, df_peso_ia, X_historico, df_partidas, ultima_data_banco = treinar_motor_dinamico_v12()
    
# --- EXECUTA A BUSCA DE JOGOS AO VIVO ---
with st.spinner("O Cão Farejador está procurando jogos oficiais na PandaScore..."):
    jogos_pandascore = buscar_jogos_pandascore(pandascore_key)

# --- SELO DE ATUALIZAÇÃO ---
status_api = "🟢 Online" if pandascore_key else "🔴 Aguardando Chave API"
st.caption(f"✅ **Inteligência Ativa.** Última partida da LCK: **{ultima_data_banco}** | 🐼 **Radar PandaScore:** {status_api}")
st.markdown("---")

aba1, aba2, aba3 = st.tabs(["🤖 Previsões, Tendências & Radar", "💰 Gestão de Banca", "📊 Diário de Apostas"])

# --- ABA 1: PREVISÕES E RADAR DE ODDS ---
with aba1:
    col_p, col_m, col_map, col_win = st.columns(4)
    p_atual = col_p.selectbox("🛠️ Patch", patches)
    is_playoff = col_m.checkbox("⚠️ MD5 (Playoffs)?")
    num_mapa = col_map.selectbox("📍 Nº do Mapa", [1, 2, 3, 4, 5])
    
    res_anterior = 0.5 
    if num_mapa > 1:
        quem_ganhou = col_win.selectbox("Quem venceu o mapa anterior?", ["-", "🟦 Lado Azul", "🟥 Lado Vermelho"])
        if quem_ganhou == "🟦 Lado Azul":
            res_anterior = 1.0
        elif quem_ganhou == "🟥 Lado Vermelho":
            res_anterior = 0.0

    st.markdown("---")
    
    # 4 Colunas agora, incluindo a linha de Kills
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    t_azul = c1.selectbox("🟦 Lado Azul", times, index=0)
    t_red = c2.selectbox("🟥 Lado Vermelho", times, index=1)
    linha_tempo_casa = c3.number_input("⏱️ Linha de Tempo", min_value=20.0, max_value=50.0, value=32.5, step=0.5)
    linha_kills_casa = c4.number_input("🩸 Linha de Kills", min_value=15.5, max_value=45.5, value=28.5, step=0.5)
    
    if st.button("Analista Noturno: Prever Confronto e Farejar Jogos", type="primary"):
        if t_azul == t_red:
            st.error("Selecione times diferentes.")
        else:
            s_a = dados[(dados['teamname'] == t_azul)]
            s_r = dados[(dados['teamname'] == t_red)]
            
            if s_a.empty or s_r.empty:
                st.error("Faltam dados destas equipas neste patch.")
            else:
                s_a = s_a.iloc[[-1]]
                s_r = s_r.iloc[[-1]]
                
                # IA Recebe as 13 Informações
                input_ia = [[s_a['media_result'].values[0], s_a['media_mais_dragons'].values[0], s_a['media_dragons'].values[0], s_a['media_firstblood'].values[0], s_a['media_ckpm'].values[0],
                             s_r['media_result'].values[0], s_r['media_mais_dragons'].values[0], s_r['media_dragons'].values[0], s_r['media_firstblood'].values[0], s_r['media_ckpm'].values[0],
                             1 if is_playoff else 0, num_mapa, res_anterior]]
                
                # Treinamento Dinâmico de Tempo
                limite_segundos = linha_tempo_casa * 60
                df_partidas_local = df_partidas.copy()
                df_partidas_local['alvo_tempo'] = (df_partidas_local['gamelength_blue'] > limite_segundos).astype(int)
                m_tempo_dinamico = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_historico, df_partidas_local['alvo_tempo'])
                prob_t_dinamica = m_tempo_dinamico.predict_proba(input_ia)[0]
                
                # Treinamento Dinâmico de KILLS
                df_partidas_local['alvo_kills'] = (df_partidas_local['total_kills_partida_blue'] > linha_kills_casa).astype(int)
                m_kills_dinamico = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_historico, df_partidas_local['alvo_kills'])
                prob_k_dinamica = m_kills_dinamico.predict_proba(input_ia)[0]
                
                # --- O RADAR PANDASCORE EM AÇÃO ---
                jogo_confirmado = False
                detalhes_radar = ""
                odd_encontrada_a = None
                odd_encontrada_r = None
                
                def obter_apelidos(nome_time):
                    nome = nome_time.lower()
                    apelidos = [nome, nome.split()[0]]
                    if "brion" in nome: apelidos.extend(["brion", "bro", "oksavingsbank"])
                    if "soop" in nome: apelidos.extend(["soop", "freecs", "kwangdong"])
                    if "drx" in nome: apelidos.extend(["drx"])
                    if "t1" in nome: apelidos.extend(["t1", "skt"])
                    if "gen.g" in nome or "geng" in nome: apelidos.extend(["gen.g", "geng", "gen"])
                    if "hanwha" in nome or "hle" in nome: apelidos.extend(["hanwha", "hle"])
                    if "kt" in nome: apelidos.extend(["kt", "rolster"])
                    if "dplus" in nome or "dk" in nome: apelidos.extend(["dplus", "dk", "damwon"])
                    if "fearx" in nome or "fox" in nome: apelidos.extend(["fearx", "fox", "liv sandbox"])
                    if "nongshim" in nome or "ns" in nome: apelidos.extend(["nongshim", "redforce", "ns"])
                    return apelidos

                if isinstance(jogos_pandascore, list):
                    apelidos_a = obter_apelidos(t_azul)
                    apelidos_r = obter_apelidos(t_red)
                    
                    for partida in jogos_pandascore:
                        if not isinstance(partida, dict): continue
                        
                        ops = partida.get('opponents', [])
                        if len(ops) == 2:
                            t1 = ops[0].get('opponent', {}).get('name', '').lower()
                            t2 = ops[1].get('opponent', {}).get('name', '').lower()
                            
                            match_a = any(apelido in t1 or apelido in t2 for apelido in apelidos_a)
                            match_r = any(apelido in t1 or apelido in t2 for apelido in apelidos_r)
                            
                            if match_a and match_r:
                                jogo_confirmado = True
                                match_id = partida.get('id')
                                inicio = partida.get('begin_at', 'Horário não definido')
                                torneio = partida.get('league', {}).get('name', 'Torneio Oficial')
                                serie = partida.get('serie', {}).get('full_name', '')
                                detalhes_radar = f"🎮 **Competição:** {torneio} {serie}\n⏰ **Início Previsto:** {inicio[:10]} às {inicio[11:16]} (Horário UTC)"
                                
                                # NOVO: Ataque ao Servidor de Odds da PandaScore
                                odd_url = f"https://api.pandascore.co/matches/{match_id}/odds"
                                try:
                                    res_odds = requests.get(odd_url, headers={"Authorization": f"Bearer {pandascore_key}"}, timeout=5)
                                    if res_odds.status_code == 200:
                                        odds_data = res_odds.json()
                                        for provider in odds_data:
                                            markets = provider.get('markets', []) if isinstance(provider, dict) else []
                                            if not markets and 'choices' in provider:
                                                markets = [provider]
                                            for market in markets:
                                                nome_mercado = market.get('name', '').lower()
                                                if 'winner' in nome_mercado or 'match' in nome_mercado or 'moneyline' in nome_mercado or not nome_mercado:
                                                    for choice in market.get('choices', []):
                                                        c_name = str(choice.get('name', '')).lower()
                                                        val = choice.get('odd')
                                                        if val:
                                                            if any(ap in c_name for ap in apelidos_a):
                                                                odd_encontrada_a = float(val)
                                                            if any(ap in c_name for ap in apelidos_r):
                                                                odd_encontrada_r = float(val)
                                except:
                                    pass
                                break

                st.session_state['analise_salva'] = True
                st.session_state['dados_analise'] = {
                    't_azul': t_azul, 't_red': t_red, 'mapa': num_mapa, 'linha_t': linha_tempo_casa, 'linha_k': linha_kills_casa,
                    'prob_v': m_vit.predict_proba(input_ia)[0],
                    'prob_d': m_dra.predict_proba(input_ia)[0],
                    'prob_td': m_tot_dra.predict_proba(input_ia)[0],
                    'prob_t': prob_t_dinamica,
                    'prob_k': prob_k_dinamica,
                    'peso_ia': df_peso_ia,
                    's_a': s_a, 's_r': s_r,
                    'achou_jogo': jogo_confirmado,
                    'detalhes': detalhes_radar,
                    'odd_a': odd_encontrada_a,
                    'odd_r': odd_encontrada_r
                }

    if st.session_state['analise_salva']:
        mem = st.session_state['dados_analise']
        
        # --- PAINEL: CÃO FAREJADOR PANDASCORE ---
        if pandascore_key:
            st.markdown("### 🐼 Radar eSports PandaScore & Odds")
            
            if mem['achou_jogo']:
                if mem['odd_a'] and mem['odd_r']:
                    st.success(f"✅ **Jogo Oficial e Odds Detectadas!** O confronto é real e está mapeado para as próximas horas.\n\n{mem['detalhes']}")
                    st.info(f"💰 **Odds do Mercado Mundial:** {mem['t_azul']} ({mem['odd_a']}) vs {mem['t_red']} ({mem['odd_r']})")
                    
                    # Cálculo de +EV Automático
                    ev_a = ((mem['prob_v'][1] * mem['odd_a']) - 1) * 100
                    ev_r = ((mem['prob_v'][0] * mem['odd_r']) - 1) * 100
                    
                    rad1, rad2 = st.columns(2)
                    with rad1:
                        if ev_a > 0:
                            st.success(f"🔥 VANTAGEM MATEMÁTICA NA **{mem['t_azul']}** (+{ev_a:.2f}%)")
                        else:
                            st.error(f"🛑 {mem['t_azul']} sem valor (-EV)")
                    with rad2:
                        if ev_r > 0:
                            st.success(f"🔥 VANTAGEM MATEMÁTICA NA **{mem['t_red']}** (+{ev_r:.2f}%)")
                        else:
                            st.error(f"🛑 {mem['t_red']} sem valor (-EV)")
                else:
                    st.success(f"✅ **Jogo Oficial Detectado no Radar!** O confronto é real e está mapeado para as próximas horas.\n\n{mem['detalhes']}")
                    st.warning("⚠️ **Aviso de Bookmaker:** O jogo está mapeado, mas as casas de apostas globais ainda não liberaram os números exatos das Odds para a PandaScore. Fique de olho mais perto da hora do jogo ou digite a odd manual na Calculadora ao lado.")
            else:
                st.warning("⚠️ O jogo não foi encontrado na base global da PandaScore para as próximas horas. Verifique se as equipes realmente jogam em breve.")
            
            # --- MODO RAIO-X PANDASCORE ---
            with st.expander("🛠️ Modo Raio-X (Últimos Jogos Detectados)"):
                if isinstance(jogos_pandascore, list):
                    if len(jogos_pandascore) == 0:
                        st.write("A API não enviou NENHUM jogo futuro no momento.")
                    else:
                        st.write(f"A PandaScore listou {len(jogos_pandascore)} jogos oficiais (Exibindo os primeiros 10):")
                        for p in jogos_pandascore[:10]:
                            if isinstance(p, dict):
                                ops = p.get('opponents', [])
                                if len(ops) == 2:
                                    t1 = ops[0].get('opponent', {}).get('name', '?')
                                    t2 = ops[1].get('opponent', {}).get('name', '?')
                                    st.write(f"🎮 {t1} vs {t2} - *{p.get('league', {}).get('name', 'Torneio')}*")
                elif isinstance(jogos_pandascore, dict) and "erro" in jogos_pandascore:
                    st.error(f"Erro da API PandaScore: {jogos_pandascore['mensagem']}")
            st.markdown("---")

        # --- PAINEL DE TENDÊNCIAS (V8) ---
        st.markdown("### 🔮 Dicas Pré-Jogo (Leia na noite anterior!)")
        dica_col1, dica_col2 = st.columns(2)
        
        fb_a = mem['s_a']['media_firstblood'].values[0]
        fb_r = mem['s_r']['media_firstblood'].values[0]
        ckpm_comb = (mem['s_a']['media_ckpm'].values[0] + mem['s_r']['media_ckpm'].values[0]) / 2
        
        with dica_col1:
            st.info(f"**🔥 Early Game (First Blood):** \n\nA equipa **{mem['t_azul']}** tem {fb_a*100:.0f}% de taxa de FB recente, contra {fb_r*100:.0f}% de **{mem['t_red']}**. \n*Ideia:* Se a diferença for grande, foque no First Blood!")
        with dica_col2:
            if ckpm_comb > 0.8:
                st.warning(f"**🩸 Volatilidade (Mortes): ALTA**\n\nEstas duas equipas gostam muito de lutar. Tendência para o **Under Tempo** e Over Kills.")
            elif ckpm_comb < 0.65:
                st.success(f"**🛡️ Volatilidade (Mortes): BAIXA**\n\nEquipas que jogam pelo controlo. Fortíssima tendência para o **Over Tempo ({mem['linha_t']}+ min)**.")
            else:
                st.info(f"**⚖️ Volatilidade (Mortes): NORMAL**\n\nEstilo de jogo equilibrado. Foca a tua aposta no mercado de Dragões ou Vencedor.")

        st.markdown("---")
        st.success(f"Análise do Mapa {mem['mapa']} concluída!")
        
        st.subheader("🎯 Mercado Principal e Banho de Sangue")
        col_a, col_b, col_k_over, col_k_under = st.columns(4)
        col_a.metric(label=f"Vitória - {mem['t_azul']}", value=f"{mem['prob_v'][1]*100:.1f}%")
        col_b.metric(label=f"Vitória - {mem['t_red']}", value=f"{mem['prob_v'][0]*100:.1f}%")
        col_k_over.metric(label=f"Over {mem['linha_k']} Kills", value=f"{mem['prob_k'][1]*100:.1f}%")
        col_k_under.metric(label=f"Under {mem['linha_k']} Kills", value=f"{mem['prob_k'][0]*100:.1f}%")
        
        st.subheader("🐲 Mercado de Dragões")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Over 4.5 Dragões", f"{mem['prob_td'][1]*100:.1f}%")
        d2.metric("Under 4.5 Dragões", f"{mem['prob_td'][0]*100:.1f}%")
        d3.metric(f"Mais Dragões ({mem['t_azul']})", f"{mem['prob_d'][1]*100:.1f}%")
        d4.metric("Mais Dragões (Red/Empate)", f"{mem['prob_d'][0]*100:.1f}%")
        
        st.subheader("⏱️ Mercado de Tempo (Dinâmico)")
        t1, t2 = st.columns(2)
        t1.metric(f"Over {mem['linha_t']} min", f"{mem['prob_t'][1]*100:.1f}%")
        t2.metric(f"Under {mem['linha_t']} min", f"{mem['prob_t'][0]*100:.1f}%")

        st.markdown("---")
        st.subheader("🔍 Raio-X do Vencedor (Peso das Variáveis)")
        st.bar_chart(mem['peso_ia'])

# --- ABA 2: CALCULADORA KELLY ---
with aba2:
    st.subheader("Calculadora Avançada de Stake (Com Trava de Plataforma)")
    col1, col2 = st.columns(2)
    with col1:
        banca = st.number_input("Banca Atual (R$)", min_value=1.0, value=71.00, step=5.0)
        chance_ia = st.number_input("Probabilidade da IA (%)", min_value=1.0, max_value=99.0, value=55.0)
        odd_casa = st.number_input("Odd da Casa de Apostas", min_value=1.01, value=1.85, step=0.05)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        perfil_risco = st.selectbox("Seu Perfil de Risco (Fração de Kelly)", [
            "Conservador (1/8 - Muito Seguro)", "Recomendado (1/4 - Equilibrado)", 
            "Agressivo (1/2 - Maior Risco)", "Kamikaze (1/1 - NÃO RECOMENDADO)"
        ], index=1)
        aposta_minima = st.number_input("Aposta Mínima da Plataforma (R$)", min_value=0.10, value=0.50, step=0.10)

    fracoes = {
        "Conservador (1/8 - Muito Seguro)": 8, 
        "Recomendado (1/4 - Equilibrado)": 4, 
        "Agressivo (1/2 - Maior Risco)": 2, 
        "Kamikaze (1/1 - NÃO RECOMENDADO)": 1
    }
    divisor_kelly = fracoes[perfil_risco]

    p = chance_ia / 100
    b = odd_casa - 1
    ev_percentual = ((p * odd_casa) - 1) * 100

    st.markdown("---")
    if b > 0:
        f_star = (b * p - (1 - p)) / b
        if f_star > 0:
            aposta_sugerida = (banca * f_star) / divisor_kelly
            
            if aposta_sugerida < aposta_minima:
                st.warning(f"⚠️ **Aviso de Gestão:** A aposta ideal seria de **R$ {aposta_sugerida:.2f}**, que é menor que o piso da plataforma (R$ {aposta_minima:.2f}). Fique de fora ou mude o perfil de risco com cautela.")
            else:
                st.success("✅ **Aposta de Valor (+EV) Encontrada e Liberada!**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Vantagem (EV)", f"+{ev_percentual:.2f}%")
                m2.metric("Tamanho (Unidades)", f"{(aposta_sugerida / banca) * 100:.2f} U")
                m3.metric("Apostar em Dinheiro", f"R$ {aposta_sugerida:.2f}")
                
                st.markdown("---")
                st.write("📝 **Salvar no Diário (Manualmente)**")
                c_merc, c_nome = st.columns(2)
                mercado_nome = c_merc.text_input("Mercado (Ex: Over 4.5 Drags / T1 Ganha)")
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
                    st.toast("Aposta salva com sucesso!")
        else:
            st.error(f"🛑 **EV Negativo ({ev_percentual:.2f}%).**")

# --- ABA 3: DIÁRIO E SCANNER OCR ---
with aba3:
    st.subheader("📊 Seu Diário de Apostas")
    
    # --- SCANNER DE BILHETES ---
    st.markdown("### 📸 Scanner Automático de Bilhetes (OCR)")
    if not api_ativa:
        st.warning("⚠️ Chave do Gemini não encontrada nos Secrets do Streamlit.")
    else:
        st.info("💡 **Dica de Foco:** Para a IA não errar, a imagem tem de ser nítida. O ideal é tirar um Print da tela ou tirar a foto com a câmara normal do celular e depois fazer o Upload da Galeria.")
        
        modo_foto = st.radio("Como deseja enviar o bilhete?", ["📁 Upload da Galeria / Print", "📷 Tirar Foto Agora (Webcam/Celular)"])
        
        imagem_upload = None
        if modo_foto == "📁 Upload da Galeria / Print":
            imagem_upload = st.file_uploader("Carregue a imagem aqui", type=['png', 'jpg', 'jpeg'])
        else:
            imagem_upload = st.camera_input("Tire a foto (Espere focar bem!)")
        
        if imagem_upload is not None:
            imagem = Image.open(imagem_upload)
            st.image(imagem, caption="Bilhete detectado para leitura", width=250)
            
            if st.button("🪄 Ler Bilhete e Salvar na Planilha", type="primary"):
                with st.spinner("A analisar todas as apostas do print..."):
                    try:
                        modelo_ideal = 'gemini-1.5-flash'
                        modelo_visao = genai.GenerativeModel(modelo_ideal)
                        
                        prompt = """
                        És um analista de apostas experiente. Lê esta imagem e extrai TODAS as apostas individuais que encontrares.
                        Para cada aposta, identifica:
                        - Mercado: Ex: 'Mais de 4.5', 'Vencedor', 'Menos de 32.5 min'.
                        - Confronto: O nome do time ou o jogo. Se houver apenas um time (ex: 'Hanwha Life'), usa esse nome.
                        - Odd: O número decimal ao lado do mercado.
                        - Stake: O valor em R$ após a palavra 'Aposta'.
                        
                        Retorna obrigatoriamente uma LISTA de objetos JSON. Exemplo:
                        [
                          {"Mercado": "Vencedor", "Confronto": "Hanwha Life", "Odd": 2.50, "Stake": 6.44},
                          {"Mercado": "Mais de 4.5", "Confronto": "Desconhecido", "Odd": 1.72, "Stake": 1.45}
                        ]
                        Retorna apenas o JSON puro, sem textos adicionais.
                        """
                        
                        resposta = modelo_visao.generate_content([prompt, imagem])
                        texto_json = resposta.text.replace('```json', '').replace('```', '').strip()
                        lista_apostas = json.loads(texto_json)
                        
                        if not isinstance(lista_apostas, list):
                            lista_apostas = [lista_apostas]

                        novas_linhas = []
                        for dados_aposta in lista_apostas:
                            novas_linhas.append({
                                'Data': datetime.now().strftime("%d/%m/%Y"),
                                'Mercado': dados_aposta.get('Mercado', 'Automático'),
                                'Confronto': dados_aposta.get('Confronto', 'Automático'),
                                'Odd': float(dados_aposta.get('Odd', 0.0)),
                                'Stake (R$)': float(dados_aposta.get('Stake', 0.0)),
                                'Status': 'Pendente',
                                'Retorno (R$)': 0.0
                            })
                        
                        df_novos = pd.DataFrame(novas_linhas)
                        st.session_state['diario_apostas'] = pd.concat([st.session_state['diario_apostas'], df_novos], ignore_index=True)
                        st.success(f"✅ Sucesso! {len(novas_linhas)} apostas foram detectadas e adicionadas.")
                    except Exception as e:
                        st.error(f"Erro ao processar imagem: {e}")
                        
    st.markdown("---")
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
    
    col_lucro, col_botao = st.columns(2)
    with col_lucro:
        if lucro_total > 0:
            st.metric("Lucro Total (PNL)", f"R$ {lucro_total:.2f}", delta="Positivo")
        else:
            st.metric("Lucro Total (PNL)", f"R$ {lucro_total:.2f}", delta="Negativo", delta_color="inverse")
            
    with col_botao:
        st.write("") 
        csv = df_editado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar Backup do Diário (.csv)",
            data=csv,
            file_name=f"diario_apostas_{datetime.now().strftime('%d_%m_%Y')}.csv",
            mime="text/csv",
        )