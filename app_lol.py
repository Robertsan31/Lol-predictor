import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
from PIL import Image

# --- IMPORTAÇÕES NOVAS PARA A IA LER IMAGENS ---
import google.generativeai as genai

st.set_page_config(page_title="LoL Predictor PRO", layout="wide")

# --- CONECTANDO O CÉREBRO DO GEMINI (SCANNER) ---
api_ativa = False
if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        api_ativa = True
    except Exception as e:
        st.error(f"Erro ao configurar IA de visão: {e}")
else:
    print("Atenção: GEMINI_API_KEY não encontrada nos Secrets do Streamlit.")

# --- MEMÓRIA DO DIÁRIO DE APOSTAS E DA ANÁLISE ---
if 'diario_apostas' not in st.session_state:
    st.session_state['diario_apostas'] = pd.DataFrame(columns=[
        'Data', 'Mercado', 'Confronto', 'Odd', 'Stake (R$)', 'Status', 'Retorno (R$)'
    ])
if 'analise_salva' not in st.session_state:
    st.session_state['analise_salva'] = False

# --- MOTOR DA INTELIGÊNCIA ARTIFICIAL ---
@st.cache_resource 
def treinar_motor_dinamico_v5():
    arquivos = ['2025_LoL_esports_match_data_from_OraclesElixir.csv', '2026_LoL_esports_match_data_from_OraclesElixir.csv']
    df_lol = pd.concat([pd.read_csv(arq, low_memory=False) for arq in arquivos], ignore_index=True)
    
    df_lck = df_lol[df_lol['league'] == 'LCK'].copy()
    df_times = df_lck[df_lck['position'] == 'team'].copy()
    
    cols = ['gameid', 'date', 'teamname', 'side', 'result', 'patch', 'playoffs', 'dragons', 'gamelength', 'game']
    df_limpo = df_times[cols].copy()
    
    df_limpo['dragons'] = pd.to_numeric(df_limpo['dragons'], errors='coerce').fillna(0)
    df_limpo['game'] = pd.to_numeric(df_limpo['game'], errors='coerce').fillna(1)
    
    df_limpo['total_dragons_partida'] = df_limpo.groupby('gameid')['dragons'].transform('sum')
    df_limpo['opp_dragons'] = df_limpo['total_dragons_partida'] - df_limpo['dragons']
    
    df_limpo['over_4_dragons'] = (df_limpo['total_dragons_partida'] > 4).astype(int)
    df_limpo['mais_dragons'] = (df_limpo['dragons'] > df_limpo['opp_dragons']).astype(int)
    
    df_limpo['date'] = pd.to_datetime(df_limpo['date'], utc=True)
    df_limpo = df_limpo.sort_values('date')
    
    df_limpo['patch'] = df_limpo['patch'].astype(str).str.extract(r'(\d+\.\d+)')[0]
    df_limpo['patch'] = df_limpo['patch'].str.replace('^16\.', '26.', regex=True)
    
    for col in ['result', 'mais_dragons', 'dragons']:
        df_limpo[f'media_{col}'] = df_limpo.groupby('teamname')[col].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)

    df_azul = df_limpo[df_limpo['side'] == 'Blue'].copy().add_suffix('_blue').rename(columns={'gameid_blue': 'gameid'})
    df_vermelho = df_limpo[df_limpo['side'] == 'Red'].copy().add_suffix('_red').rename(columns={'gameid_red': 'gameid'})
    df_p = pd.merge(df_azul, df_vermelho, on='gameid').dropna()

    features = ['media_result_blue', 'media_mais_dragons_blue', 'media_dragons_blue',
                'media_result_red', 'media_mais_dragons_red', 'media_dragons_red', 'playoffs_blue', 'game_blue']
    X = df_p[features]
    
    m_vit = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df_p['result_blue'])
    m_dra = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df_p['mais_dragons_blue'])
    m_tot_dra = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df_p['over_4_dragons_blue'])
    
    importancias = m_vit.feature_importances_
    nomes_variaveis = ['Blue - Win Rate', 'Blue - +Dragões', 'Blue - Média Dragões',
                       'Red - Win Rate', 'Red - +Dragões', 'Red - Média Dragões', 'Fator MD5/Playoff', 'Número do Mapa']
    df_peso_ia = pd.DataFrame({'Peso na Decisão (%)': importancias * 100}, index=nomes_variaveis)
    
    lista_times = sorted(df_limpo['teamname'].unique())
    lista_patches = sorted(df_limpo['patch'].dropna().unique(), reverse=True)
    ultimos_dados = df_limpo.groupby(['teamname', 'patch']).last().reset_index()
    
    return lista_times, lista_patches, ultimos_dados, m_vit, m_dra, m_tot_dra, df_peso_ia, X, df_p

with st.spinner("Construindo o novo Motor Dinâmico e Conectando Scanner..."):
    times, patches, dados, m_vit, m_dra, m_tot_dra, df_peso_ia, X_historico, df_partidas = treinar_motor_dinamico_v5()

# --- FRONT-END ---
st.title("🏆 LoL Predictor PRO (v5.3 - Vision Engine)")
st.markdown("---")

aba1, aba2, aba3 = st.tabs(["🤖 Previsões & Raio-X", "💰 Gestão de Banca", "📊 Diário de Apostas"])

# --- ABA 1: PREVISÕES ---
with aba1:
    col_p, col_m, col_map, col_t = st.columns(4)
    p_atual = col_p.selectbox("🛠️ Patch", patches)
    is_playoff = col_m.checkbox("⚠️ MD5 (Playoffs)?")
    num_mapa = col_map.selectbox("📍 Nº do Mapa", [1, 2, 3, 4, 5])
    linha_tempo_casa = col_t.number_input("⏱️ Linha de Tempo (Min)", min_value=20.0, max_value=50.0, value=32.5, step=0.5)
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
                         1 if is_playoff else 0, num_mapa]]
            
            # Treinamento Dinâmico de Tempo
            limite_segundos = linha_tempo_casa * 60
            df_partidas_local = df_partidas.copy()
            df_partidas_local['alvo_tempo'] = (df_partidas_local['gamelength_blue'] > limite_segundos).astype(int)
            m_tempo_dinamico = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_historico, df_partidas_local['alvo_tempo'])
            prob_t_dinamica = m_tempo_dinamico.predict_proba(input_ia)[0]
            
            st.session_state['analise_salva'] = True
            st.session_state['dados_analise'] = {
                't_azul': t_azul, 't_red': t_red, 'mapa': num_mapa, 'linha_t': linha_tempo_casa,
                'prob_v': m_vit.predict_proba(input_ia)[0],
                'prob_d': m_dra.predict_proba(input_ia)[0],
                'prob_td': m_tot_dra.predict_proba(input_ia)[0],
                'prob_t': prob_t_dinamica,
                'peso_ia': df_peso_ia
            }

    if st.session_state['analise_salva']:
        mem = st.session_state['dados_analise']
        st.success(f"Análise do Mapa {mem['mapa']} concluída!")
        
        st.subheader("🎯 Mercado Principal (Vencedor)")
        col_a, col_b = st.columns(2)
        col_a.metric(label=f"Vitória - {mem['t_azul']}", value=f"{mem['prob_v'][1]*100:.1f}%")
        col_b.metric(label=f"Vitória - {mem['t_red']}", value=f"{mem['prob_v'][0]*100:.1f}%")
        
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
        st.subheader("🔍 Raio-X do Vencedor")
        st.bar_chart(mem['peso_ia'])

# --- ABA 2: CALCULADORA KELLY ---
with aba2:
    st.subheader("Calculadora Avançada de Stake (Com Trava de Plataforma)")
    col1, col2 = st.columns(2)
    with col1:
        banca = st.number_input("Banca Atual (R$)", min_value=1.0, value=100.0, step=10.0)
        chance_ia = st.number_input("Probabilidade da IA (%)", min_value=1.0, max_value=99.0, value=55.0)
        odd_casa = st.number_input("Odd da Casa de Apostas", min_value=1.01, value=1.85, step=0.05)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        perfil_risco = st.selectbox("Seu Perfil de Risco (Fração de Kelly)", [
            "Conservador (1/8 - Muito Seguro)", "Recomendado (1/4 - Equilibrado)", 
            "Agressivo (1/2 - Maior Risco)", "Kamikaze (1/1 - NÃO RECOMENDADO)"
        ], index=1)
        aposta_minima = st.number_input("Aposta Mínima da Plataforma (R$)", min_value=0.10, value=0.50, step=0.10)

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
    st.subheader("📊 O Seu Diário de Apostas")
    
    st.markdown("### 📸 Scanner Automático de Bilhetes (OCR)")
    if not api_ativa:
        st.warning("⚠️ Chave do Gemini não encontrada nos Secrets do Streamlit. Vá as definições da sua app na nuvem e adicione a GEMINI_API_KEY para libertar o scanner.")
    else:
        imagem_upload = st.file_uploader("Arraste ou carregue o ecrã (print) da sua aposta aqui", type=['png', 'jpg', 'jpeg'])
        
        if imagem_upload is not None:
            imagem = Image.open(imagem_upload)
            st.image(imagem, caption="Bilhete detetado pela IA", width=250)
            
            if st.button("🪄 Ler Bilhete e Salvar na Planilha", type="primary"):
                with st.spinner("A IA está a dissecar o talão..."):
                    try:
                        modelo_visao = genai.GenerativeModel('gemini-1.5-flash')
                        prompt = """
                        És um assistente de extração de dados. Lê esta imagem de um bilhete de aposta desportiva.
                        Extrai a informação e devolve APENAS um formato JSON exato. Não escrevas mais nada.
                        Estrutura obrigatória:
                        {"Mercado": "Texto", "Confronto": "Texto", "Odd": numero_decimal, "Stake": numero_decimal}
                        Se não encontrares algo, coloca 0.0 para números ou 'Desconhecido' para textos.
                        """
                        resposta = modelo_visao.generate_content([prompt, imagem])
                        
                        texto_json = resposta.text.replace('```json', '').replace('```', '').strip()
                        dados = json.loads(texto_json)
                        
                        nova_aposta = pd.DataFrame([{
                            'Data': datetime.now().strftime("%d/%m/%Y"),
                            'Mercado': dados.get('Mercado', 'Automático'),
                            'Confronto': dados.get('Confronto', 'Automático'),
                            'Odd': float(dados.get('Odd', 0.0)),
                            'Stake (R$)': float(dados.get('Stake', 0.0)),
                            'Status': 'Pendente',
                            'Retorno (R$)': 0.0
                        }])
                        st.session_state['diario_apostas'] = pd.concat([st.session_state['diario_apostas'], nova_aposta], ignore_index=True)
                        st.success("✅ Bilhete lido com sucesso e adicionado à tabela abaixo!")
                    except Exception as e:
                        st.error(f"Não foi possível ler o bilhete com clareza. Tente um print com melhor resolução. Erro técnico: {e}")
    
    st.markdown("---")
    st.write("Dê dois cliques na coluna **Status** para mudar de 'Pendente' para 'Ganha' ou 'Perdida'. Dê dois cliques em **Retorno** para adicionar o lucro final.")
    
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