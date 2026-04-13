import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

st.set_page_config(page_title="LoL Predictor PRO", layout="wide")

# --- MEMÓRIA DO DIÁRIO DE APOSTAS ---
# Isso cria uma tabela invisível quando você abre o site para guardar suas anotações
if 'diario_apostas' not in st.session_state:
    st.session_state['diario_apostas'] = pd.DataFrame(columns=[
        'Data', 'Confronto', 'Odd', 'Stake (R$)', 'Status', 'Retorno (R$)'
    ])

@st.cache_data
def treinar_modelo_super_cerebro():
    arquivos = ['2025_LoL_esports_match_data_from_OraclesElixir.csv', '2026_LoL_esports_match_data_from_OraclesElixir.csv']
    df_lol = pd.concat([pd.read_csv(arq, low_memory=False) for arq in arquivos], ignore_index=True)
    
    df_lck = df_lol[df_lol['league'] == 'LCK'].copy()
    df_times = df_lck[df_lck['position'] == 'team'].copy()
    
    df_limpo = df_times[['gameid', 'date', 'teamname', 'side', 'result', 'patch']].copy()
    df_limpo['date'] = pd.to_datetime(df_limpo['date'], utc=True)
    df_limpo = df_limpo.sort_values('date')
    df_limpo['patch'] = df_limpo['patch'].astype(str).str.extract(r'(\d+\.\d+)')[0]
    
    df_limpo['wr_geral'] = df_limpo.groupby('teamname')['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    df_limpo['wr_5_jogos'] = df_limpo.groupby('teamname')['result'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0.5)
    df_limpo['wr_lado'] = df_limpo.groupby(['teamname', 'side'])['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    df_limpo['wr_patch'] = df_limpo.groupby(['teamname', 'patch'])['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    
    df_azul = df_limpo[df_limpo['side'] == 'Blue'].copy().add_suffix('_blue').rename(columns={'gameid_blue': 'gameid'})
    df_vermelho = df_limpo[df_limpo['side'] == 'Red'].copy().add_suffix('_red').rename(columns={'gameid_red': 'gameid'})
    df_partidas = pd.merge(df_azul, df_vermelho, on='gameid').rename(columns={'result_blue': 'blue_win'}).dropna()
    
    X = df_partidas[['wr_geral_blue', 'wr_5_jogos_blue', 'wr_lado_blue', 'wr_patch_blue',
                     'wr_geral_red', 'wr_5_jogos_red', 'wr_lado_red', 'wr_patch_red']]
    y = df_partidas['blue_win']
    
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    modelo = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    modelo.fit(X_treino, y_treino)
    acerto = accuracy_score(y_teste, modelo.predict(X_teste)) * 100
    
    modelo.fit(X, y) # Treino final
    
    # Dicionário com a importância de cada variável para o gráfico
    importancias = modelo.feature_importances_
    nomes_variaveis = [
        'Blue - Histórico Geral', 'Blue - Momento (5 Jogos)', 'Blue - Força no Lado', 'Blue - Força no Patch',
        'Red - Histórico Geral', 'Red - Momento (5 Jogos)', 'Red - Força no Lado', 'Red - Força no Patch'
    ]
    df_importancia = pd.DataFrame({'Peso na Decisão (%)': importancias * 100}, index=nomes_variaveis)
    
    lista_times = sorted(df_limpo['teamname'].unique())
    lista_patches = sorted(df_limpo['patch'].dropna().unique(), reverse=True)
    ultimos_dados = df_limpo.groupby(['teamname', 'side', 'patch']).last().reset_index()
    
    return modelo, lista_times, lista_patches, ultimos_dados, acerto, df_importancia

with st.spinner("Conectando o Super Cérebro e gerando painéis visuais..."):
    modelo_ia, times_disponiveis, patches_disponiveis, dados_recentes, acuracia_modelo, df_peso_ia = treinar_modelo_super_cerebro()

# --- FRONT-END ---
st.title("🏆 LoL Predictor PRO (v4.0 - MasterMind)")
st.caption(f"🧠 Acurácia atual da IA: **{acuracia_modelo:.1f}%**")
st.markdown("---")

# AGORA TEMOS 3 ABAS!
aba1, aba2, aba3 = st.tabs(["🤖 Previsão & Raio-X", "💰 Gestão de Banca", "📊 Diário de Apostas"])

with aba1:
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
            stats_azul = dados_recentes[(dados_recentes['teamname'] == time_azul) & (dados_recentes['side'] == 'Blue') & (dados_recentes['patch'] == patch_atual)]
            stats_verm = dados_recentes[(dados_recentes['teamname'] == time_vermelho) & (dados_recentes['side'] == 'Red') & (dados_recentes['patch'] == patch_atual)]
            
            if stats_azul.empty or stats_verm.empty:
                st.warning("⚠️ Um dos times ainda não jogou neste Patch. Usando taxa base para adaptação ao meta.")
                wr_patch_a = 0.5 if stats_azul.empty else stats_azul['wr_patch'].values[0]
                wr_patch_v = 0.5 if stats_verm.empty else stats_verm['wr_patch'].values[0]
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
            
            # --- O NOVO RAIO-X DA IA ---
            st.markdown("---")
            st.subheader("🔍 Raio-X: Como a IA tomou essa decisão?")
            st.write("Este gráfico mostra o peso que a Inteligência Artificial dá para cada estatística na hora de calcular o favorito.")
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
            
            # Botão para enviar aposta para o Diário
            confronto_nome = st.text_input("Nome do Jogo para Salvar (Ex: T1 vs GenG)")
            if st.button("Salvar no Diário de Apostas"):
                nova_aposta = pd.DataFrame([{
                    'Data': datetime.now().strftime("%d/%m/%Y"),
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
    st.write("Dê dois cliques na coluna **Status** para mudar de 'Pendente' para 'Ganha' ou 'Perdida'. Dê dois cliques no **Retorno** para anotar o lucro/prejuízo exato.")
    
    # A MÁGICA: Uma tabela que você pode editar como se fosse o Excel!
    df_editado = st.data_editor(
        st.session_state['diario_apostas'],
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status da Aposta",
                help="O jogo já acabou?",
                options=["Pendente", "Ganha", "Perdida", "Reembolso"],
                required=True,
            ),
            "Retorno (R$)": st.column_config.NumberColumn(
                "Retorno (R$)",
                help="Se perdeu, coloque o valor negativo (ex: -45. Se ganhou lucro de 30, coloque 30)",
                format="R$ %.2f"
            )
        },
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True
    )
    
    # Salva as edições que você fez de volta na memória
    st.session_state['diario_apostas'] = df_editado
    
    # Calcula e mostra o seu Lucro Total
    lucro_total = df_editado['Retorno (R$)'].sum()
    st.markdown("---")
    if lucro_total > 0:
        st.metric("Lucro Total (PNL)", f"R$ {lucro_total:.2f}", delta="Positivo")
    else:
        st.metric("Lucro Total (PNL)", f"R$ {lucro_total:.2f}", delta="Negativo", delta_color="inverse")