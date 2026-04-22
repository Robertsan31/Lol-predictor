import requests
import os
import sys

# URLs diretas dos arquivos no Oracle's Elixir
urls = {
    "2025_LoL_esports_match_data_from_OraclesElixir.csv": "https://oracleselixir-downloadable-files.s3-us-west-2.amazonaws.com/2025_LoL_esports_match_data_from_OraclesElixir.csv",
    "2026_LoL_esports_match_data_from_OraclesElixir.csv": "https://oracleselixir-downloadable-files.s3-us-west-2.amazonaws.com/2026_LoL_esports_match_data_from_OraclesElixir.csv"
}

# --- A MÁSCARA TURBINADA (Com Crachá de Visitante) ---
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://oracleselixir.com/", # O "Crachá" que engana a Amazon
    "Connection": "keep-alive"
}

def baixar_arquivos():
    print("🤖 Iniciando atualização automática dos dados...")
    sucesso_total = True
    
    for nome_arquivo, url in urls.items():
        print(f"⏳ Baixando {nome_arquivo}...")
        try:
            # Faz o download usando a máscara e um tempo limite de 30 segundos
            resposta = requests.get(url, headers=headers, timeout=30)
            
            if resposta.status_code == 200:
                with open(nome_arquivo, 'wb') as f:
                    f.write(resposta.content)
                print(f"✅ {nome_arquivo} atualizado com sucesso!")
            else:
                print(f"❌ O servidor recusou o download de {nome_arquivo}. Status Code: {resposta.status_code}")
                sucesso_total = False
        except Exception as e:
            print(f"❌ Erro crítico de conexão ao tentar baixar {nome_arquivo}: {e}")
            sucesso_total = False

    # --- O ALARME DE INCÊNDIO ---
    # Se algum download falhou, nós "matamos" o processo (sys.exit(1))
    # Isso obriga o GitHub Actions a mostrar um X Vermelho na aba Actions!
    if not sucesso_total:
        print("🛑 Automação falhou. Acionando alerta vermelho no GitHub!")
        sys.exit(1)

if __name__ == "__main__":
    baixar_arquivos()