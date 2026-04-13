import requests
import os

# URLs diretas dos arquivos no Oracle's Elixir
urls = {
    "2025_LoL_esports_match_data_from_OraclesElixir.csv": "https://oracleselixir-downloadable-files.s3-us-west-2.amazonaws.com/2025_LoL_esports_match_data_from_OraclesElixir.csv",
    "2026_LoL_esports_match_data_from_OraclesElixir.csv": "https://oracleselixir-downloadable-files.s3-us-west-2.amazonaws.com/2026_LoL_esports_match_data_from_OraclesElixir.csv"
}

def baixar_arquivos():
    print("🤖 Iniciando atualização automática dos dados...")
    for nome_arquivo, url in urls.items():
        print(f"⏳ Baixando {nome_arquivo}...")
        resposta = requests.get(url)
        if resposta.status_code == 200:
            with open(nome_arquivo, 'wb') as f:
                f.write(resposta.content)
            print(f"✅ {nome_arquivo} atualizado!")
        else:
            print(f"❌ Erro ao baixar {nome_arquivo}: Status {resposta.status_code}")

if __name__ == "__main__":
    baixar_arquivos()