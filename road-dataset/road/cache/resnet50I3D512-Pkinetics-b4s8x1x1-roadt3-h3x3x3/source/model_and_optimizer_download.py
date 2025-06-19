import requests
import os

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    print(f"✅ Scaricato: {destination}")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    files_to_download = {
        "1zA055aBTWN_9f9DsCsiCjpCR7eCrLMuj": "model_000030.pth",
        "1cMbc-S9SUJyH4enb0XA2Sj-t3MmlFwRA": "optimizer_000030.pth",
    }

    for file_id, filename in files_to_download.items():
        download_file_from_google_drive(file_id, filename)

    print("\n🎉 Download completato per tutti i file!")
