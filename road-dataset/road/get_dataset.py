import requests
import zipfile
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

# --- MAIN ---
if __name__ == "__main__":
    # Video dataset
    download_file_from_google_drive("1YQ9ap3o9pqbD0Pyei68rlaMDcRpUn-qz", "videos.zip")
    with zipfile.ZipFile("videos.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove("videos.zip")

    # JSON files
    download_file_from_google_drive("1HAJpdS76TVK56Qvq1jXr-5hfFCXKHRZo", "road_trainval_v1.0.json")
    download_file_from_google_drive("1NfSoI1yVTA46YY7AwVIGRolAqtWfoa8V", "instance_counts.json")

    print("✅ Dataset scaricato e estratto!")
