import requests

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
    files = {
        "17xiC_Wrdv1noD9NZmgXGIZQWSnW0wnxP": "resnet50C2D.pth",
        "1XBMs4TLt2H378M_a0k23l8let-Ae2AlB": "resnet50I3D-NL.pth",
        "1psXLeYkrZYhqqCOGDym8XGBrRG8XyDHs": "resnet50RCN.pth",
        "1ZpbvJzvnDxJmKCFTs9wKmmA2qvm2aFBX": "resnet50I3D.pth",
        "1kHSu5PDd3LxkOFDL6dBGsxRySi7GVasi": "resnet50RCLSTM.pth",
        "1YgeXZk45V7F2chSyRK-uNWrBF5BskdVj": "resnet50RCGRU.pth",
        "1kQO_dnM9JjV3sBtvowXqQa6d0xcxn5rs": "SLOWFAST_R50_K400.pth.tar",
        "1qDdAntE5Onh7btniftOL8MrbsD7OIqj4": "SLOWFAST_R101_K700.pth.tar"
    }

    for file_id, file_name in files.items():
        download_file_from_google_drive(file_id, file_name)
        print(f"✅ Scaricato: {file_name}")

    print("✅ Download dei pesi completato!")
