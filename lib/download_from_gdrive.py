import gdown

def download_file_from_google_drive(id, destination):
    url = f"https://drive.google.com/uc?id={id}"
    gdown.download(url, destination, quiet=False)
