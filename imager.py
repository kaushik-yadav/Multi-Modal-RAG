import requests


def upload_image_to_imgbb(local_file_path, imgbb_api_key, expiration_seconds=None):
    """
    Upload a local image to imgbb and get a public URL.

    Args:
        local_file_path (str): Path to the image file.
        imgbb_api_key (str): Your imgbb API key.
        expiration_seconds (int, optional): Auto-delete time in seconds (60-15552000). Default: None

    Returns:
        str: Public URL of the uploaded image (i.ibb.co)
    """
    try:
        url = f"https://api.imgbb.com/1/upload?key={imgbb_api_key}"
        if expiration_seconds:
            url += f"&expiration={expiration_seconds}"

        with open(local_file_path, "rb") as f:
            files = {"image": f}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data["data"]["url"]  # main image URL
            else:
                print(f"Upload failed: {data}")
                return None
        else:
            print(f"HTTP error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"Exception during upload: {e}")
        return None
