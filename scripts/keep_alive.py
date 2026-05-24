import requests

url = "https://sales-watch.onrender.com"

try:
    response = requests.get(url)
    print("Pinged successfully:", response.status_code)
except Exception as e:
    print("Error:", e)