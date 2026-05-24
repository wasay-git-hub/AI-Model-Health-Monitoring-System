import requests
import time
from datetime import datetime

# Configure your Render URL here or via environment variable
RENDER_URL = "https://sales-watch.onrender.com/"

def ping_server():
    if not RENDER_URL or "REPLACE_WITH" in RENDER_URL:
        print("Error: RENDER_BACKEND_URL not set. Please update the script or environment variable.")
        return

    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pinging {RENDER_URL}...")
        response = requests.get(RENDER_URL, timeout=30)
        if response.status_code == 200:
            print(f"Success: Received {response.json()}")
        else:
            print(f"Warning: Received status code {response.status_code}")
    except Exception as e:
        print(f"Error: Could not connect to server. {e}")

if __name__ == "__main__":
    print("Keep-Alive script started. Press Ctrl+C to stop.")
    # Ping once immediately
    ping_server()
    
    # Ping every 10 minutes (600 seconds)
    # Render free tier sleeps after 15 mins, so 10 mins is a safe interval
    while True:
        time.sleep(600)
        ping_server()