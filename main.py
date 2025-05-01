# main.py

import subprocess
import os

def run_streamlit_app():
    app_path = "streamlit_app.py"
    if os.path.exists(app_path):
        print("🚀 Launching Streamlit app...")
        subprocess.run(["streamlit", "run", app_path])
    else:
        print(f"❌ {app_path} not found. Make sure the file exists.")

if __name__ == "__main__":
    run_streamlit_app()