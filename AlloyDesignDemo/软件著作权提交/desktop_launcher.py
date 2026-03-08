import sys
import time
import socket
import subprocess
from pathlib import Path
import webview

PORT = 8503
HOST = "127.0.0.1"


def resource_path(rel: str) -> Path:
    meipass = Path(getattr(sys, "_MEIPASS", ""))
    if meipass:
        p = meipass / rel
        if p.exists():
            return p
    exe_dir = Path(sys.executable).resolve().parent
    p2 = exe_dir / rel
    if p2.exists():
        return p2
    return Path(__file__).parent / rel


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def wait_port(host: str, port: int, timeout: int = 25):
    start = time.time()
    while time.time() - start < timeout:
        if port_open(host, port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"等待 {host}:{port} 超时，Streamlit 可能未启动成功。")


def run_streamlit_server():
    app_path = resource_path("app-3.py")

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        f"--server.port={PORT}",
        f"--server.address={HOST}",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
    ]

    from streamlit.web.cli import main as st_main
    st_main()


def main_gui():
    if port_open(HOST, PORT):
        url = f"http://{HOST}:{PORT}"
        webview.create_window("铝合金智能设计软件", url, width=1200, height=800)
        webview.start()
        return

    cmd = [sys.executable, "--server"]
    proc = subprocess.Popen(cmd)

    try:
        wait_port(HOST, PORT, timeout=30)
        url = f"http://{HOST}:{PORT}"
        webview.create_window("铝合金智能设计软件", url, width=1200, height=800)
        webview.start()
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    if "--server" in sys.argv:
        run_streamlit_server()
    else:
        main_gui()