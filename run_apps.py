import subprocess
import time
import os
import webbrowser

BASE_DIR = r"C:\Users\user\github\sukon"

apps = [
    {"file": os.path.join(BASE_DIR, "app.py"), "port": 8501},
    {"file": os.path.join(BASE_DIR, "app2.py"), "port": 8502}
]

processes = []

for app in apps:
    print(f"กำลังรัน {app['file']} ที่พอร์ต {app['port']} ...")
    cmd = [
        "streamlit",
        "run",
        app["file"],
        "--server.address=0.0.0.0",
        f"--server.port={app['port']}"
    ]
    process = subprocess.Popen(cmd)
    processes.append(process)
    time.sleep(2)  # ให้เวลาแอปเริ่มต้นก่อนเปิดเบราว์เซอร์
    webbrowser.open(f"http://localhost:{app['port']}")

print("✅ รันแอปทั้งหมดเรียบร้อยแล้ว!")

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\n⛔ ตรวจพบการกด Ctrl+C: ปิดแอปทั้งหมด...")
    for p in processes:
        p.terminate()
