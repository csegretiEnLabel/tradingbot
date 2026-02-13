import subprocess
import time
import sys
import os
import signal

def kill_process_on_port(port):
    """Find and kill process listening on a specific port (Windows)."""
    try:
        # Find PID
        cmd = f"netstat -ano | findstr :{port}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    try:
                        pid_int = int(pid)
                        if pid_int > 0:
                            print(f"[CLEAN] Killing zombie process {pid} on port {port}...")
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[WARN] Failed to cleanup port {port}: {e}")

def run_command(cmd, cwd=None, name="Process"):
    print(f"[START] Starting {name}...")
    # Use the local .venv python if it exists
    python_exe = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    cmd = cmd.replace("python ", f'"{python_exe}" ')
    
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',  # Keep utf-8 just in case, but input should now be cleaner
        bufsize=1,
        universal_newlines=True
    )

def main():
    print("=== AI TRADING BOT UNIFIED LAUNCHER ===")
    
    # 0. Cleanup ports
    print("[CLEAN] Cleaning up ports...")
    kill_process_on_port(8000)  # API
    kill_process_on_port(5173)  # Dashboard Frontend
    time.sleep(1)

    # 1. Start API Backend
    api_process = run_command("python api.py", name="FastAPI Backend")
    time.sleep(2)  # Wait for API to warm up
    
    # 2. Start Dashboard Frontend
    dashboard_process = run_command("npm run dev", cwd="dashboard", name="Vite Dashboard")
    
    # 3. Start the Trading Bot (main loop)
    # Using --status first to verify account
    print("[INFO] Verifying account status...")
    subprocess.run("python main.py --status", shell=True)
    
    print("\n[INFO] Bot Command Center ready at http://localhost:5173")
    print("[INFO] API Backend running at http://localhost:8000")
    print("Press CTRL+C to shut down all components.\n")
    
    bot_process = run_command("python main.py", name="Trading Bot")

    try:
        while True:
            # Check if processes are still running
            if api_process.poll() is not None:
                print("[ERROR] API Backend stopped unexpectedly.")
                break
            if dashboard_process.poll() is not None:
                print("[ERROR] Dashboard stopped unexpectedly.")
                break
            if bot_process.poll() is not None:
                print("[ERROR] Trading Bot stopped unexpectedly.")
                break
                
            # Optional: Stream bot output to console
            line = bot_process.stdout.readline()
            if line:
                print(f"[BOT] {line.strip()}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down gracefully...")
    finally:
        api_process.terminate()
        dashboard_process.terminate()
        bot_process.terminate()
        print("[STOP] All processes stopped.")

if __name__ == "__main__":
    main()
