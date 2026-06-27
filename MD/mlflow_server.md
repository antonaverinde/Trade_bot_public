# Running MLflow UI from WSL with a Windows Browser

No SSH needed. WSL2 forwards `localhost` to Windows automatically on modern Windows 10/11,
so the browser just works — as long as you use the right address.

---

## Quick Start (works in 99% of cases)

Open your WSL terminal and run:

```bash
cd ~/Trade_bot
source /home/anton/Trade_bot/.venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow///mlflow.db --host 0.0.0.0 --port 5000
```

Then open this in your **Windows browser**:

```
http://localhost:5000
```

The `--host 0.0.0.0` flag makes MLflow listen on all interfaces, which is required for
Windows to reach it through the WSL2 network bridge.

Leave the terminal open while you browse — closing it stops the server.
To stop the server: `Ctrl+C` in the terminal.

---

## If `localhost:5000` Does Not Open

WSL2 localhost forwarding can occasionally fail (older Windows builds, VPN, firewall).
In that case use the WSL2 internal IP address instead.

**Step 1** — find the WSL2 IP:
```bash
hostname -I | awk '{print $1}'
```
Example output: `172.26.144.5`

**Step 2** — open that IP in your Windows browser:
```
http://172.26.144.5:5000
```

> Note: this IP changes every time WSL restarts. `localhost` is always stable, so fix the
> root cause (see below) if you rely on this workaround regularly.

### Fix localhost forwarding permanently

Run this in **PowerShell as Administrator** on Windows (one-time):

```powershell
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=127.0.0.1 connectport=5000 connectaddress=$(wsl hostname -I | ForEach-Object { $_.Trim().Split(' ')[0] })
```

Or the simpler fix: ensure your Windows build is 21H2 or later — localhost forwarding
has been reliable since that release.

---

## Do You Need SSH?

**No.** SSH port forwarding (`ssh -L`) is a technique for tunnelling through a remote server.
WSL2 runs locally on your machine, so Windows and WSL share the same physical host —
no tunnelling required.

The only scenario where SSH would help is if you were running MLflow on a **remote Linux
machine** (cloud VM, HPC cluster) and wanted to access its UI locally. That is not this case.

---

## Port Already in Use?

If port 5000 is taken (common on macOS/Windows where AirPlay or other services grab it):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001
```

Then open `http://localhost:5001`.

---

## Full Command Reference

```bash
# Standard — run from Trade_bot project root
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 \
  --port 5000

# With explicit project path (run from anywhere)
mlflow ui \
  --backend-store-uri sqlite:////home/anton/Trade_bot/mlflow.db \
  --host 0.0.0.0 \
  --port 5000
```

Access: `http://localhost:5000`
