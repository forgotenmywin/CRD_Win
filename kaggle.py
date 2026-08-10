#!/usr/bin/env python3
import os, time, json, subprocess, urllib.request, re

# ← TAILSCALE_AUT (نه TAILSCALE_AUTH)
SSH_PASS = os.getenv("TAILSCALE_AUT", "kaggle123")

print("🚀 Starting Kaggle GPU + Cloudflare Tunnel...")

# 1. SSH
os.system("apt-get update -qq && apt-get install -y openssh-server")
os.system(f"echo 'root:{SSH_PASS}' | chpasswd")
os.system("mkdir -p /var/run/sshd && service ssh start")
print("✅ SSH ready")

# 2. cloudflared
os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared")
os.system("chmod +x cloudflared && mv cloudflared /usr/local/bin/")

# 3. Tunnel
proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "tcp://localhost:22"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

time.sleep(20)

# 4. گرفتن لینک
output = ""
for _ in range(50):
    line = proc.stdout.readline()
    if line:
        output += line
        if "trycloudflare.com" in line:
            break

match = re.search(r'([a-z0-9-]+\.trycloudflare\.com)', output)
if match:
    host = match.group(1)
    print("\n" + "="*60)
    print("🎯 KAGGLE GPU ONLINE!")
    print(f"🔗 Host: {host}")
    print(f"🔑 Password: {SSH_PASS}")
    print("="*60)
else:
    print("❌ Tunnel failed!")

# نگه داشتن
while True:
    time.sleep(60)
    print("💓 alive")
