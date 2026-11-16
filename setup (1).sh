#!/bin/bash
# One-time setup for the Multilingual Fake News Detector on an Ubuntu EC2 server.
# Run from the folder that contains app.py, requirements.txt and fake_news_classifier.joblib:
#   bash setup.sh
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

if [ ! -f fake_news_classifier.joblib ]; then
  echo "ERROR: fake_news_classifier.joblib is missing from $APP_DIR"; exit 1
fi

echo "== 1/5 System packages =="
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip

echo "== 2/5 Swap space (4 GB safety margin for model loading) =="
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
fi

echo "== 3/5 Hugging Face token (needed for the gated IndicNER model) =="
read -s -p "Paste your HF token (input hidden): " HF_TOKEN_INPUT
echo
cat > "$APP_DIR/.env" << ENVFILE
HF_TOKEN=$HF_TOKEN_INPUT
QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ENVFILE
chmod 600 "$APP_DIR/.env"

echo "== 4/5 Python packages (this takes several minutes) =="
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements.txt

echo "== 5/5 Start the app as a background service =="
sudo tee /etc/systemd/system/fakenews.service > /dev/null << SERVICE
[Unit]
Description=Multilingual Fake News Detector
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable --now fakenews

echo
echo "Done. The first start downloads the models and takes about 5-10 minutes."
echo "Watch progress with:  sudo journalctl -u fakenews -f"
echo "Then open:  http://<your-server-public-ip>:7860"
