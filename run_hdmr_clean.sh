#!/bin/bash

SESSION_NAME="hdmr_full_run"
PROJECT_DIR="$HOME/hdmr-opt"
VENV_PATH="$PROJECT_DIR/hdmr-env"

echo "=========================================="
echo "HDMR Deneyleri - Temiz Başlatma"
echo "=========================================="

# Varsa eski hdmr oturumunu kapat
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Yeni tmux oturumu
tmux new-session -d -s $SESSION_NAME

# İlk komut: Environment'ı temizle
tmux send-keys -t $SESSION_NAME "unset VIRTUAL_ENV" C-m
tmux send-keys -t $SESSION_NAME "export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin" C-m

# DevToolset yükle
tmux send-keys -t $SESSION_NAME "source /opt/rh/devtoolset-9/enable" C-m

# Proje dizinine git
tmux send-keys -t $SESSION_NAME "cd $PROJECT_DIR" C-m

# Sadece hdmr-env aktive et
tmux send-keys -t $SESSION_NAME "source $VENV_PATH/bin/activate" C-m

# Kontrol göster
tmux send-keys -t $SESSION_NAME "echo '=== Ortam Kontrolü ==='" C-m
tmux send-keys -t $SESSION_NAME "echo 'VIRTUAL_ENV: '\$VIRTUAL_ENV" C-m
tmux send-keys -t $SESSION_NAME "which python" C-m
tmux send-keys -t $SESSION_NAME "python --version" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# Paket kontrolü
tmux send-keys -t $SESSION_NAME "echo '=== Paket Kontrolü ==='" C-m
tmux send-keys -t $SESSION_NAME "python -c 'import numpy, pandas, torch, xgboost; print(\"✓ Tüm paketler yüklü\")' || echo '✗ Paket eksik!'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# Deneyleri başlat
LOG_FILE="experiments_$(date +%Y%m%d_%H%M%S).log"
tmux send-keys -t $SESSION_NAME "echo '=== Deneyler Başlıyor ==='" C-m
tmux send-keys -t $SESSION_NAME "echo 'Log dosyası: $LOG_FILE'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "python run_all_experiments.py --full 2>&1 | tee $LOG_FILE" C-m

echo ""
echo "✓ Deneyler temiz ortamda başlatıldı!"
echo ""
echo "Oturum adı: $SESSION_NAME"
echo "Log dosyası: $LOG_FILE"
echo ""
echo "Durumu görmek için:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "Log'u takip etmek için:"
echo "  tail -f $PROJECT_DIR/$LOG_FILE"
echo ""
echo "Diğer oturumlar:"
echo "  tmux ls"
echo ""
echo "Ayrılmak için: Ctrl+B sonra D"
echo "=========================================="
