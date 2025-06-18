#!/usr/bin/env bash

# git clone https://github.com/charles-cai/MLX8-W2-DocumentSearch.git
# cd MLX8-W2-DocumentSearch

apt update
apt install -y vim rsync git git-lfs nvtop htop tmux curl btop

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# starship
curl -sS https://starship.rs/install.sh | sh
echo 'eval "$(starship init bash)"' >> ~/.bashrc

mkdir -p "~/.config"
cat > ~/.config/startship.toml <<EOF
[directory]
truncation_length = 3
truncate_to_repo = false
fish_style_pwd_dir_length = 1
home_symbol = "~"
EOF

# duckdb
curl https://install.duckdb.org | sh
echo "export PATH='/root/.duckdb/cli/latest':\$PATH" >> ~/.bashrc
source ~/.bashrc

# redis
apt install sudo -y
sudo apt-get install lsb-release curl gpg
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis

apt install locales -y
# To start (in GPU Docker, there's no systemd), and to ping
# /usr/bin/redis-server 
# redis-cli ping 

uv sync
# activate virtual environment for running python scripts
source .venv/bin/activate
echo "Setup complete - virtual environment activated. You can now run Python scripts directly."
echo "Run 'git lfs pull' to download large files."

which python
which uv

echo "Please don't forget to copy .env.example to .env in your work folder, and add API Keys"
