#!/usr/bin/env bash

# git clone https://github.com/charles-cai/MLX8-W2-DocumentSearch.git
# cd MLX8-W2-DocumentSearch.git

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

uv sync
# activate virtual environment for running python scripts
source .venv/bin/activate
echo "Setup complete - virtual environment activated. You can now run Python scripts directly."
echo "Run 'git lfs pull' to download large files."

which python
which uv