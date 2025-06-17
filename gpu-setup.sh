# git clone https://github.com/charles-cai/MLX8-W2-DocumentSearch.git
# cd MLX8-W2-DocumentSearch.git

#!/usr/bin/env bash
apt update
# ensure we have all the utils we need
apt install -y vim rsync git git-lfs nvtop htop tmux curl
# apt upgrade -y
# install uv and sync
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync
# activate virtual environment for running python scripts
source .venv/bin/activate
echo "Setup complete - virtual environment activated. You can now run Python scripts directly."
echo "Run 'git lfs pull' to download large files."
