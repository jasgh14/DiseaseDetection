python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "If you have an NVIDIA GPU, install the CUDA build of torch from https://pytorch.org/get-started/locally/"
