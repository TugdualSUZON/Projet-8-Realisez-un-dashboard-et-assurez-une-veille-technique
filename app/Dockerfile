FROM ubuntu/python:3.13-22.04@sha256:e535cf7b1bc84127d022a2d5b6be4ebeb9624edbf3281147f997def4e7468067

# Mettre a jours de système et installer les packet
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libssl-dev \
        libffi-dev \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# récupération du Model, requirement and main.py et le dockerfile
# Mettre les fichier dans répertoire temporaire pour éviter les problèmes d'accès
COPY ./requirements_main_main.txt /tmp/requirements_main.txt
     ./mlflow_to_deploy /temp/mlflow_to_deploy

#   - `--no-cache-dir` évite de laisser le cache dans l’image
#   - `--disable-pip-version-check` accélère le build
RUN pip install --no-cache-dir \
    --disable-pip-version-check \
    -r /tmp/requirements_main.txt
 
#Install pyenv for mlflow API REST https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix

## Geting Pyenv
RUN curl -fsSL https://pyenv.run | bash
        ## Set up your shell environnement for Pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
        ## Install python build dependency
RUN sudo apt update; sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

## Instaler la version de python pour pyenv
CMD ["pyenv", "install", "3.13.7"]

# Lancer l'API REST du model
CMD ["python", "mlflow", "models", "serve", "-m", "./mlflow_to_deploy"] # ADD path to file

# Lancer l'iinterface graphique pour intéroger l'API
# CMD ["streamlit", "run", "P7_dashboard.py"]