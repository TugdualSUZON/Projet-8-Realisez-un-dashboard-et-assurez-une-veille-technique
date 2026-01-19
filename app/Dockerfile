# Definir l'image de base pour le docker
FROM ubuntu:resolute-20260106.1@sha256:f3218e7a90a09be6fa17dfba992ce84391701cd9a6fb0b2e7e084eb276da4e7a

# Passer a un utilisateur root qui marche sans fichier passwd
USER 0 

# Mettre a jours de système et installer les packet
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        gcc \
        # Pour installer pyenv
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev wget ca-certificates \
        curl llvm libncurses5-dev xz-utils tk-dev \ 
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
        mecab-ipadic-utf8 libncursesw5-dev \
        git \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Installer de quoi créer une nevironnement virtuel
# Depuis la version 23.10 de Ubuntu python est concidérer comme "externally managed" apt concidère que python doit être installer
# via les packet Débian (python-foo). Par sécuriter pip refuser d'intaller dans les package python système
# Il faut créer une environnement virtuel dans le quel installer ces packets avec python3-venv

# Définition de la variable VENV_PATH
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH && \
    $VENV_PATH/bin/pip install --upgrade pip setuptools wheel

# récupération requirement
# Mettre les fichier dans répertoire temporaire pour éviter les problèmes d'accès
COPY ./requirements_main.txt /tmp/requirements_main.txt

# Instalation des package dans le venv
#   - `--no-cache-dir` évite de laisser le cache dans l’image
#   - `--disable-pip-version-check` accélère le build
RUN $VENV_PATH/bin/pip install --no-cache-dir \
        --disable-pip-version-check \
        -r /tmp/requirements_main.txt


# Installer pyenv pour mlflow
ENV PYTHON_VERSION=3.13.7

# Spécifier les chemine pour pyenv
RUN curl -fsSL https://pyenv.run | bash \
        ## Set up your shell environnement for Pyenv
    && echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    && echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    && echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash

# Récupération du modèle        
COPY ./mlflow_to_deploy /app/mlflow_to_deploy

# Instaler la version de python pour pyenv
#CMD ["pyenv", "install", "3.13.7"]

# Définir le repertoire de travail
WORKDIR /app

# Lancer l'API REST du model en utilisant l'environnement viruel
ENV PATH="$VENV_PATH/bin:$PATH"
CMD ["python", "-m", "mlflow", "models", "serve", "-m", "./mlflow_to_deploy"]
# Exposer le port par défaut de mlflow
EXPOSE 5000
# ADD path to file

# Lancer l'iinterface graphique pour intéroger l'API
# CMD ["streamlit", "run", "P7_dashboard.py"]
