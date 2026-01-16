FROM ubuntu/python:3.13-22.04_stable
WORKDIR ./app
COPY ./app # Model, requirement and main.py et le dockerfile

RUN pip install -r requirements_main.txt
 
#pyenv for mlflow API REST https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix

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