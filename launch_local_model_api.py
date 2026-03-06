# File_name set_launch_mlflow_server.py
# Author Tugdual SUZON 27/11/2025
# Created 06/03/2026
# Rendre disponible en local le modèle de "predict_client_payment" avec le tag "best"

if __name__ == '__main__':

    # Packages
    import os
    import subprocess
    
    # Cd ce placer dans le même répertoire que la base de données de mlflow
    os.chdir(./data)
    
    # Nom du model à charger depuis la base de donnée de mlflow
    model_name = "predict_client_payment" # Non du modèle dans le registre
    model_version_alias = "best" # tag du model a exporté
    
    # Lancer le processus
    process = subprocess.Popen([
        "mlflow", "models, ""server",
        "-m", f"models:/{model_name}@{model_version_alias}",
        "--host", "127.0.0.1",
        "--port", "10000"
    ])
    
    print("Model Mlflow disponible pour inférence à l'adresse http://127.0.0.1:10000")