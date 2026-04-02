# File_name set_launch_mlflow_server.py
# Author Tugdual SUZON 27/11/2025
# Created 06/03/2026
# Update 02/04/2026
# Rendre disponible en local le modèle de "predict_client_payment" avec le tag "best" ou charge le model depuis le repertoire de l'app

if __name__ == '__main__':

    # Packages
    import os
    import subprocess

    import webbrowser

    def select_launch_type():
        
        while True:
            launch_type = input("Lancer le model depuis la {db} ou le {repertoire} app ?")
            
            if launch_type == "db":
                
                print("Chargement du model depuis la base de données")
                os.chdir(r"./data")
                # Nom du model à charger depuis la base de donnée de mlflow
                model_name = "predict_client_payment" # Non du modèle dans le registre
                model_version_alias = "best" # tag du model a exporté
                launch_path = f"models:/{model_name}@{model_version_alias}"
                
                break
                
            elif launch_type == "repertoire":
                print("Charmenet du modèle depuis le dossier app mlflow_ui mlflow_to_deploy")
                launch_path = r"./app/mlflow_api/mlflow_to_deploy"
                break
                
            else :
                print("Le type de lancement demander est invalide, valeurs possible {db} ou {repertoire}")
                
        
        return launch_path
        
    # Demander à l'utilisateur
    launch_type = select_launch_type()
    
    # Lancer le processus
    process = subprocess.Popen([
        "mlflow", "models", "serve",
        "-m", launch_type,
        "--host", "127.0.0.1",
        "--port", "10000"
    ])
    
    print("Model Mlflow disponible pour inférence à l'adresse http://127.0.0.1:10000")

    input("Appuyez sur Entrée pour arrêter le serveur...\n")
    process.terminate()
    process.wait()
    print("Serveur arrêté.")