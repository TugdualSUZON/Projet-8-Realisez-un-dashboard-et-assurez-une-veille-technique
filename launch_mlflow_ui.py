# File_name set_launch_mlflow_server.py
# Author Tugdual SUZON 27/11/2025
# Created 27/11/2025

if __name__ == '__main__':

    # Packages
    import mlflow
    import subprocess
    import time

    import webbrowser
    
    # Lancer le processus
    process = subprocess.Popen([
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///data/mlflow.db",
        #"--default-artifact-root", "file:///chemin/absolu/vers/data/mlruns",
        "--host", "127.0.0.1",
        "--port", "5000"
    ])
    
    print("Serveur UI MLflow démarré.")
    
    #Bash
    # Launch : mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
    # Stop : mlflow server stop
    
    # Set :
    # Python
    #  mlflow.set_tracking_uri("http://localhost:5000")

    # Faire ouvrir l'interface web
    webbrowser.open("http://127.0.0.1:5000")
    
    input("Appuyez sur Entrée pour arrêter le serveur...\n")
    process.terminate()
    process.wait()
    print("Serveur arrêté.")

    

