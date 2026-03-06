# Projet Implémentation d’un modèle de scoring
date : 06/03/2026  
## Objectif du projet  
Ce projet vise à développer un modèle de scoring supervisé, suivi et déployé via une API. Il inclut la gestion du cycle de vie du modèle, ainsi que la mise en production et l’interface de test.  
Le modèle développer ce base sur des infomations bancaires et a pour but de permettre aux équipes de décider d'accorder un prêt à un client. Pour ce faire le modèle associer aux clients une probabiltée d'appartenir aux clients qui ont rembourser. En ce basant sur cette probabilté et un seuil qui est déterminer durant l'entrainement, le modèle calsse les clients en :
- 0 Le client est succéptible de rembourser
- 1 Le client est succéptible de ne pas remboursser

## Organisation du dossier API  
- `/app` : dossier principal de l’application de prévision de renboursement des clients.
  - `/mlflow_api` : dossier qui contient les fichiers pour le déploiment de l'api en docker.
  - `/Streamlit_ui` : dossier qui contient les fichiers pour le déploiment de l'interface utilisateur.
  - `/Dockerfile_mlflow_api` : dockerfile pour construire le docker qui permet de faire tourner l'api sur le cloud.
    - `/mlflow_to_deploy` : dossier qui contient le model a déployer pour docker.
    - `/requirements_api.txt` : requirement pour l'environnement dans le docker api.
  - `/Dockerfile_mlflow_api` : !non fonctionnel! dockerfile pour construire le docker pour stremalit en cas de déloiment sur le même service cloud que l'api mlflow.
    - `/P7_dashboard.py` : script python pour le dashboard de requête du modèle
    - `/requirement_ui.txt` : requirement pour l'environnement dans le docker streamlit_ui.
  - `/test_data.csv` : fichier qui permet contient une ligne du dataset d'entrainement et permet de tester l'api.  
  - `test_unitest.py` : fichier pour réaliser les unitests sur le cloud  
  - `test_requirements.txt` : liste des packages Python requis pour unitest
- `/data`
  - `/modeles` : dossier qui contient les models exporter volontairement de la base de données de mlflow et qui ont été déploiyer à un moment.
  - `/raw` : dossier pour les données brut du projte
  - `/transformed` : dossier pour les données issue du feature engineering et utiliser pour l'entrainement.
  - `/mlflow.db` : base de données de mlflow tracker, contient le résultats des différents entrainements.
- `/Suzon_Tugdual_2_notebook_modélisation_21112025` : notebook de feature engineering, d'entrainement de modèle tracker par mlflow, de sauvegarde du model en local or de la base de données de mlflow et étude du datadrift avec evidently.
- `/DataDrift_train_vs_test.html` : résultat de l'étude du datadrift avec evidently entre le jeux de données d'entrainement et de test.
- `/launch_local_model.py` : script pour lancer l'api du model stocker dans `/app` en local.
- `/launch_mlflow_ui.py` : script pour lancer l'ui de mlflow en local.
- `README.md` : ce fichier
- `test_unitest_locel.py` : script utiliser par unitest pour teste en local la présence du modèle et faire une prévision.
- `requirements.txt` : liste des package de l'environnement conda utilisé.  
  
Chaque dossier contient le code lié à sa fonctionnalité, facilitant la maintenance et l’évolution du projet.