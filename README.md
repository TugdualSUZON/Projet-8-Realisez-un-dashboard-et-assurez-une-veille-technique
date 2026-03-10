# Projet 7 Implémentation d’un modèle de scoring
date : 06/03/2026  
Organisme de formation : Openclassrooms  
Préparation au diplôme d'expert data scientist  
  
## Objectif du projet  
Ce projet vise à développer un modèle de scoring supervisé, suivi et déployé via une API. Il inclut la gestion du cycle de vie du modèle, ainsi que la mise en production et l’interface de test.  
  
Le code est versionner avec l'outil git et stocker sur github.  
Github se charge des tests unitaires, de la construction du docker, stockage de l'image docker et de sont déploiement sur le service cloud Render.  
  
Le modèle développé ce base sur des informations bancaires et a pour but de permettre aux équipes de décider d'accorder un prêt à un client. Pour ce faire le modèle associé aux clients une probabilité d'appartenir aux clients qui ont remboursé. En se basant sur cette probabililté et un seuil qui est déterminé durant l'entraînement, le modèle classe les clients en :
- 0 Le client est susceptible de rembourser.
- 1 Le client est susceptible de ne pas rembourser.
  
## Prérequis d'installations  

Environnement conda avec :
- python 3.11.9
- pyarrow : 12.0.0
- pip : mlflow 3.7.0, suivie de l'entraînement des modèles et déploiement du modèle sous la forme d'API.
- pip : virtualenv 20.35.4, environnement virtuel utiliser pour le déploiement du modèle.
- scikit-learn 1.8.0, fournis la structure et les algorithmes pour entraîner les modèles de machine learning.
- imbalanced-learn, permet de traiter le problème de déséquilibre des classes dans le dataset.
- streamlit 1.55.0, pour l'ui de requête de l'api
- evidently, etude du datadrift
- pyarrow==12.0.0, pour sklearn et evidently
  
En local dans le cas d'une machine windows a installer sur le système :
- pyenv installation : [gihub](https://github.com/pyenv-win/pyenv-win/blob/master/docs/installation.md#add-system-settings).
- ajouter la version 3.11.9 de python au registre de pyenv.
  
## Utilisation  
  
1) Copier le git en local.
2) Installer l'environnement conda avec le requirements.txt, si test en local de l'api model installer pyenv sur la machine.
3) Lancer le sevice de suivie de mlflow et ce connecter au server local avec `launch_mlflow_ui.py`.
4) Utiliser le notebook `partie I Suzon_Tugdual_2_notebook_modélisation_112025.ipynb` pour appliquer le feature engineering.
5) Utiliser la notebook `partie II`, en `modifiant` le point `3.1 définition du pipeline` et le point `3.3 définition` du grid-search au besoin.
6) `Tester sur un échantillon` du dataset le grid-search+pipline avec le point `3.4 test du pipline`
7) `Entraîner` le modèle avec un `suivi de mlflow` avec le point `4`, modifier le nom d'expérience dans le `point 4.2`, le nom de la run dans le point `4.3` et `4.5`.
8) `Utiliser l'ui de mlflow` pour déterminer le meilleur modèle, l'ajouter au registre et le tag `best`.
9) Utiliser le `point 5.` pour sauver en local ce modèle dans le dossier `./data/modeles`.
10) Copier le dossier de ce modèle dans le dossier `app/mlflow_api/` avec le nom `mlflow_to_deploy`
11) Déployer l'api en local avec le script `launch_local_model_api.py`.
12) Lancer le script de l'ui de requête `/app/Streamlit_ui/P7_dashboard.py`.

## Organisation du dossier  
  
- `/app` : dossier principal de l’application de prévision de remboursement des clients.  
  
  - `/mlflow_api` : dossier qui contient les fichiers pour le déploiement de l'api en docker.  
  
    - `/mlflow_to_deploy` : dossier qui contient le modèle a déployé pour docker.
    - `/requirements_api.txt` : requirements pour l'environnement dans le docker api.  

  - `/Streamlit_ui` : dossier qui contient les fichiers pour le déploiement de l'interface utilisateur.  
 
    - `/P7_dashboard.py` : script python pour le dashboard de requête du modèle
    - `/requirement_ui.txt` : requirement pour l'environnement dans le docker streamlit_ui.  
  
  - `/Dockerfile_mlflow_api` : !non fonctionnel! dockerfile pour construire le docker pour Streamlit en cas de déploiement sur le même service cloud que l'api mlflow.
  - `/test_data.csv` : fichier qui permet contient une ligne du dataset d'entraînement et permet de tester l'api.  
  - `test_unitest.py` : fichier pour réaliser les unitests sur le cloud.
  - `test_requirements.txt` : liste des packages Python requis pour unitest.  
  
- `/data` : centraliser tous les fichiers de données à un seul endroit.  
  
  - `/modeles` : dossier qui contient les modèles exportés volontairement de la base de données de mlflow et qui ont été déployer à un moment.
  - `/raw` : dossier pour les données brut du projet.
  - `/transformed` : dossier pour les données issu du feature engineering et utiliser pour l'entraînement.
  - `/mlflow.db` : base de données de mlflow tracker, contient le résultat des différents entraînements.  
  
- `/Suzon_Tugdual_2_notebook_modélisation_21112025` : notebook de feature engineering, d'entrainement de modèle tracker par mlflow, de sauvegarde du model en local or de la base de données de mlflow et étude du datadrift avec evidently.
- `/DataDrift_train_vs_test.html` : résultat de l'étude du datadrift avec evidently entre le jeu de données d'entraînement et de test.
- `/launch_local_model.py` : script pour lancer l'api du modèle stocker dans `/app` en local.
- `/launch_mlflow_ui.py` : script pour lancer l'ui de mlflow en local.
- `/README.md` : ce fichier
- `/test_unitest_locel.py` : script utiliser par unitest pour test en local la présence du modèle et faire une prévision.
- `/requirements.txt` : liste des packages de l'environnement conda utilisé.  
  
Chaque dossier contient le code lié à sa fonctionnalité, facilitant la maintenance et l’évolution du projet.  

