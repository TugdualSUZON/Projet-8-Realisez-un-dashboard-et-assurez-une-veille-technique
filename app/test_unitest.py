# But, réaliser des test unitaire
# Charger le model : est-ce que le model est disponible
# Faire une prévision en donnant au model un demmy pour vérifier sont fonctionnement

import os
import pandas as pd
import mlflow.sklearn

from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.model_selection import GridSearchCV

from dotenv import load_dotenv

# charger les variables d'envrionnement
load_dotenv(".env")
# Chemin vers le model a déployer
# Chemin vers les données test
    
def test_load_model():
    
    #MODEL_PATH = r"./app/mlflow_api/mlflow_to_deploy"

    # Arrange , recuperer le chemin du model
    MODEL_PATH = os.getenv("MODEL_PATH")
    
    # Act , charge le model
    model = mlflow.sklearn.load_model(MODEL_PATH)
    
    # Assert ,  test le type de model
    assert type(model) == TunedThresholdClassifierCV or type(model) == GridSearchCV #Le model doit ce charger et être un optimiser de seuil
    
def test_model_predict():

    #MODEL_PATH = r"./app/mlflow_api/mlflow_to_deploy"
    #TEST_DATA_PATH = r"./app/test_data.csv"
        
    # Arrange , recuperer les données du test
    TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")
    MODEL_PATH = os.getenv("MODEL_PATH")
    
    # Charger et formater les données
    data = pd.read_csv(TEST_DATA_PATH)
    # retirer les features qui ne sont pas nécéssaire a la prévision
    X = data.drop(["SK_ID_CURR", "TARGET"], axis=1)
    
    # Charger le model
    model = mlflow.sklearn.load_model(MODEL_PATH)
    
    # Act , faire la prévision
    pred = model.predict(X)
    
    # Assert , Le model renvio une prévision
    assert type(pred) != type(None)
