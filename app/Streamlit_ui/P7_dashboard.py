import copy

import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO

# Fonctions
def request_prediction(df):
    # donner l'URL
    ## Fonctionnement tout en local
    MLFLOW_URL = "http://127.0.0.1:5000" # Pour l'utilisation en local
    
    ## Fonctionnement en loca et serveur cloud
    #MLFLOW_URL = "http://projet-7-implementez-un-modele-de-scoring.onrender.com"
    
    ## fonctionnement tous cloud
    #MLFLOW_URL = os.getenv("MLFLOW_URL")

    # Formater le csv pour l'envoi au modèle
    df_line = df.iloc[[0], :]  # votre ligne
    csv_buffer = StringIO()
    df_line.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0) # Remettre le pointer à 0 suite à l'écriture
    #csv_data = csv_buffer.getvalue() # Pour voir ce qu'il y a dans le fichier csv produit, peux aussi être passer à request
    
    response = requests.post(
        url=f"{MLFLOW_URL}/invocations", 
        data=csv_buffer,
        headers={"Content-Type": "text/csv"},
        verify=False
    )
    
    return response.status_code, response.json()

def main():
    uploaded_file = st.file_uploader("Déposé ici le fichier d'information clients en format csv", type="csv")
    if uploaded_file is not None:
        # Charger le fichier en mémoire
        raw = pd.read_csv(uploaded_file)
        # Vérifier l'upload
        st.write(raw)

        df = copy.copy(raw)
        SK_ID_CURR = df["SK_ID_CURR"]
        SK_ID_CURR = SK_ID_CURR.values[0]
        del df["SK_ID_CURR"]
        st.write(f"Identifiant client : {SK_ID_CURR}")

        
        # Préprocess
        ## Gestion de l'annomalie dans "DAYS_EMPLOYED_ANOM"
        #df = copy.copy(raw)
        #df['DAYS_EMPLOYED_ANOM'] = 0
        #mask = df["DAYS_EMPLOYED"] == 365243
        #df.loc[mask, "DAYS_EMPLOYED_ANOM"] = 1
        #df.loc[mask, "DAYS_EMPLOYED"] = np.nan
        
        
    predict_btn = st.button('Prédire')
    
    if predict_btn:
        pred = None
        response = None
        response, pred_dict = request_prediction(df)
    
        pred = pred_dict["predictions"][0]
        if pred == 0:
            st.write(f"Le client {SK_ID_CURR} appartient a la catégorie 0, le modèle prévoit qu'il remboursera son crédit"
                )
        elif pred == 1:
            st.write(f"Le client {SK_ID_CURR} appartient a la catégorie 1, le modèle prévoit qu'il ne remboursera pas son crédit"
                )
        #st.write(f"Réponse de l'API {response}")
        
"""
# Affectation des crédit
Déterminer la probabilté de remboursement du client :
"""

if __name__ == '__main__':
    main()

"""
INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
INFO:     127.0.0.1:47046 - "HEAD / HTTP/1.1" 404 Not Found
"""