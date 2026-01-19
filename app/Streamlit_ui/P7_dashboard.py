import copy

import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO

# Fonctions
def request_prediction(df):
    # Charger la variables d'environnement pour render
    MLFLOW_URL = os.getenv("MLFLOW_URL")

    # Formater le csv pour l'envoi au modèle
    df_line = df.iloc[[0], :]  # votre ligne
    csv_buffer = StringIO()
    df_line.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0) # Remettre le pointer à 0 suite à l'écriture
    #csv_data = csv_buffer.getvalue() # Pour voir ce qu'il y a dans le fichier csv produit, peux aussi être passer à request
    
    response = requests.post(
        # url="http://127.0.0.1:5000/invocations", # Pour l'utilisation en local
        url=f"{MLFLOW_URL}/invocations",
        data=csv_buffer,
        headers={"Content-Type": "text/csv"}
    )
    
    return response.status_code, response.json()

def main():
    uploaded_file = st.file_uploader("Déposé ici le fichier d'information clients en format csv", type="csv")
    if uploaded_file is not None:
        # Charger le fichier en mémoire
        raw = pd.read_csv(uploaded_file, sep=",")
        # Vérifier l'upload
        st.write(raw)
    
        # Préprocess
        ## Gestion de l'annomalie dans "DAYS_EMPLOYED_ANOM"
        df = copy.copy(raw)
        df['DAYS_EMPLOYED_ANOM'] = 0
        mask = df["DAYS_EMPLOYED"] == 365243
        df.loc[mask, "DAYS_EMPLOYED_ANOM"] = 1
        df.loc[mask, "DAYS_EMPLOYED"] = np.nan
        
    predict_btn = st.button('Prédire')
    
    if predict_btn:
        pred = None
        response = None
        response, pred_dict = request_prediction(df)
        pred = pred_dict["predictions"][0][1]
        st.write("Rique que le client ne rembouse par {:.2f}".format(pred)
                )
        st.write(f"Réponse de l'API {response}")
        
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