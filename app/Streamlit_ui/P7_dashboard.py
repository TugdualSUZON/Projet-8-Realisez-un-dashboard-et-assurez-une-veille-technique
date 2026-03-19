import copy

import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Fonctions
def request_prediction(df):
    # donner l'URL
    ## Fonctionnement tout en local
    # MLFLOW_URL = "http://127.0.0.1:10000" # Pour l'utilisation en local

    ## Fonctionnement en loca et serveur cloud
    MLFLOW_URL = "https://predict-client-payment-main.onrender.com"
    
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

def render_threshold_value(value) :
    
    # Style seaborn pour de belles couleurs par défaut
    sns.set(style="whitegrid")
    
    st.title("Valeur du seuil de décison du modèle et positionnement du client")
    
    # Définir les variable de la fonction
    threshold = 0.5
    
    # Options de couleur pour les deux zones (avant/après seuil)
    cmap_left = plt.get_cmap("YlOrRd_r")   # couleur pour [0, t] _r pour inverse la gamme
    cmap_right = plt.get_cmap("YlGn")  # couleur pour [t, 1]
    
    # Résolution horizontale (nombre de colonnes dans la barre)
    N = 512
    
    # Indices de position uniformes entre 0 et 1
    x = np.linspace(0.0, 1.0, N)
    
    # Construire un tableau couleurs de taille (hauteur, N, 3)
    height = 40  # épaisseur visible de la barre en pixels (dans l'image)
    bar_img = np.zeros((height, N, 3))
    
    # Nombre de colonnes à colorer selon le seuil
    n_left = int(np.round(threshold * (N - 1)))  # colonnes pour la partie gauche
    n_right = N - n_left
    
    # Gérer cas limites (threshold == 0 ou 1)
    if n_left <= 0:
        # Tout en droite (aucune section gauche)
        colors_right = cmap_right(np.linspace(0.0, 1.0, N))
        for i in range(height):
            bar_img[i, :, :] = colors_right[:, :3]
            
    elif n_right <= 0:
        # Tout en gauche (aucune section droite)
        colors_left = cmap_left(np.linspace(0.0, 1.0, N))
        for i in range(height):
            bar_img[i, :, :] = colors_left[:, :3]
            
    else:
        # Échantillonner les deux colormaps proportionnellement
        # La partie gauche parcourt cmap_left de 0 → 1
        colors_left = cmap_left(np.linspace(0.0, 1.0, n_left))
        # La partie droite parcourt cmap_right de 0 → 1
        colors_right = cmap_right(np.linspace(0.0, 1.0, n_right))
    
        # Concaténer et remplir l'image
        colors_combined = np.vstack([colors_left, colors_right])
        for i in range(height):
            bar_img[i, :, :] = colors_combined[:, :3]
    
    # Tracer la barre avec Matplotlib
    fig, ax = plt.subplots(figsize=(8, 1.2))  # taille adaptée pour une barre horizontale
    ax.imshow(bar_img, extent=[0, 1, 0, 1], aspect="auto")
    
    # Supprimer axes inutiles
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité d'acceptation du dossier")
    
    # Tracer la ligne verticale du seuil (trait noir pointillé)
    ax.axvline(x=threshold, color="k", linestyle="-", linewidth=2, label=f"Seuil (t) = {threshold:.2f}")
    
    # Tracer la ligne verticale de la valeur (trait bleu plein)
    ax.axvline(x=value, color="k", linestyle="-", linewidth=3, label=f"Score client (v) = {value:.2f}")
    
    # Ajouter annotations textuelles au-dessus des lignes
    # Positionner un peu au-dessus de la barre (y ~ 1.05)
    ax.text(threshold, 1.05, f"seuil = {threshold:.2f}", ha="center", va="bottom", fontsize=10, color="k")
    ax.text(value, 1.05, f"score client = {value:.2f}", ha="center", va="bottom", fontsize=10, color="k")
    
    # Ajuster l'affichage
    plt.tight_layout()
    
    # Afficher la figure dans Streamlit
    st.pyplot(fig)
    
    # Indication texte expliquant les contrôles (avec variables mathématiques en LaTeX)
    st.markdown(
        "Aide sous le graphique "
    )

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
        
        #pred = None
        #response = None
        #response, pred_dict = request_prediction(df)
    
        #pred = pred_dict["predictions"][0]
        #if pred == 0:
            #st.write(f"Le client {SK_ID_CURR} appartient a la catégorie 0, le modèle prévoit qu'il remboursera son crédit"
                #)
        #elif pred == 1:
            #st.write(f"Le client {SK_ID_CURR} appartient a la catégorie 1, le modèle prévoit qu'il ne remboursera pas son crédit"
                #)
        #st.write(f"Réponse de l'API {response}")
        
        render_threshold_value(0.20)

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