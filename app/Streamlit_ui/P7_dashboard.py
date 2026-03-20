import copy
import os 

import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO

import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
from streamlit_shap import st_shap # https://github.com/snehankekre/streamlit-shap/tree/main

# Fonctions

@st.cache_data
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

@st.cache_data    
def load_shap_values(shap_pkl_path=r"./shap_values_unscaled.pkl"):
    
    # Charger le fichier des valeurs de shap
    with open(shap_pkl_path, "rb") as f :
        unpickeled_object = pickle.load(f)

    # Verifier que l'objet est bien un dictionnaire.
    if type(unpickeled_object) == dict:
        
        # Extraire les deux fichier du dictionnaire
        key_tab = unpickeled_object["key_tab"]
        shap_values_unscaled = unpickeled_object["shap_values_unscaled"]

        return key_tab, shap_values_unscaled

    else :
        print("erreur le fichier charger n'est pas dans le format attendu")
        return

@st.cache_data
def load_data():

    # Charger les données générale
    ## Dataset d'entrainement complet ?
    all_data = pd.read_csv(r"C:\Users\SUZON\OneDrive - CNR\Documents\Jupyter\Openclassrooms\Projets Openclassrooms\Projet-8-Realisez-un-dashboard-et-assurez-une-veille-technique\data\transformed\train_data_V1.csv")
    
    ## Clé primaire et Valeurs de shap
    key_tab, shap_values_unscaled = load_shap_values()
   
    return all_data, key_tab, shap_values_unscaled
    
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
    
def render_shap_plot(SK_ID_CURR, key_tab, shap_values_unscaled):

    # Tracer le Feature importance global
    st.title("Feature importance global du model")
    st_shap(shap.plots.bar(shap_values_unscaled),
            width=1400,
            height=500
           )

    # Relier l'id client avec l'index du fichier de shap_values_unscaled
    mask = key_tab["SK_ID_CURR"] == SK_ID_CURR
    index = key_tab.loc[mask, :].index[0]

    # Feature importance local pour un client
    st.title(f"Feature importance local du client {SK_ID_CURR}")
    st_shap(shap.waterfall_plot(shap_values_unscaled[index]),
            width=1400,
            height=500
           )


    
def main():

    all_data, key_tab, shap_values_unscaled = load_data()
    
    # Haut de page, deux colonnes, une pour séléctionner un client ou déposer un fichier, l'autre pour afficher les valeurs principale de ce client
    ## Définir deux colonnes
    left_column, right_column = st.columns(2)

    with left_column:
        
        # Charger les données
        ## Depuis un fichier
        uploaded_file = st.file_uploader("- Charger les données client au format .csv ici :",
                                         type="csv"
                                        )
        
        if uploaded_file is not None:
            # Charger le fichier en mémoire
            raw = pd.read_csv(uploaded_file)
            # Vérifier l'upload
            st.write(raw)

            # Récupérer du dataframe ajouter l'ID du client
            df = copy.copy(raw)
            SK_ID_CURR = df["SK_ID_CURR"]
            SK_ID_CURR = SK_ID_CURR.values[0]
            del df["SK_ID_CURR"]

        ## Depuis la base de données client
        if uploaded_file == None :
            SK_ID_CURR = st.number_input("- Définiser un identifiant client ici :", 
                                         value=None,
                                         format="%0f",
                                         placeholder="Identifiant client : SK_ID_CURR"
                                        )
            
            # Transformer le float produit par input_number en entier
            SK_ID_CURR = int(SK_ID_CURR)

            # Vérifier que l'ID demander est dans le dataframe des données d'entrainement
            if all_data.loc[all_data["SK_ID_CURR"] == SK_ID_CURR, :].empty:
                SK_ID_CURR = None
                st.write("ERREUR : Identifiant inconnu entrée un identifiant valide")

    with right_column:
        if SK_ID_CURR is not None :
            '''
            - Identifiant client actuel
            '''
            st.write(f"Identifiant client : {SK_ID_CURR}")
            '''
            - Valeurs des features les plus importante
            '''
            raw_main_features = copy.copy(all_data)
            
            # Liste des colonnes à afficher
            main_features = ["TARGET", "CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            raw_main_features = raw_main_features.loc[raw_main_features["SK_ID_CURR"] == SK_ID_CURR, main_features]
            # Filtrer le dataframe selon la liste des colonnes
            raw_main_features = raw_main_features.loc[:, main_features]
            # Pivoter le dataframe
            raw_main_features = pd.melt(raw_main_features,
                                        #id_vars="SK_ID_CURR",
                                        value_vars=raw_main_features)
            # Affcher le dataframe
            st.dataframe(raw_main_features)
            
        else :
            '''
            - Valeurs des features les plus importante  
              
            Aucune données client renseigner.
            '''
        
        
        
    predict_btn = st.button('Prédire')
    feature_importance_btn = st.button('Afficher les graph de shap')
    
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

    if SK_ID_CURR is not None :
        render_shap_plot(SK_ID_CURR, key_tab, shap_values_unscaled)

"""
# Affectation des crédit
Déterminer la probabilté de remboursement du client :  
  
1) **Ajouter des information client ou saisisez un identifiant client**
"""

if __name__ == '__main__':
    
    st.set_page_config(layout="wide")
    
    # Définir le repertoire actif
    os.chdir(r"C:\Users\SUZON\OneDrive - CNR\Documents\Jupyter\Openclassrooms\Projets Openclassrooms\Projet-8-Realisez-un-dashboard-et-assurez-une-veille-technique\app\Streamlit_ui")
    
    main()