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

#------------------------Charger des données--------------------------#

@st.cache_data    
def load_shap_values():
    
    # Dans le cas du déploiment sur streamlit cloud
    if st.secrets["SHAP_PKL_PATH"]:
        shap_pkl_path = st.secrets["SHAP_PKL_PATH"]
      
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

    ## Clé primaire et Valeurs de shap
    key_tab, shap_values_unscaled = load_shap_values()
    
    ## Consrtuire all_data à partir de l'objet shap
    all_data = shap_values_unscaled[:, :].data
    all_data = pd.DataFrame(all_data, columns=shap_values_unscaled.feature_names)
   
    return all_data, key_tab, shap_values_unscaled
def click_button(button_variable):
        st.session_state[button_variable] = True
    
def button_set_value(button_variable, new_value):
        st.session_state[button_variable] = new_value
    
#------------------------Requet_au_sevice_modele------------------------

def request_ping():
    # donner l'URL
    ## Fonctionnement tout en local
    #MLFLOW_URL = "http://127.0.0.1:10000" # Pour l'utilisation en local

    ## Fonctionnement en loca et serveur cloud
    MLFLOW_URL = "https://predict-client-payment-sha256.onrender.com"
    
    ## fonctionnement tous cloud
    #MLFLOW_URL = os.getenv("MLFLOW_URL")

    try:
        response = requests.get(
                                url=f"{MLFLOW_URL}/ping",
                                verify=False,
                                timeout=5
                                )
            
        return response.status_code, response.text
        
    except Exception as e:
        return None, str(e)

@st.cache_data
def request_prediction(df):
    # donner l'URL
    ## Fonctionnement tout en local
    #MLFLOW_URL = "http://127.0.0.1:10000" # Pour l'utilisation en local

    ## Fonctionnement en loca et serveur cloud
    MLFLOW_URL = "https://predict-client-payment-sha256.onrender.com"
    
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
    
#------------------------Produire les graphiques--------------------------#
def render_threshold_value(value) :
    
    # Style seaborn pour de belles couleurs par défaut
    sns.set(style="whitegrid")
    
    st.title("Valeur du seuil de décison du modèle et positionnement du client")
    
    # Définir les variable de la fonction
    threshold = 0.5
    
    # Options de couleur pour les deux zones (avant/après seuil)
    cmap_left = plt.get_cmap("YlGn_r")   # couleur pour [0, t] _r pour inverse la gamme
    cmap_right = plt.get_cmap("YlOrRd")  # couleur pour [t, 1]
    
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
    ax.set_xlabel("Probabilité de refuser le prêt")
    
    # Tracer la ligne verticale du seuil (trait noir pointillé)
    ax.axvline(x=threshold, color="k", linestyle="-", linewidth=2, label=f"Seuil (t) = {threshold:.2f}")
    
    # Tracer la ligne verticale de la valeur (trait bleu plein)
    ax.axvline(x=value, color="b", linestyle="-", linewidth=2, label=f"Score client (v) = {value:.2f}")
    
    # Ajouter annotations textuelles au-dessus des lignes
    # Positionner un peu au-dessus de la barre (y ~ 1.05)
    ax.text(threshold, 1.3, f"seuil = {threshold:.2f}", ha="center", va="bottom", fontsize=10, color="k")
    ax.text(value, 1.03, f"score client = {value:.2f}", ha="center", va="bottom", fontsize=10, color="b")
    
    # Ajuster l'affichage
    plt.tight_layout()
    
    # Afficher la figure dans Streamlit
    st.pyplot(fig)

    plt.close(fig)
    
    st.markdown('''Le score client représente la probabilité que le client ne rembourse pas le prêt qui lui sera accordé.  
    - Proche de 0 le client a de grande chance de rembourser sont prêt.  
    - Proche de 1 le client à de grande chance de ne pas rembourser sont prêt.''')
    
@st.cache_data
def render_feature_importance(_shap_values_unscaled) :
        st.write(f"-Feature importance global du modèle")
            
        fig, ax = plt.subplots(
                              figsize=(10, 5),
                            )
            
        ax = shap.plots.bar(_shap_values_unscaled, ax=ax, show=True)
    
        st.pyplot(fig, width="content")

def get_client_index(SK_ID_CURR, key_tab):
    
    # Relier l'id client avec l'index du fichier de shap_values_unscaled
    try:
        # Relier l'id client avec l'index du fichier de shap_values_unscaled
        mask = key_tab["SK_ID_CURR"] == SK_ID_CURR
        index = key_tab.loc[mask, :].index[0]

    except:
        index = "error"

    return index

@st.cache_data
def render_shap_waterfall(_shap_values_unscaled, SK_ID_CURR, key_tab):

    index = get_client_index(SK_ID_CURR, key_tab)
    
    # Feature importance local pour un client
    st.write(f"-Feature importance local du client {SK_ID_CURR}")
    st_shap(shap.waterfall_plot(_shap_values_unscaled[index]),
            width=1400,
            height=500
           )

    st.markdown('''La valeurs f est un logarithme de la probabilité.  
    - Une valeur négative indique une probabilité de non remboursement inférieur à 50%.  
    - A l'inverse une valeurs positive indique une propbabilté de non remboursement supérieur à 50%.''')
        
@st.cache_data
def render_violineplot(_all_data, key_tab, feature, SK_ID_CURR) :
    
    # Index du client
    index = get_client_index(SK_ID_CURR, key_tab)
    
    # Client value
    client_value = _all_data.loc[index, feature]

    # Description du graphique
    st.write("Position du client au sein de l'ensemble des clients, la valeur de la feature du client est donné par le point rouge")
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    sns.violinplot(_all_data[feature], ax=ax)
    sns.stripplot({0: client_value}, edgecolor='black', linewidth=1, palette=['red'], ax=ax)

    ax.set_title(f"Graphique de densité de la feature {feature}")
    
    st.pyplot(fig, width="content")

@st.cache_data
def render_shap_scatter_plot(_shap_values_unscaled, SK_ID_CURR, key_tab, feature):

    # Client value
    index = get_client_index(SK_ID_CURR, key_tab)
    
    # Description du graphique
    st.text(f"Lien entre {feature} et la valeurs de shap la valeurs du client est représenté par un point rouge")
    
    fig, ax = plt.subplots(figsize=(4, 3))
            
    ax.set_title(f"Valeur de shap VS feature {feature}")
                
    shap.plots.scatter(_shap_values_unscaled[:, feature],
                       dot_size=3,
                       ax=ax)
    
    ax.scatter(_shap_values_unscaled[index, feature].data,
               _shap_values_unscaled[index, feature].values,
               s = 3,
               c = "red",
                )
                
    st.pyplot(fig, width="content")

@st.cache_data
def render_bivariate_scatterplot(_shap_values_unscaled, SK_ID_CURR, key_tab, feature_1, feature_2):
    
    # Client value
    index = get_client_index(SK_ID_CURR, key_tab)
            
    # Description du graphiques
    st.write(f"Nuage de point montrant le lien entre deux features {feature_1} vs {feature_2}")

    # Produire le graphique
    fig, ax = plt.subplots(figsize=(6,6))
            
    ax.set_title(f"Variation de {feature_2} en fonction de {feature_1}")
            
    ax.scatter( 
                x = _shap_values_unscaled[:, feature_1].data,
                y = _shap_values_unscaled[:, feature_2].data,
                s = 3
               )
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)

    ax.scatter(x = _shap_values_unscaled[index, feature_1].data,
               y = _shap_values_unscaled[index, feature_2].data,
               s = 3,
               c = "red",
                )  
          
    st.pyplot(fig, 
              width="content"
             )


#------------------------Main fonction--------------------------#
    
def main():
    
    '''
    # Affectation des crédit
    Déterminer la probabilté de remboursement du client :  
    '''
    
#------------------------Charger des données--------------------------#
    all_data, key_tab, shap_values_unscaled = load_data()

#------------------------Haut de page choix du client--------------------------#
    # Haut de page, deux colonnes, une pour sélectionner un client ou déposer un fichier, l'autre pour afficher les valeurs principale de ce client
    '''
    ## 1) **Ajouter des information client ou saisisez un identifiant client**
    '''
    ## Définir deux colonnes
    left_column, right_column = st.columns(2)

##-------------------- Choix du client
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
            if SK_ID_CURR is not None :
                SK_ID_CURR = int(SK_ID_CURR)
                index = get_client_index(SK_ID_CURR, key_tab)

                # Vérifier que l'ID demander est dans le dataframe des données d'entraînement
                if index == "error" :
                    st.write("ERREUR : Identifiant inconnu entrée un identifiant valide")
                    st.stop()
                
##-------------------- Détaille des valeurs des features les plus importante pour le client sélectionné
    with right_column:
        if SK_ID_CURR is not None :
            '''
            - Identifiant client actuel
            '''
            st.write(f"Identifiant client : {SK_ID_CURR}")
            '''
            - Valeurs des features les plus importante
            '''

            # Liste des colonnes à afficher
            main_features = ["CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            
            # Estraire les données pour les feature les plus importantes et calculer des stats descriptive
            extract = all_data.loc[:, main_features]
            extract = pd.melt(extract, value_vars=extract)
            extract = extract.groupby("variable").agg({'value' : ["mean", "median"]})                  

            # Index du client
            index = get_client_index(SK_ID_CURR, key_tab)
            
            # Extraire les valeurs des features pour le client
            all_client_values = shap_values_unscaled[index, :].data
            all_client_values = pd.DataFrame([all_client_values], columns = shap_values_unscaled.feature_names)
            client_values = all_client_values[main_features]
            client_values = shap_values_unscaled[index, main_features].data
            client_values = pd.DataFrame([client_values], columns = main_features)
            client_values = pd.melt(client_values, value_vars=client_values)
            client_values = client_values.set_index("variable")

            # Coller les morceau
            features_values_sumup = pd.concat([client_values, extract], axis=1)
            
            # Afficher le dataframe
            st.dataframe(features_values_sumup)
            
        else :
            '''
            - Valeurs des features les plus importante  
              
            Aucune données client renseigner.
           '''

    del left_column, right_column

    
    ##-------------------- Interrogation de l'API, Prédire la probabilité du client en fonction des données d'entré, 
    
    # point 2, une colonne, tester la disponibilité du serveur, lancer une pédiction
    # Uniquement si un client a été séléctionné
    
    if SK_ID_CURR is not None :
        
        '''
        ## 2) **Demander au model la probabilité de remboursement du client**
        '''
        # Tester la disponibilté de l'API du model
        if st.button('Vérifier la disponibilté du serveur'):
            
            ping_response, ping_text = request_ping()
    
            if ping_response == 200 :
                st.write(f"Le serveur est actif")
            else :
                st.write(f"Le serveur n'est pas actif")
                st.write(f"reponse : {ping_response}, text : {ping_text}")
            
        # Utiliser la persistance des valeurs de variable entre les runs
        if "predict_btn" not in st.session_state:
            st.session_state["predict_btn"] = False
    
        # Lancer la requet à l'API
        st.button('Prédire', on_click=click_button("predict_btn"))
        
        if st.session_state.predict_btn :
            
            pred = None
            response = None
            response, pred_dict = request_prediction(all_client_values)
            
            # Gerer la réponse du serveur
            if response == 200 :
                
                # récupèrer la prédiction du modèle, propabilité d'appartenance a la class 0 vas rembourser sont crédit
                pred = pred_dict["predictions"][0][0]

                # Produire le graphique
                render_threshold_value(pred)
                
            elif response == 400:
                st.write("La requête au modèle n'a pas marché : Bad Request")
            
            else:
                st.write(f"!! Code résponse inconnue: {response}")

      ##-------------------- Modifier les données d'entrée et demander une nouvelle prédiction

      # point 3, une colonne, proposer de modifier des valeurs avec un bouton, afficher l'inteface de modification, faire une prédiction et affichier le graph
        '''
        ## 3) **Modifier les informations client**
        '''
        
        if "modification_button" not in st.session_state:
            st.session_state["modification_button"] = 0
            
        # Ouvrire l'interface de modification
        st.button('Modifier des variables', on_click = button_set_value, args=["modification_button",1])

        # Effacer l'interface de modification, annuler les modfication
        if st.button("Effacer les modifications"):
            button_set_value("modification_button", 0)
            
        # Afficher le l'interface de modification du dataframe
        if st.session_state.modification_button >= 1 :
            
            # Copier le dataset d'origine
            all_client_values_modified = copy.copy(all_client_values)
    
            # Modifier le nouveau dataset, juste certaine colonne
            all_client_values_modified = st.data_editor(all_client_values_modified,
                                                       column_order=("CODE_GENDER", "PAYMENT_RATE", "AMT_ANNUITY", "AMT_INCOME_TOTAL",
                                                                     "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
                                                                    )
                                                       )
            # Lancer la prédiction par le modèle avec le nouveau dataset
            st.button("Prédire avec modification", 
                      on_click=button_set_value, args=["modification_button", 2])
            
        # Prédiction avec le nouveau dataset
        if st.session_state.modification_button >= 2 :
            
            pred_bis = None
            response_bis = None
            response_bis, pred_dict_bis = request_prediction(all_client_values_modified)
            
            # Gerer la réponse du serveur
            if response_bis == 200 :
    
                # récupèrer la prédiction du modèle, propabilité d'appartenance a la class 0 vas rembourser sont crédit
                pred_bis = pred_dict_bis["predictions"][0][0]
                
                # Produire le graphique
                render_threshold_value(pred_bis)
                
            elif response == 400:
                st.write("La requête au modèle n'a pas marché : Bad Request")
                
            else:
                st.write(f"!! Code résponse inconnue: {response_bis}")
        
    ##-------------------- Feature importance 
        '''
        ## 4) **Comportement du model**
        '''
        render_feature_importance(shap_values_unscaled)
        
    ###------------------
        #st.title(f"Feature importance local du client {SK_ID_CURR}")

        fig, ax = plt.subplots(
                              figsize=(15, 5),
                            )
        
        # Relier l'id client avec l'index du fichier de shap_values_unscaled
        #mask = key_tab["SK_ID_CURR"] == SK_ID_CURR
        #index = key_tab.loc[mask, :].index[0]
    
        # Feature importance local pour un client
        #shap.plots.waterfall(shap_values_unscaled[index], show=False)
    
        #st.pyplot(fig, width="content")
        #del ax, fig

        render_shap_waterfall(shap_values_unscaled, SK_ID_CURR, key_tab )
        
    ##-------------------- Position du client par rapport au reste de la base de données     
        '''
        ## 5) **Etudier la place du clients pour certaine feature**
        '''
        # Utiliser la persistance des valeurs de variable entre les runs
        if "feature" not in st.session_state:
            st.session_state["feature"] = None
            
        st.session_state.feature = st.selectbox("Choisir une feature", main_features,  index=None)
        
        index = get_client_index(SK_ID_CURR, key_tab)
        
        ## Définir deux colonnes
        left_column, right_column = st.columns(2)

        if st.session_state.feature == None:
            st.write("Renseigner le nom de la feature pour afficher les graphiques")

        else :
            with left_column:
                render_violineplot(all_data, key_tab, st.session_state.feature, SK_ID_CURR)
    
            with right_column:
                render_shap_scatter_plot(shap_values_unscaled, SK_ID_CURR, key_tab, st.session_state.feature)

    ##-------------------- Etude bivarié des variables
        '''
        ## 6) **Etudier l'intéraction entre deux features**
        '''
        # Utiliser la persistance des valeurs de variable entre les runs
        if "feature_1" not in st.session_state:
            st.session_state["feature_1"] = None
            
        if "feature_2" not in st.session_state:
            st.session_state["feature_2"] = None

        # Définir les variables à utiliser avec des liste déroulante
        ## Copier la liste de feature de base.
        list_feature_2 = main_features.copy()

        ## définir la valeurs de feature_1
        st.session_state.feature_1 = st.selectbox("Choisir une feature X", main_features, index=None, key="f1")

        ## Empecher l'affectation de la feature 2 si feature 1 n'est pas affecté.
        if st.session_state.feature_1 == None:
            feature_2_disabled = True
        else :
            ## Permettre l'affecation de feature 2
            feature_2_disabled = False
            ## Retirer la valeurs de feature_1 de la liste de séléction de feature 2 pour éviter qu'il soit identique
            list_feature_2.remove(st.session_state.feature_1)
            
        ## définir la valeurs de feature_2
        st.session_state.feature_2 = st.selectbox("Choisir une feature Y", list_feature_2, index=None, disabled = feature_2_disabled, key="f2")

        ## Vérifier que feature_1 a été renseigné
        if st.session_state.feature_1 == None :
            st.write("La valeurs de la première variable X n'est pas renseigner")
            #st.stop()

        ## Vérifier que feature_2 a été renseigné
        elif st.session_state.feature_2 == None :
            st.write("La valeurs de la deuxième variable Y n'est pas renseigner")
            #st.stop()

        ## Afficher le graphique
        else :
            render_bivariate_scatterplot(shap_values_unscaled, SK_ID_CURR, key_tab, st.session_state.feature_1, st.session_state.feature_2)
         
if __name__ == '__main__':
    
    st.set_page_config(layout="wide")
    
    main()