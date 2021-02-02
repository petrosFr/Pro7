import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from lightgbm import LGBMClassifier
import shap
import streamlit.components.v1 as components
import re
import pickle
import altair as alt
import numpy as np

path_info = 'data/data_complet.csv'
model_path = 'data/finalized_model.sav'
shap_path = 'data/shap_model.sav'

@st.cache #mise en cache de la fonction pour exécution unique
def chargement_data(path):
    dataframe = pd.read_csv(path)
    return dataframe

@st.cache #mise en cache de la fonction pour exécution unique
def chargement_info(path):
    dataframe = pd.read_csv(path)
    return dataframe    

@st.cache
def load_logo():
    logo = Image.open("img/logo.PNG") 
    return logo

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

@st.cache
def load_model():
    m = pickle.load(open(model_path, 'rb'))
    return m
@st.cache
def load_shap(X):
    sh = pickle.load(open(shap_path, 'rb'))
    ex = load_model()
    sh = ex.shap_values(X)
    return sh, ex

def draw_chart(x1,x2,x3,f):
    # initialize list of lists 
    data = [['Client', x1], ['Clients sovable', x2], ['Clients non-sovable', x3]] 
    df = pd.DataFrame(data, columns = ['Name', f]) 
    chart_v1 = alt.Chart(df).mark_bar(size=20).encode(
    x='Name',
    y=f).configure_view(
    strokeWidth=4.0,
    height=250,
    width=300,
    )
    st.write("", "", chart_v1)


# Affichage du titre et du sous-titre
st.title("Implémenter un modèle de scoring")
st.markdown("<i>Projet 7 pour le parcours Data Scientist OpenClassRoom</i>", unsafe_allow_html=True)

# Chargement du logo
logo = load_logo()
st.sidebar.image(logo,width=200)

dataframe = chargement_data(path_info)
liste_id = dataframe['SK_ID_CURR'].tolist()


#affichage formulaire
st.sidebar.title('Dashboard Scoring Credit')
st.sidebar.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")
id_input = st.sidebar.selectbox('Veuillez saisir l\'identifiant d\'un client:',liste_id)

dataframe = dataframe.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X = dataframe.drop(columns=['TARGET','NAME_FAMILY_STATUS','solvability'])
y = dataframe['TARGET']

shap_values, ex = load_shap(X)

dataframe['EXT_SOURCE_1'] = pd.to_numeric(dataframe['EXT_SOURCE_1'], errors='coerce')
dataframe['EXT_SOURCE_1'] = dataframe['EXT_SOURCE_1'].fillna(0)

dataframe['EXT_SOURCE_2'] = pd.to_numeric(dataframe['EXT_SOURCE_2'], errors='coerce')
dataframe['EXT_SOURCE_2'] = dataframe['EXT_SOURCE_2'].fillna(0)

dataframe['EXT_SOURCE_3'] = pd.to_numeric(dataframe['EXT_SOURCE_3'], errors='coerce')
dataframe['EXT_SOURCE_3'] = dataframe['EXT_SOURCE_3'].fillna(0)


if (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API
    selected_id = dataframe[dataframe['SK_ID_CURR']==int(id_input)]
    prob = round(selected_id['solvability'].values[0],5)*100
    chaine = 'Le client n° '+str(id_input)+' a de risque de défaut **' + str(prob) + '**%'
    indexx = dataframe.index[dataframe['SK_ID_CURR']==int(id_input)].tolist()
    st.markdown(chaine)
    st_shap(shap.force_plot(ex.expected_value[1], shap_values[1][indexx[0],:], X.iloc[indexx[0],:]))

    if st.sidebar.checkbox("Afficher les informations du client?"):
        st.write("Statut famille :**", selected_id["NAME_FAMILY_STATUS"].iloc[0], "**")
        st.write("Nombre d'enfant(s) :**", selected_id["CNT_CHILDREN"].iloc[0], "**")
        st.write("Age client :**", int(selected_id["DAYS_BIRTH"].values / 365), "**", "ans.")
        st.write("DAYS_LAST_PHONE_CHANGE", selected_id['DAYS_LAST_PHONE_CHANGE'].iloc[0])
        st.write("AMT CREDIT", selected_id['AMT_CREDIT'].iloc[0])
        st.write("AMT INCOME TOTAL", selected_id['AMT_INCOME_TOTAL'].iloc[0])
        st.write("AMT ANNUITY", selected_id['AMT_ANNUITY'].iloc[0])

fig, axs = plt.subplots(nrows=1, ncols=1)
shap.summary_plot(shap_values[0], X, plot_type='bar')
st.sidebar.pyplot(fig)

fig1, ax = plt.subplots(nrows=1, ncols=1)
shap.summary_plot(shap_values[0], X)
st.sidebar.pyplot(fig1)

vals= np.abs(shap_values[0])
feature_importance = pd.DataFrame(list(zip(X.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
val = feature_importance['col_name'].head(6)

f = st.selectbox('Choose your feature',val.tolist())
st.write(f)
if (f=='DAYS_BIRTH'):
    x1 = int(selected_id['DAYS_BIRTH'].iloc[0]/365)
    x2 = dataframe.loc[dataframe['TARGET'] == 1, ('DAYS_BIRTH')].values/365
    x3 = dataframe.loc[dataframe['TARGET'] == 0, ('DAYS_BIRTH')].values/365
    x2 = int(x2.mean())
    x3 = int(x3.mean())
    draw_chart(x1,x2,x3,f)
elif(f=='EXT_SOURCE_1'):
    x1 = selected_id[f].iloc[0]
    x2 = dataframe.loc[dataframe['TARGET'] == 1, f]
    x3 = dataframe.loc[dataframe['TARGET'] == 0, f]
    x2 = x2.mean()
    x3 = x3.mean()
    draw_chart(x1,x2,x3,f)
elif(f=='EXT_SOURCE_2'):
    x1 = selected_id[f].iloc[0]
    x2 = dataframe.loc[dataframe['TARGET'] == 1, f]
    x3 = dataframe.loc[dataframe['TARGET'] == 0, f]
    x2 = x2.mean()
    x3 = x3.mean()
    draw_chart(x1,x2,x3,f)
elif(f=='EXT_SOURCE_3'):
    x1 = selected_id[f].iloc[0]
    print(x1)
    x2 = dataframe.loc[dataframe['TARGET'] == 1, f]
    x3 = dataframe.loc[dataframe['TARGET'] == 0, f]
    x2 = x2.mean()
    x3 = x3.mean()
    draw_chart(x1,x2,x3,f)
else:
    x1 = int(selected_id[f].iloc[0])
    st.write(x1)
    x2 = dataframe.loc[dataframe['TARGET'] == 1, f]
    x3 = dataframe.loc[dataframe['TARGET'] == 0, f]
    x2 = int(x2.mean())
    x3 = int(x3.mean())
    draw_chart(x1,x2,x3,f)
    