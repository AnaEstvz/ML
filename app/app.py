import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 



pickle_in = open('../models_class/final_model2.pkl', 'rb') 
modelo_rf = pickle.load(pickle_in) 


st.set_page_config(page_title="Predicción de danzabilidad", page_icon=":music")


seleccion = st.sidebar.selectbox("Selecciona una opción", ["Home", "Predicción"])


if seleccion == "Home":
    st.title("Predicción de la danzabilidad")

    
    with st.expander("¿Qué es esta aplicación?"):
        st.write("Es una primera aproximación para poder predecir cuáles son las canciones más bailables y de esta manera poder añadirlas a la lista de baile")

   
    img = Image.open('baile.jpg')
    st.image(img)


elif seleccion == "Predicción":
    st.title("¡Vamos a predecir la danceability de la canción!")

    nombre_cancion = st.text_input("Ingresa el nombre de la canción")
    
    # Variables predictoras
    popularity = st.slider("Popularity", min_value=-60.0, max_value=0.0, value=-10.0)
    acousticness = st.slider("Energy", min_value=0.0, max_value=1.0, value=0.7)
    instrumentalness = st.slider("Instrumentalness", min_value=0.0, max_value=1.0, value=0.1)
    valence = st.slider("Valence", min_value=0.0, max_value=1.0, value=0.1)
    loudness = st.slider("Loudness", min_value=0.0, max_value=1.0, value=0.1)
    speechiness = st.slider("Speechiness", min_value=0.0, max_value=1.0, value=0.1)
    energy = st.slider("Energy", min_value=0.0, max_value=1.0, value=0.1)
    
    # Tranformaciones 
    loudness_category_numerica = 0 if -47.047 <= loudness < -15 else \
                                  1 if -15 <= loudness < -10 else \
                                  5 if -10 <= loudness < -5 else \
                                  4 if -5 <= loudness < 0 else \
                                  1 if 0 <= loudness <= 3.8 else None

    loudness_category_numerica2 = 0 if loudness_category_numerica == 0 else \
                                  6 if loudness_category_numerica == 4 or loudness_category_numerica == 1 else \
                                  9 if loudness_category_numerica == 5 else None

    speech_cat_num = 0 if speechiness < 0.05 else \
                      1 if 0.05 <= speechiness < 0.1 else \
                      2 if 0.1 <= speechiness < 0.15 else \
                      3 if 0.15 <= speechiness < 0.2 else \
                      4 if 0.2 <= speechiness < 0.25 else \
                      5 if 0.25 <= speechiness < 0.3 else \
                      6 if 0.3 <= speechiness < 0.35 else \
                      7 if 0.35 <= speechiness < 0.4 else \
                      8 if 0.4 <= speechiness < 0.45 else \
                      9 if 0.45 <= speechiness < 0.5 else \
                      10 if 0.5 <= speechiness < 0.55 else \
                      11 if 0.55 <= speechiness < 0.6 else \
                      12 if 0.6 <= speechiness < 0.65 else \
                      13 if 0.65 <= speechiness < 0.7 else \
                      14 if 0.7 <= speechiness < 0.75 else \
                      15 if 0.75 <= speechiness < 0.8 else \
                      16 if 0.8 <= speechiness < 0.85 else \
                      17 if 0.85 <= speechiness < 0.9 else \
                      18 if 0.9 <= speechiness < 0.95 else None

    speech_cat_num2 = 0 if speech_cat_num == 0 else \
                      2 if speech_cat_num == 1 else \
                      4 if speech_cat_num == 17 or speech_cat_num == 16 else \
                      6 if speech_cat_num == 2 or speech_cat_num == 14 else \
                      8 if speech_cat_num == 10 or speech_cat_num == 11 or speech_cat_num == 15 or speech_cat_num == 3 else \
                      10 if speech_cat_num == 9 or speech_cat_num == 12 else \
                      12 if speech_cat_num == 8 or speech_cat_num == 18 else \
                      14 if speech_cat_num == 7 or speech_cat_num == 6 or speech_cat_num == 4 else \
                      16 if speech_cat_num == 13 or speech_cat_num == 5 else None

    energy_cat_num = 0 if energy < 0.2 else \
                     1 if 0.2 <= energy < 0.4 else \
                     2 if 0.4 <= energy < 0.6 else \
                     3 if 0.6 <= energy < 0.8 else \
                     4 if 0.8 <= energy < 0.9999 else None

    energy_cat_num2 = 0 if energy_cat_num == 0 else \
                      2 if energy_cat_num == 1 else \
                      4 if energy_cat_num == 4 else \
                      6 if energy_cat_num == 3 else \
                      7 if energy_cat_num == 2 else None

    # Dataframe con los valores transformados
    datos = pd.DataFrame({
        'popularity': [popularity],
        'acousticness': [acousticness],
        'instrumentalness': [instrumentalness],
        'valence': [valence],
        'loudness_category_numerica2': [loudness_category_numerica2],
        'speech_cat_num2': [speech_cat_num2],
        'energy_cat_num2': [energy_cat_num2]
    })

    # Hacemos la predicción
    prediccion = modelo_rf.predict(datos)

    # Resultado de la predicción
    st.write("Resultado de la predicción:")
    if prediccion[0] < 0.5:
        st.write("La canción tiene poca danceability.")
    else:
        st.write("La canción tiene mucha danceability.")

    # Dataframe con los resultados de la predicción
    df_resultado = pd.DataFrame({
        'Nombre de la canción': [nombre_cancion],
        'Popularity': [popularity],
        'Acousticness': [acousticness],
        'Instrumentalness': [instrumentalness],
        'Valence': [valence],
        'Loudness Category': [loudness_category_numerica2],
        'Speechiness Category': [speech_cat_num2],
        'Energy Category': [energy_cat_num2],
        'Predicción': [prediccion[0]]
    })

    # Mostrar el dataframe con los resultados
    st.write("Resultados de la predicción:")
    st.write(df_resultado)












