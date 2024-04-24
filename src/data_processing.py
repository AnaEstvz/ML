import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

music =  pd.read_csv('../data/raw/music_genre.csv')



# Separar target danceability en dos categorías
umbral_danceability = (music['danceability'].max() + music['danceability'].min()) / 2
music['danceability_category'] = music['danceability'].apply(lambda x: 'mucha' if x >= umbral_danceability else 'poca')

conteo_categorias = music['danceability_category'].value_counts()

proporcion_mucha_danceability = conteo_categorias['mucha'] / len(music)
proporcion_poca_danceability = conteo_categorias['poca'] / len(music)

print("Proporción de canciones con mucha danceability:", proporcion_mucha_danceability)
print("Proporción de canciones con poca danceability:", proporcion_poca_danceability)



# Codificación de la categoría
music['danceability_category_encoded'] = music['danceability_category'].map({'poca': 0, 'mucha': 1})


# Tratamos de aumentar la correlación de loudness 
# Definimos rangos para cada grupo y asignamos valor numérico según el grupo
rango1 = (-47.047, -15)
rango2 = (-15, -10)
rango3 = (-10, -5)
rango4 = (-5, 0)
rango5 = (0, 3.8)


def asignar_valor_numerico_loudness(loudness):
    if rango1[0] <= loudness < rango1[1]:
        return 0
    elif rango2[0] <= loudness < rango2[1]:
        return 1
    elif rango3[0] <= loudness < rango3[1]:
        return 5
    elif rango4[0] <= loudness < rango4[1]:
        return 4
    elif rango5[0] <= loudness <= rango5[1]:
        return 1
    else:
        return None  


music['loudness_category_numerica'] = music['loudness'].apply(asignar_valor_numerico_loudness)

# Agrupamos y ordenamos por la media (danceability)
music.groupby('loudness_category_numerica')['danceability'].mean().sort_values(ascending=True)

music.loc[music['loudness_category_numerica'] == 0,"loudness_category_numerica2"]=0

music.loc[(music["loudness_category_numerica"] == 4) | (music["loudness_category_numerica"] == 1), "loudness_category_numerica2"] = 6
music.loc[music['loudness_category_numerica'] == 5,"loudness_category_numerica2"]=9


# Tratamos de aumentar la correlación de valence 
music.loc[music["valence"] < 0.2, "valence_cat_num"] = 0
music.loc[(music["valence"] >= 0.2) & (music["valence"] < 0.4), "valence_cat_num"] = 2
music.loc[(music["valence"] >= 0.4) & (music["valence"] < 0.6), "valence_cat_num"] = 4
music.loc[(music["valence"] >= 0.6) & (music["valence"] < 0.8), "valence_cat_num"] = 6
music.loc[(music["valence"] >= 0.8) & (music["valence"] < 0.994), "valence_cat_num"] = 8



# Tratamos de aumentar la correlación de speechiness 
music.loc[music["speechiness"] < 0.05, "speech_cat_num"] = 0
music.loc[(music["speechiness"] >= 0.05) & (music["speechiness"] < 0.1), "speech_cat_num"] = 1
music.loc[(music["speechiness"] >= 0.1) & (music["speechiness"] < 0.15), "speech_cat_num"] = 2
music.loc[(music["speechiness"] >= 0.15) & (music["speechiness"] < 0.2), "speech_cat_num"] = 3
music.loc[(music["speechiness"] >= 0.2) & (music["speechiness"] < 0.25), "speech_cat_num"] = 4
music.loc[(music["speechiness"] >= 0.25) & (music["speechiness"] < 0.3), "speech_cat_num"] = 5
music.loc[(music["speechiness"] >= 0.3) & (music["speechiness"] < 0.35), "speech_cat_num"] = 6
music.loc[(music["speechiness"] >= 0.35) & (music["speechiness"] < 0.4), "speech_cat_num"] = 7
music.loc[(music["speechiness"] >= 0.4) & (music["speechiness"] < 0.45), "speech_cat_num"] = 8
music.loc[(music["speechiness"] >= 0.45) & (music["speechiness"] < 0.5), "speech_cat_num"] = 9
music.loc[(music["speechiness"] >= 0.5) & (music["speechiness"] < 0.55), "speech_cat_num"] = 10
music.loc[(music["speechiness"] >= 0.55) & (music["speechiness"] < 0.6), "speech_cat_num"] = 11
music.loc[(music["speechiness"] >= 0.6) & (music["speechiness"] < 0.65), "speech_cat_num"] = 12
music.loc[(music["speechiness"] >= 0.65) & (music["speechiness"] < 0.7), "speech_cat_num"] = 13
music.loc[(music["speechiness"] >= 0.7) & (music["speechiness"] < 0.75), "speech_cat_num"] = 14
music.loc[(music["speechiness"] >= 0.75) & (music["speechiness"] < 0.8), "speech_cat_num"] = 15
music.loc[(music["speechiness"] >= 0.8) & (music["speechiness"] < 0.85), "speech_cat_num"] = 16
music.loc[(music["speechiness"] >= 0.85) & (music["speechiness"] < 0.9), "speech_cat_num"] = 17
music.loc[(music["speechiness"] >= 0.9) & (music["speechiness"] < 0.95), "speech_cat_num"] = 18


music.groupby('speech_cat_num')['danceability'].mean().sort_values(ascending=True)

music.loc[music['speech_cat_num'] == 0,"speech_cat_num2"]=0
music.loc[music['speech_cat_num'] == 1,"speech_cat_num2"]=2
music.loc[(music['speech_cat_num'] == 17) | (music['speech_cat_num'] == 16) ,"speech_cat_num2"]=4
music.loc[(music['speech_cat_num'] == 2 )| (music['speech_cat_num'] == 14)  ,"speech_cat_num2"]=6
music.loc[(music['speech_cat_num'] == 10 )| (music['speech_cat_num'] == 11) | (music['speech_cat_num'] == 15) | (music['speech_cat_num'] == 3),"speech_cat_num2"]=8
music.loc[(music['speech_cat_num'] == 9) | (music['speech_cat_num'] == 12) ,"speech_cat_num2"]=10
music.loc[(music['speech_cat_num'] == 8) | (music['speech_cat_num'] == 18) ,"speech_cat_num2"]=12
music.loc[(music['speech_cat_num'] == 7 )| (music['speech_cat_num'] == 6) | (music['speech_cat_num'] == 4) ,"speech_cat_num2"]=14
music.loc[(music['speech_cat_num'] == 13) | (music['speech_cat_num'] == 5) ,"speech_cat_num2"]=16


# Tratamos de aumentar la correlación de energy 
music.loc[music["energy"] < 0.2, "energy_cat_num"] = 0
music.loc[(music["energy"] >= 0.2) & (music["energy"] < 0.4), "energy_cat_num"] = 1
music.loc[(music["energy"] >= 0.4) & (music["energy"] < 0.6), "energy_cat_num"] = 2
music.loc[(music["energy"] >= 0.6) & (music["energy"] < 0.8), "energy_cat_num"] = 3
music.loc[(music["energy"] >= 0.8) & (music["energy"] < 0.9999), "energy_cat_num"] = 4

music.groupby('energy_cat_num')['danceability'].mean().sort_values(ascending=True)

music.loc[music['energy_cat_num'] == 0,"energy_cat_num2"]=0
music.loc[music['energy_cat_num'] == 1,"energy_cat_num2"]=2
music.loc[music['energy_cat_num'] == 4,"energy_cat_num2"]=4
music.loc[music['energy_cat_num'] == 3,"energy_cat_num2"]=6
music.loc[music['energy_cat_num'] == 2,"energy_cat_num2"]=7

# Tratamos de aumentar la correlación de popularity 
music.loc[music["popularity"] < 20.0, "popularity_cat_num"] = 0
music.loc[(music["popularity"] >= 20.0) & (music["popularity"] < 40.0), "popularity_cat_num"] = 1
music.loc[(music["popularity"] >= 40.0) & (music["popularity"] < 60.0), "popularity_cat_num"] = 2
music.loc[(music["popularity"] >= 60.0) & (music["popularity"] < 80.0), "popularity_cat_num"] = 3
music.loc[(music["popularity"] >= 80.0) & (music["popularity"] < 99.1), "popularity_cat_num"] = 4

popularity_danceability_mode = music.groupby('popularity_cat_num')['danceability'].apply(lambda x: x.mode().iloc[0])

popularity_danceability_mode_sorted = popularity_danceability_mode.sort_values(ascending=True)

music.loc[music['popularity_cat_num'] == 0,"popularity_cat_num2"]=0
music.loc[music['popularity_cat_num'] == 3,"popularity_cat_num2"]=2
music.loc[music['popularity_cat_num'] == 1,"popularity_cat_num2"]=4
music.loc[music['popularity_cat_num'] == 2,"popularity_cat_num2"]=6
music.loc[music['popularity_cat_num'] == 4,"popularity_cat_num2"]=8

# Para evitar Nan
music.dropna(subset=['instance_id'], inplace=True)

# Guardo dataset procesado
music.to_csv('../data/processed/music_final_processed.csv', index= False)
