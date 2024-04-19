# Importamos librerias necesarias
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Leer csv definitvo con feature engineering
music = pd.read_csv('../data/processed/music_final_class1.csv')

# Dividir en train y test
train_df, test_df = train_test_split(music, test_size=0.20, random_state=42 )

# Guardar en csv
train_df.to_csv('../data/train/train2.csv', index= False)
test_df.to_csv('../data/test/test2.csv', index= False)

# Abrir train para entrenar mi modelo
df = pd.read_csv('../data/train/train2.csv')

# Datos de entrenamiento

X_train = df[['popularity',
       'acousticness',
       'instrumentalness',
        'valence',  'loudness_category_numerica2',
       'speech_cat_num2',
        'energy_cat_num2']]
y_train = df["danceability_category_encoded"]

# Mejor modelo entrenado

steps = [('classifier', XGBClassifier(random_state=42))]
pipeline = Pipeline(steps)


param_dist = {
    'classifier__n_estimators': [100, 150, 200],
    'classifier__max_depth': [9, 10, 15],
    'classifier__learning_rate': [0.04,0.05],
    'classifier__subsample': [0.8, 0.9],
    'classifier__colsample_bytree': [0.8, 0.9],
    'classifier__gamma': [0, 0.1],
    'classifier__reg_alpha': [0.05, 0.3,0.5],
    'classifier__reg_lambda': [ 0.3,0.4]
}


random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=4, n_iter=30, n_jobs=-1, random_state=42, verbose=2)
random_search.fit(X_train, y_train)


final_model1 = random_search.best_estimator_

# Obtener los mejores resultados
best_score = random_search.best_score_
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_

# Entrenar el mejor modelo con los datos de entrenamiento
final_model1.fit(X_train, y_train)

print("Best Score:", best_score)
print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Guardar modelo con pickle
filename = '../models_class/final_model2.pkl'
with open(filename, 'wb') as archivo_salida:
    pickle.dump(final_model1, archivo_salida)

