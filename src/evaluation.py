import pandas as pd
import pickle 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Importar el modelo con pickle 
filename = '../models_class/final_model.pkl'

with open(filename, "rb") as archivo_entrada:
    final_model = pickle.load(archivo_entrada)

# Leer df test para hacer predicciones
df1 = pd.read_csv('../data/test/test2.csv')

# Datos test
X_test = df1[['popularity',
       'acousticness',
       'instrumentalness',
        'valence',  'loudness_category_numerica2',
       'speech_cat_num2',
        'energy_cat_num2']]
y_test = df1["danceability_category_encoded"]

# Hacemos predicciones 

pred_test = final_model.predict(X_test)
pred_proba_test = final_model.predict_proba(X_test)

print("Resultados sobre TEST")
print('Accuracy:', accuracy_score(y_test, pred_test))
print('Precision:', precision_score(y_test, pred_test))
print('Recall:', recall_score(y_test, pred_test))
print('F1 Score:', f1_score(y_test, pred_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, pred_test))