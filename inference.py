import joblib

features= [[26, 0, 180, 110, 29]]

modelo=joblib.load("knn_modelo.pkl")

prediccion = modelo.predict(features)
print("Prediccion:", prediccion[0])
