import joblib

features= [[5.1, 3.5, 1.4, 0.2]]

modelo=joblib.load("modelo.pkl")

prediccion = modelo.predict(features)
print("Prediccion:", prediccion[0])
