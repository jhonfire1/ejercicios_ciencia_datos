import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

data = {
'horas_estudio': [2, 5, 3, 8, 1, 7, 4, 6, 2, 9],
'asistencia': [60, 90, 70, 95, 50, 85, 75, 80, 65, 98],
'tareas': [1, 5, 2, 6, 1, 5, 3, 4, 2, 6],
'promedio': [4.0, 5.5, 4.5, 6.0, 3.8, 5.8, 5.0, 5.3, 4.2, 6.2],
'aprueba': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X=df[['horas_estudio','asistencia','tareas','promedio']]
y=df['aprueba']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Predicciones:", accuracy_score(y_test, y_pred))  
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("")
plt.figure(figsize=(12, 8))
tree.plot_tree(
modelo,
feature_names=X.columns, class_names=['No Aprueba', 'Aprueba'], filled=True)

nuevo=[[5,80, 4, 5.2]]
pred=modelo.predict(nuevo)


print("\nuevo estudiante:")
print("Predicción:", "Aprueba" if pred[0] == 1 else "No Aprueba")

tree.plot_tree(
modelo,
feature_names=X.columns,
class_names=['No Aprueba', 'Aprueba'],
filled=True,
rounded=True, # bordes redondeados
fontsize=10, # tamaño texto
proportion=True # tamaños proporcionales
)
plt.title("Árbol de decisión - Rendimiento estudiantes")
plt.show()