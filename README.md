# Cardiac-risk
🫀 Sistema de predicción de riesgo cardíaco con Machine Learning 

Autor: Franco Naldini

📌 ¿Qué hace este proyecto?
Desarrollé un modelo predictivo de machine learning que analiza factores clínicos para evaluar el riesgo de enfermedades cardiovasculares, con:

🔍 Módulo de Diagnóstico Predictivo
Análisis automatizado de 14 variables médicas (edad, colesterol, presión arterial, etc.).

Clasificación de riesgo:

✅ Bajo riesgo (Predicción: 0)

❌ Alto riesgo (Predicción: 1)

Métricas avanzadas: Exactitud (accuracy), AUC-ROC, F1-Score.

📊 Módulo de Visualización

Gráficos interactivos:

Distribución de diagnósticos por edad.

Matriz de correlación entre variables.

Importancia de características en el modelo.

Reporte médico automatizado con:

Probabilidad de riesgo individual.

Factores clínicos más influyentes.

🛠️ Tecnologías usadas
Lenguaje: Python

Librerías principales:

Pandas | Scikit-learn | Matplotlib | Seaborn | Joblib

Algoritmos implementados:

python
Random Forest | Regresión Logística | Gradient Boosting | SVM

📥 ¿Cómo ejecutarlo?

Ejecutar el script principal:

python Riesgo_Cardíaco.py

El sistema generará:

✅ Modelo entrenado (modelo_cardiaco_final.pkl).

📊 Reporte visual (gráficos en carpeta /results).

📊 Resultados destacados

Modelo	Exactitud (Accuracy)	AUC-ROC
Random Forest	92.3%	0.95
Gradient Boosting	90.1%	0.93
Regresión Logística	88.5%	0.91
Resultados obtenidos con validación cruzada (5 folds).

📞 Contacto
✉️ Email: franco.naldini@outlook.com

📱 Teléfono: (549) 2604392862

🌐 Repositorio: github.com/naldinif98/Cardiac-risk
