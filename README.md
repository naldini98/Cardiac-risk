# Cardiac-risk
Sistema de predicción de riesgo cardíaco con Machine Learning  

**Autor:** Franco Naldini  

---

## ¿Qué hace este proyecto?
Desarrollé un modelo predictivo de machine learning que analiza factores clínicos para evaluar el riesgo de enfermedades cardiovasculares.

### Módulo de Diagnóstico Predictivo
- Análisis automatizado de 14 variables médicas (edad, colesterol, presión arterial, etc.).
- Clasificación de riesgo:
  - Bajo riesgo → Predicción: 0  
  - Alto riesgo → Predicción: 1
- Métricas avanzadas: Accuracy, AUC-ROC, F1-Score.

### Módulo de Visualización
- Gráficos interactivos:
  - Distribución de diagnósticos por edad.
  - Matriz de correlación entre variables.
  - Importancia de características en el modelo.
- Reporte médico automatizado con:
  - Probabilidad de riesgo individual.
  - Factores clínicos más influyentes.

---

## Tecnologías usadas
- Lenguaje: Python  
- Librerías principales: Pandas | Scikit-learn | Matplotlib | Seaborn | Joblib  
- Algoritmos implementados:  
  - Random Forest  
  - Regresión Logística  
  - Gradient Boosting  
  - SVM  

---

## ¿Cómo ejecutarlo?
Ejecutar el script principal:

```bash
python Riesgo_Cardíaco.py
