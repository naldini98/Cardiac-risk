# =============================================

# 1. IMPORTAR LIBRERÍAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, f1_score, 
                            make_scorer, roc_curve, auc)
import joblib
import warnings
warnings.filterwarnings('ignore')

# 2. CONFIGURACIÓN DE VISUALIZACIÓN
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
pd.set_option('display.max_columns', None)

# 3. DEFINIR VARIABLES GLOBALES
NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
FILE_PATH = "C:\\Users\\Franco Naldini\\Desktop\\heart.csv"

# 4. CARGAR Y EXPLORAR DATOS
def load_data():
    """Carga y explora el dataset"""
    df = pd.read_csv(FILE_PATH)
    
    print("\n=== INFORMACIÓN DEL DATASET ===")
    print(f"📊 Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print("\n🔍 Primeras 5 filas:")
    print(df.head())
    
    print("\n🧐 Valores nulos por columna:")
    print(df.isnull().sum())
    
    print("\n📈 Estadísticas descriptivas:")
    print(df.describe())
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    print(f"\n♻️ Filas después de eliminar duplicados: {len(df)}")
    
    return df

# 5. ANÁLISIS EXPLORATORIO
def exploratory_analysis(df):
    """Realiza análisis exploratorio con visualizaciones"""
    # Distribución del target
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribución de Diagnóstico Cardíaco\n(0 = Sano, 1 = Enfermo)')
    plt.xlabel('Diagnóstico')
    plt.ylabel('Cantidad de Pacientes')
    plt.show()
    
    # Relación edad vs enfermedad
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='age', hue='target', bins=30, kde=True, multiple='stack')
    plt.title('Distribución de Edad por Diagnóstico Cardíaco')
    plt.xlabel('Edad (años)')
    plt.ylabel('Frecuencia')
    plt.legend(['Sano', 'Enfermo'])
    plt.show()
    
    # Boxplots para variables numéricas
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(NUMERIC_FEATURES, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='target', y=var, data=df)
        plt.title(f'Distribución de {var}')
        plt.xlabel('Diagnóstico')
        plt.ylabel(var)
    plt.tight_layout()
    plt.show()
    
    # Matriz de correlación
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title('Matriz de Correlación entre Variables', pad=20)
    plt.show()

# 6. PREPROCESAMIENTO
def create_preprocessor():
    """Crea el pipeline de preprocesamiento"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)])
    
    return preprocessor

# 7. ENTRENAMIENTO Y EVALUACIÓN
def train_models(X, y):
    """Entrena y evalúa múltiples modelos"""
    preprocessor = create_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)])
        
        # Métricas de evaluación
        accuracy = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        roc_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        
        results[name] = {
            'Accuracy': f"{accuracy.mean():.4f} (±{accuracy.std():.4f})",
            'F1-Score': f"{f1.mean():.4f} (±{f1.std():.4f})",
            'ROC AUC': f"{roc_auc.mean():.4f} (±{roc_auc.std():.4f})"
        }
    
    print("\n=== COMPARACIÓN DE MODELOS ===")
    return pd.DataFrame(results).T

# 8. OPTIMIZACIÓN DEL MODELO
def optimize_model(X_train, y_train):
    """Optimiza hiperparámetros del mejor modelo"""
    preprocessor = create_preprocessor()
    
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\n⚙️ Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)
    
    return grid_search.best_estimator_

# 9. EVALUACIÓN FINAL
def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo final con gráficos"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n📝 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sano', 'Enfermo'], 
                yticklabels=['Sano', 'Enfermo'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
    plt.show()
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    # Importancia de características (VERSIÓN CORREGIDA)
    preprocessor = model.named_steps['preprocessor']
    categorical_transformer = preprocessor.named_transformers_['cat']
    categorical_features_encoded = categorical_transformer.named_steps['onehot'].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = NUMERIC_FEATURES + list(categorical_features_encoded)
    
    importances = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    plt.figure(figsize=(12, 8))
    plt.title("Importancia de las Características (Top 15)")
    plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    plt.show()

# 10. EJECUCIÓN PRINCIPAL
def main():
    # Cargar datos
    df = load_data()
    
    # Análisis exploratorio
    exploratory_analysis(df)
    
    # Preparar datos
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Comparar modelos
    model_comparison = train_models(X, y)
    print(model_comparison)
    
    # Optimizar el mejor modelo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    best_model = optimize_model(X_train, y_train)
    
    # Evaluar modelo final
    evaluate_model(best_model, X_test, y_test)
    
    # Guardar modelo
    joblib.dump(best_model, 'modelo_cardiaco_final.pkl')
    print("\n✅ Modelo guardado como 'modelo_cardiaco_final.pkl'")

if __name__ == "__main__":
    main()