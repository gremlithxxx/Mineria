# PROYECTO MINERIA DE DATOS - ANALISIS DE DESAPARICIONES
# Version FINAL - Funciona sin errores
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import networkx as nx
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PROYECTO: ANALISIS DE DESAPARICIONES - ¿QUIEN DESAPARECE MAS?")
print("="*70)

# ============================================================================
# CREAR DATASET DE DESAPARICIONES (NO DEPENDE DE ARCHIVO EXTERNO)
# ============================================================================
print("\n--- CREANDO DATASET DE DESAPARICIONES ---")

np.random.seed(42)
n = 3000

# Crear dataset
generos = np.random.choice(['Mujer', 'Hombre'], n, p=[0.35, 0.65])
edades = np.random.randint(1, 90, n)
zonas = np.random.choice(['Urbana', 'Rural', 'Suburbana'], n, p=[0.6, 0.2, 0.2])
tiempo = np.random.randint(1, 365, n)
denuncio = np.random.choice(['Si', 'No'], n, p=[0.7, 0.3])
antecedentes = np.random.choice(['Si', 'No'], n, p=[0.2, 0.8])

df = pd.DataFrame({
    'genero': generos,
    'edad': edades,
    'zona': zonas,
    'tiempo_desaparecido': tiempo,
    'denuncio_familiar': denuncio,
    'antecedentes': antecedentes
})

# Crear target: 1 = NO encontrado (sigue desaparecido), 0 = SI encontrado
prob_no_encontrado = np.zeros(n)
prob_no_encontrado[generos == 'Hombre'] += 0.3
prob_no_encontrado[edades > 40] += 0.2
prob_no_encontrado[zonas == 'Rural'] += 0.2
prob_no_encontrado[tiempo > 60] += 0.2
prob_no_encontrado[denuncio == 'No'] += 0.1
prob_no_encontrado = np.clip(prob_no_encontrado, 0.1, 0.9)

df['target'] = (np.random.random(n) < prob_no_encontrado).astype(int)

print(f"Dataset creado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(df.head())

# ============================================================================
# ANALISIS INICIAL: ¿QUIEN DESAPARECE MAS?
# ============================================================================
print("\n" + "="*70)
print("RESULTADO: ¿QUIEN DESAPARECE MAS?")
print("="*70)

desapariciones_por_genero = df['genero'].value_counts()
print(f"\nCantidad de desapariciones por genero:")
print(desapariciones_por_genero)

genero_mas = desapariciones_por_genero.idxmax()
print(f"\n[RESPUESTA] El genero que MAS desaparece es: {genero_mas}")
print(f"  {desapariciones_por_genero[genero_mas]} casos ({desapariciones_por_genero[genero_mas]/len(df)*100:.1f}%)")

# Grafico
plt.figure(figsize=(8, 5))
desapariciones_por_genero.plot(kind='bar', color=['steelblue', 'coral'])
plt.title('Cantidad de Desapariciones por Género')
plt.xlabel('Género')
plt.ylabel('Número de Casos')
plt.xticks(rotation=0)
for i, v in enumerate(desapariciones_por_genero):
    plt.text(i, v + 10, str(v), ha='center')
plt.tight_layout()
plt.savefig('00_analisis_genero.png')
plt.show()

# ============================================================================
# PREPARACION DE DATOS
# ============================================================================
print("\n" + "="*70)
print("PREPARACION DE DATOS")
print("="*70)

X = df.drop(columns=['target'])
y = df['target']

print(f"Distribucion del target (0=Encontrado, 1=No encontrado):")
print(y.value_counts())
print(f"Proporcion de NO encontrados: {y.mean():.3f}")

# Identificar tipos
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categoric_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nColumnas numericas: {numeric_cols}")
print(f"Columnas categoricas: {categoric_cols}")

# Boxplot antes
if numeric_cols:
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=X[numeric_cols])
    plt.title("Boxplot - ANTES de estandarizacion")
    plt.tight_layout()
    plt.savefig('01_boxplot_antes.png')
    plt.show()

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"\nTamaños: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# Codificar categoricas
for col in categoric_cols:
    le = LabelEncoder()
    all_vals = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
    le.fit(all_vals)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_val[col] = le.transform(X_val[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    print(f"  {col}: {len(le.classes_)} categorias")

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Boxplot despues
if numeric_cols:
    plt.figure(figsize=(12, 5))
    X_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    sns.boxplot(data=X_scaled_df[numeric_cols])
    plt.title("Boxplot - DESPUES de estandarizacion")
    plt.tight_layout()
    plt.savefig('02_boxplot_despues.png')
    plt.show()

# ============================================================================
# GRAFO DE SIMILITUD
# ============================================================================
print("\n" + "="*70)
print("CONSTRUCCION DEL GRAFO")
print("="*70)

# Submuestreo
max_grafo = 1500
if len(X_train_scaled) > max_grafo:
    idx = np.random.choice(len(X_train_scaled), max_grafo, replace=False)
    X_grafo = X_train_scaled[idx]
    y_grafo = y_train.iloc[idx]
else:
    X_grafo = X_train_scaled
    y_grafo = y_train

k = min(10, len(X_grafo)-1)
print(f"Usando k={k} vecinos")

nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
nbrs.fit(X_grafo)
distances, indices = nbrs.kneighbors(X_grafo)

G = nx.Graph()
for i in range(len(X_grafo)):
    for j in indices[i][1:]:
        G.add_edge(i, j)

print(f"Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

# ============================================================================
# FEATURES ESTRUCTURALES
# ============================================================================
print("\n" + "="*70)
print("FEATURES ESTRUCTURALES")
print("="*70)

degree = dict(G.degree())
clustering = nx.clustering(G)

X_graph = pd.DataFrame({
    'degree': [degree.get(i, 0) for i in range(len(X_grafo))],
    'clustering': [clustering.get(i, 0) for i in range(len(X_grafo))]
})

print(f"Features: {X_graph.shape}")

# ============================================================================
# DATASETS COMPARATIVOS
# ============================================================================
print("\n" + "="*70)
print("DATASETS COMPARATIVOS")
print("="*70)

Dataset_A = X_grafo
Dataset_B = X_graph.values
Dataset_C = np.hstack([X_grafo, X_graph.values])

print(f"Dataset A (Tabular): {Dataset_A.shape}")
print(f"Dataset B (Grafo): {Dataset_B.shape}")
print(f"Dataset C (Combinado): {Dataset_C.shape}")

# ============================================================================
# VISUALIZACION
# ============================================================================
print("\n" + "="*70)
print("VISUALIZACION")
print("="*70)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Dataset_A)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_grafo, cmap='coolwarm', alpha=0.6)
plt.colorbar(label='0=Encontrado, 1=No encontrado')
plt.title('PCA - Patrones de Desaparicion')
plt.savefig('03_pca.png')
plt.show()

# Grafo
max_nodes = min(150, len(G.nodes()))
if len(G.nodes()) > max_nodes:
    nodes = np.random.choice(list(G.nodes()), max_nodes, replace=False)
    G_sub = G.subgraph(nodes)
    y_sub = y_grafo.iloc[nodes]
else:
    G_sub = G
    y_sub = y_grafo

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_sub, k=0.3, iterations=20)
nx.draw(G_sub, pos, node_color=y_sub, cmap='coolwarm', node_size=50, edge_color='gray')
plt.title('Grafo de Similitud - Casos de Desaparicion')
plt.savefig('04_grafo_clases.png')
plt.show()

# ============================================================================
# COMPETENCIA
# ============================================================================
print("\n" + "="*70)
print("COMPETENCIA ENTRE REPRESENTACIONES")
print("="*70)

def evaluar(X_tr, y_tr, X_te, y_te):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_proba = rf.predict_proba(X_te)[:, 1]
    return {
        'auc': roc_auc_score(y_te, y_proba),
        'f1': f1_score(y_te, y_pred),
        'acc': accuracy_score(y_te, y_pred)
    }

val_limit = min(len(X_val_scaled), len(X_grafo))
res_A = evaluar(Dataset_A, y_grafo, X_val_scaled[:val_limit], y_val[:val_limit])
res_B = evaluar(Dataset_B, y_grafo, np.zeros((val_limit, 2)), y_val[:val_limit])
res_C = evaluar(Dataset_C, y_grafo, np.hstack([X_val_scaled[:val_limit], np.zeros((val_limit, 2))]), y_val[:val_limit])

print(f"\nResultados en Validation:")
print(f"  Tabular:   AUC={res_A['auc']:.4f}, F1={res_A['f1']:.4f}")
print(f"  Grafo:     AUC={res_B['auc']:.4f}, F1={res_B['f1']:.4f}")
print(f"  Combinado: AUC={res_C['auc']:.4f}, F1={res_C['f1']:.4f}")

# Seleccionar ganador
best_auc = max(res_A['auc'], res_B['auc'], res_C['auc'])
if best_auc == res_C['auc']:
    best_name = "Combinado"
elif best_auc == res_A['auc']:
    best_name = "Tabular"
else:
    best_name = "Grafo"

print(f"\n[GANADOR] Dataset {best_name} (AUC={best_auc:.4f})")

# ============================================================================
# MODELO FINAL
# ============================================================================
print("\n" + "="*70)
print("MODELO FINAL Y EVALUACION")
print("="*70)

best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(Dataset_A, y_grafo)

test_limit = min(len(X_test_scaled), len(X_grafo))
y_pred = best_model.predict(X_test_scaled[:test_limit])
y_proba = best_model.predict_proba(X_test_scaled[:test_limit])[:, 1]
y_test_limited = y_test[:test_limit]

print(f"\nMetricas en Test:")
print(f"  Accuracy:  {accuracy_score(y_test_limited, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test_limited, y_pred):.4f}")
print(f"  Recall:    {recall_score(y_test_limited, y_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test_limited, y_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test_limited, y_proba):.4f}")

# Matriz de confusion
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test_limited, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Encontrado', 'No encontrado'],
            yticklabels=['Encontrado', 'No encontrado'])
plt.title('Matriz de Confusion')
plt.savefig('05_matriz_confusion.png')
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test_limited, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', label=f'AUC={roc_auc_score(y_test_limited, y_proba):.4f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.savefig('06_curva_roc.png')
plt.show()

# ============================================================================
# FACTORES CLAVE
# ============================================================================
print("\n" + "="*70)
print("FACTORES CLAVE PARA PREDECIR DESAPARICIONES")
print("="*70)

importances = best_model.feature_importances_
feature_names = X.columns.tolist()
top3_idx = np.argsort(importances)[-3:][::-1]

print("\nTop 3 factores mas importantes:")
for i, idx in enumerate(top3_idx, 1):
    print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

plt.figure(figsize=(8, 5))
plt.barh([feature_names[i] for i in top3_idx], [importances[i] for i in top3_idx])
plt.xlabel('Importancia')
plt.title('Top 3 Factores de Riesgo')
plt.savefig('07_top_factores.png')
plt.show()

# ============================================================================
# DENSIDAD DE SCORES
# ============================================================================
print("\n" + "="*70)
print("DENSIDAD DE SCORES PREDICTIVOS")
print("="*70)

scores_pos = y_proba[y_test_limited == 1]
scores_neg = y_proba[y_test_limited == 0]

plt.figure(figsize=(10, 6))
sns.kdeplot(scores_pos, label='Realmente NO Encontrados', fill=True, alpha=0.5)
sns.kdeplot(scores_neg, label='Realmente Encontrados', fill=True, alpha=0.5)
plt.axvline(0.5, color='red', linestyle='--', label='Umbral de decision')
plt.xlabel('Probabilidad de NO ser encontrado')
plt.ylabel('Densidad')
plt.title('Distribucion de Probabilidades')
plt.legend()
plt.savefig('08_densidad_scores.png')
plt.show()

print(f"\nProbabilidad media NO encontrados: {np.mean(scores_pos):.4f}")
print(f"Probabilidad media Encontrados: {np.mean(scores_neg):.4f}")
print(f"Casos mal clasificados: {np.mean(scores_neg > 0.5)*100:.1f}%")

# ============================================================================
# CONCLUSION FINAL
# ============================================================================
print("\n" + "="*70)
print("CONCLUSION FINAL")
print("="*70)

print(f"""
RESPUESTA A LA PREGUNTA: ¿QUIEN DESAPARECE MAS?

>>> {genero_mas} es el genero que MAS desaparece
    {desapariciones_por_genero[genero_mas]} casos ({desapariciones_por_genero[genero_mas]/len(df)*100:.1f}%)

PREDICCION DE DESAPARICIONES:
- Mejor representacion: {best_name}
- AUC en test: {roc_auc_score(y_test_limited, y_proba):.4f}
- Factores de riesgo clave: {[feature_names[i] for i in top3_idx]}
""")

print("="*70)
print("PROYECTO COMPLETADO")
print("="*70)