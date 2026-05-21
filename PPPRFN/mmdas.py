# CODIGO PARA MOSTRAR TABLA DE FALSOS POSITIVOS EN TERMINAL
# Ejecutar en VS Code - Muestra la tabla formateada en consola

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

print("="*70)
print("TOP 5 FALSOS POSITIVOS - DESAPARICIONES")
print("="*70)

np.random.seed(42)
n = 1000

generos = np.random.choice(['Mujer', 'Hombre'], n, p=[0.35, 0.65])
edades = np.random.randint(1, 90, n)
zonas = np.random.choice(['Urbana', 'Rural', 'Suburbana'], n, p=[0.6, 0.2, 0.2])
tiempo = np.random.randint(1, 365, n)
denuncio = np.random.choice(['Si', 'No'], n, p=[0.7, 0.3])

prob = np.zeros(n)
prob[generos == 'Hombre'] += 0.3
prob[edades > 40] += 0.2
prob[zonas == 'Rural'] += 0.2
prob[tiempo > 60] += 0.2
prob[denuncio == 'No'] += 0.1
prob = np.clip(prob, 0.1, 0.9)

df = pd.DataFrame({
    'edad': edades,
    'tiempo': tiempo,
    'genero': generos,
    'zona': zonas,
    'denuncio': denuncio,
    'target': (np.random.random(n) < prob).astype(int)
})

X = df.drop(columns=['target'])
y = df['target']

for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_proba_train = model.predict_proba(X_train)[:, 1]

fp_mask = (y_pred_train == 1) & (y_train == 0)
fp_indices = np.where(fp_mask)[0]
fp_scores = y_proba_train[fp_mask]

if len(fp_indices) > 0:
    top_fp = fp_indices[np.argsort(fp_scores)[-5:][::-1]]
    
    print("\n")
    print("┌─────┬──────────┬──────────┬────────────┬───────────┬───────────┬─────────────┬───────────┐")
    print("│  #  │ ID Caso  │  Score   │    Edad    │  Género   │   Zona    │   Tiempo    │ Denuncio  │")
    print("├─────┼──────────┼──────────┼────────────┼───────────┼───────────┼─────────────┼───────────┤")
    
    for i, idx in enumerate(top_fp, 1):
        caso = X_train[idx]
        score_idx = np.where(fp_indices == idx)[0][0]
        score = fp_scores[score_idx]
        
        edad = f"{int(caso[0])} años"
        genero = "Hombre" if caso[2] == 1 else "Mujer"
        zona_map = {0: "Urbana", 1: "Suburbana", 2: "Rural"}
        zona = zona_map[int(caso[3])]
        tiempo_str = f"{int(caso[1])} días"
        denuncio_str = "Sí" if caso[4] == 1 else "No"
        
        print(f"│  {i}  │ Caso {idx:3d} │ {score:.4f} │ {edad:10} │ {genero:9} │ {zona:9} │ {tiempo_str:11} │ {denuncio_str:9} │")
    
    print("└─────┴──────────┴──────────┴────────────┴───────────┴───────────┴─────────────┴───────────┘")
    
    print("\n")
    print("="*70)
    print("INTERPRETACIÓN")
    print("="*70)
    print("""
¿Qué es un Falso Positivo?
• El modelo predijo "NO ENCONTRADO" (alto riesgo)
• Pero la realidad es que SÍ fue ENCONTRADO

¿Por qué son importantes?
• Pueden revelar FACTORES PROTECTORES no capturados
• Ayudan a identificar limitaciones del modelo
• Pueden sugerir nuevas variables a incluir

Preguntas para cada caso:
1. ¿Tiene alta centralidad en el grafo?
2. ¿Se parece demasiado a los casos no encontrados?
3. ¿Podría representar un hallazgo oculto?
    """)
    
    print("="*70)
    print("\nRESUMEN DE FALSOS POSITIVOS:")
    print(f"  Total de falsos positivos: {len(fp_indices)}")
    print(f"  Score promedio: {np.mean(fp_scores):.4f}")
    print(f"  Score mínimo: {np.min(fp_scores):.4f}")
    print(f"  Score máximo: {np.max(fp_scores):.4f}")
    
else:
    print("\n")
    print("┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("│                                                                                 │")
    print("│                    NO SE ENCONTRARON FALSOS POSITIVOS                            │")
    print("│                                                                                 │")
    print("│  El modelo clasificó correctamente todos los casos encontrados                  │")
    print("│  No hay casos que hayan sido predichos como 'NO ENCONTRADOS'                    │")
    print("│  siendo que en realidad SÍ fueron encontrados.                                  │")
    print("│                                                                                 │")
    print("└─────────────────────────────────────────────────────────────────────────────────┘")

print("="*70)