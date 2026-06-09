import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. MONTAR O DATASET
red   = pd.read_csv('winequality-red.csv',   sep=';')
white = pd.read_csv('winequality-white.csv', sep=';')
red['tipo'] = 0 
white['tipo'] = 1 

dados = pd.concat([red, white], ignore_index=True)
print(f"Dataset montado: {dados.shape[0]} amostras, {dados.shape[1]} colunas")
print(f"\nDistribuição da classe 'quality':")
print(dados['quality'].value_counts().sort_index())

X = dados.drop(columns=['quality']).values
y = dados['quality'].values

# 2. NORMALIZAÇÃO

print("\n--- NORMALIZAÇÃO (StandardScaler) ---")
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
col_names = dados.drop(columns=['quality']).columns.tolist()
print(f"  '{col_names[0]}': média {X[:,0].mean():.2f} → {X_norm[:,0].mean():.4f} | "
      f"std {X[:,0].std():.2f} → {X_norm[:,0].std():.4f}")


# 3. BALANCEAMENTO (SMOTE)
print("\n--- BALANCEAMENTO (SMOTE) ---")
resampler = SMOTE(random_state=42, k_neighbors=4)
X_b, y_b = resampler.fit_resample(X_norm, y)
print(f"  Antes: {dict(Counter(y))}")
print(f"  Após:  {dict(Counter(y_b))}")


# 4. HIPERPARAMETRIZAÇÃO (RandomizedSearchCV → Random Forest)
print("\n--- HIPERPARAMETRIZAÇÃO (RandomizedSearchCV) ---")
rf_grid = {
    'n_estimators':      [int(x) for x in np.linspace(10, 100, 10)],
    'criterion':         ['gini', 'entropy'],
    'max_depth':         [int(x) for x in np.linspace(10, 50, 5)],
    'min_samples_split': [2, 5, 10],
    'max_features':      ['sqrt', 'log2'],
}
rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_grid,
    n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=1
)
rf_search.fit(X_b, y_b)
print(f"\n  Melhores parâmetros: {rf_search.best_params_}")

# 5. DEFINIR OS 3 CLASSIFICADORES
classificadores = {
    'Decision Tree':     DecisionTreeClassifier(random_state=42),
    'Random Forest':     RandomForestClassifier(**rf_search.best_params_, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=20, max_depth=4, random_state=42),
}

# 6. AVALIAÇÃO CRUZADA
print("\n--- AVALIAÇÃO CRUZADA ---")
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
resultados = {}

for nome, clf in classificadores.items():
    print(f"\n  Avaliando: {nome} ...")
    scores = cross_validate(clf, X_b, y_b, scoring=scoring, cv=5, n_jobs=-1)
    resultados[nome] = {
        'accuracy':  scores['test_accuracy'].mean(),
        'f1':        scores['test_f1_macro'].mean(),
        'precision': scores['test_precision_macro'].mean(),
        'recall':    scores['test_recall_macro'].mean(),
    }
    print(f"    Acurácia:  {resultados[nome]['accuracy']:.4f}")
    print(f"    F1-Score:  {resultados[nome]['f1']:.4f}")
    print(f"    Precision: {resultados[nome]['precision']:.4f}")
    print(f"    Recall:    {resultados[nome]['recall']:.4f}")

# 7. SELECIONAR O MELHOR CLASSIFICADOR
melhor_nome = max(resultados, key=lambda k: resultados[k]['accuracy'])
print(f"\n>>> MELHOR CLASSIFICADOR: {melhor_nome}")
print(f"    Acurácia: {resultados[melhor_nome]['accuracy']:.4f}")


# 8. TREINAR E CALCULAR MÉTRICAS FINAIS
melhor_clf = classificadores[melhor_nome]
X_tr, X_te, y_tr, y_te = train_test_split(X_b, y_b, test_size=0.3, random_state=42)
melhor_clf.fit(X_tr, y_tr)
y_pred = melhor_clf.predict(X_te)

# Especificidade e Sensibilidade por classe
cm = confusion_matrix(y_te, y_pred)
especificidades, sensibilidades = [], []
for i in range(cm.shape[0]):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    sensibilidades.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

print(f"\n--- MÉTRICAS FINAIS ({melhor_nome}) ---")
print(f"  Acurácia Global:       {resultados[melhor_nome]['accuracy']:.4f}")
print(f"  Especificidade média:  {np.mean(especificidades):.4f}")
print(f"  Sensibilidade média:   {np.mean(sensibilidades):.4f}")
print(f"  F1-Score:              {resultados[melhor_nome]['f1']:.4f}")
print(f"\n  Relatório por classes:\n{classification_report(y_te, y_pred)}")

# 9. MATRIZ DE CONFUSÃO
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(y_te, y_pred, ax=ax, colorbar=False)
ax.set_title(f'Matriz de Confusão - {melhor_nome}\nWine Quality', fontsize=13)
plt.tight_layout()
plt.savefig('wine_matriz_confusao.png', dpi=100)
plt.show()
print("Salvo: wine_matriz_confusao.png")
