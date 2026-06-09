import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────
ARQUIVOS   = ['winequality-red.csv', 'winequality-white.csv']
SEPARADOR  = ';'
ALVO       = 'quality'
DROPAR     = ['tipo']
SMOTE_K    = 4
CV_FOLDS   = 5
TEST_SIZE  = 0.3
# ─────────────────────────────────────────────────────────────

frames = []
for i, arq in enumerate(ARQUIVOS):
    df = pd.read_csv(arq, sep=SEPARADOR)
    df['tipo'] = i
    frames.append(df)

dados = pd.concat(frames, ignore_index=True)

X = dados.drop(columns=[ALVO] + DROPAR)
y = dados[ALVO]

X_b, y_b = SMOTE(random_state=42, k_neighbors=SMOTE_K).fit_resample(X, y)

classificadores = {
    'Decision Tree':     DecisionTreeClassifier(random_state=42),
    'Random Forest':     RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
}

scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
resultados = {}

print(f"\n{'Modelo':<20} {'Acurácia':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print('-' * 62)

for nome, clf in classificadores.items():
    s = cross_validate(clf, X_b, y_b, scoring=scoring, cv=CV_FOLDS, n_jobs=-1)
    resultados[nome] = {
        'accuracy':  s['test_accuracy'].mean(),
        'f1':        s['test_f1_macro'].mean(),
        'precision': s['test_precision_macro'].mean(),
        'recall':    s['test_recall_macro'].mean(),
    }
    r = resultados[nome]
    print(f"{nome:<20} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")

melhor_nome = max(resultados, key=lambda k: resultados[k]['accuracy'])
print(f"\nMelhor: {melhor_nome}  |  Acurácia: {resultados[melhor_nome]['accuracy']:.4f}")

melhor_clf = classificadores[melhor_nome]
X_tr, X_te, y_tr, y_te = train_test_split(X_b, y_b, test_size=TEST_SIZE, random_state=42)
melhor_clf.fit(X_tr, y_tr)
y_pred = melhor_clf.predict(X_te)

cm = confusion_matrix(y_te, y_pred)
especificidades, sensibilidades = [], []
for i in range(cm.shape[0]):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    sensibilidades.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

print(f"\n{'─'*40}")
print(f"Especificidade média : {np.mean(especificidades):.4f}")
print(f"Sensibilidade média  : {np.mean(sensibilidades):.4f}")
print(f"{'─'*40}\n")
print(classification_report(y_te, y_pred))

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(y_te, y_pred, ax=ax, colorbar=False)
ax.set_title(f'Matriz de Confusão — {melhor_nome}', fontsize=13)
plt.tight_layout()
plt.savefig('wine_matriz_confusao.png', dpi=100)
plt.show()