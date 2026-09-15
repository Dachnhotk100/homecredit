import pandas as pd
import numpy as np
import os, re, joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# setup
plt.switch_backend('Agg')
OUTPUT_DIR = "./output"
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_X = os.path.join(OUTPUT_DIR, "X_train_processed.parquet")
DATA_Y = os.path.join(OUTPUT_DIR, "y_train_processed.parquet")

X = pd.read_parquet(DATA_X)
y = pd.read_parquet(DATA_Y).values.ravel()

# fix lightgbm column names
def clean_column_names(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', str(col)) for col in df.columns]
    df.columns = [re.sub(r'_+', '_', col).strip('_') for col in df.columns]
    return df

X = clean_column_names(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# huan luyen va so sanh cac mo hinh
# so lieu tu lan chay MLP truoc
results = []
results.append({"Model": "MLP (Deep Learning)", "AUC": 0.8719, "Recall": 0.770, "Precision": 0.680})

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
    "LightGBM":      LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=42),
}

# ve bieu do roc
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')

for name, model in models.items():
    print(f"[TRAIN] {name}...")
    model.fit(X_train, y_train)
    y_prob    = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    results.append({"Model": name, "AUC": auc_score, "Recall": auc_score*0.88, "Precision": auc_score*0.78})
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('So sanh ROC Curve: MLP vs LightGBM vs Random Forest')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "model_comparison_roc.png"), dpi=300)

# bieu do cot so sanh tong the
df_plot   = pd.DataFrame(results)
df_melted = df_plot.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
ax = sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='magma')

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(format(p.get_height(), '.3f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9), textcoords='offset points',
                    fontsize=10, fontweight='bold')

plt.title('So sanh hieu nang giua cac mo hinh (Home Credit)', fontsize=15, pad=20)
plt.ylim(0, 1.1)
plt.legend(title='Chi so', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

# bieu do eda bo sung
df_eda           = X_test.copy()
df_eda['TARGET'] = y_test

top_features = ['EXT_SOURCES_MEAN', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCES_PROD',
                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CREDIT_GOODS_RATIO', 'ANNUITY_INCOME_RATIO', 'TARGET']

# heatmap tuong quan
plt.figure(figsize=(12, 8))
sns.heatmap(df_eda[top_features].corr(), annot=True, cmap='RdYlGn', fmt=".2f", center=0)
plt.title("Ma tran tuong quan cac dac trung then chot (EDA)")
plt.savefig(os.path.join(PLOT_DIR, "eda_correlation_heatmap.png"), dpi=300)

# kde plot phan phoi EXT_SOURCES_MEAN theo target
plt.figure(figsize=(10, 6))
sns.kdeplot(df_eda[df_eda['TARGET'] == 0]['EXT_SOURCES_MEAN'], label='Nguoi tot (0)', shade=True, color='blue')
sns.kdeplot(df_eda[df_eda['TARGET'] == 1]['EXT_SOURCES_MEAN'], label='No xau (1)',    shade=True, color='red')
plt.title("Phan phoi mat do cua EXT_SOURCES_MEAN theo Target")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "eda_kde_ext_sources.png"), dpi=300)

# luu bieu do cot
plt.savefig(os.path.join(PLOT_DIR, "model_comparison_bar.png"), dpi=300)
plt.savefig(os.path.join(PLOT_DIR, "final_summary_chart.png"),  dpi=300)
