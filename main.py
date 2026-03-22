import os, re, json, pickle, warnings, gc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
warnings.filterwarnings('ignore')

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)

from google.colab import drive
drive.mount('/content/drive')

folder_path = "/content/drive/MyDrive/Raw Data"

fakeAndTrueDataFrame     = pd.read_csv(f"{folder_path}/fake_or_real_news.csv")
fakeDataFrame            = pd.read_csv(f"{folder_path}/FAKE AND TRUE DATASET/Fake.csv")
trueDataFrame            = pd.read_csv(f"{folder_path}/FAKE AND TRUE DATASET/True.csv")
realNewsNetDataFrame     = pd.read_csv(f"{folder_path}/FakeNewsNet/BuzzFeed_real_news_content.csv")
fakeNewsNetDataFrame     = pd.read_csv(f"{folder_path}/FakeNewsNet/BuzzFeed_fake_news_content.csv")
realPolNewsNetDataFrame  = pd.read_csv(f"{folder_path}/FakeNewsNet/PolitiFact_real_news_content.csv")
fakePolNewsNetDataFrame  = pd.read_csv(f"{folder_path}/FakeNewsNet/PolitiFact_fake_news_content.csv")

for frame in [fakeAndTrueDataFrame, fakeDataFrame, trueDataFrame,
              realNewsNetDataFrame, fakeNewsNetDataFrame,
              realPolNewsNetDataFrame, fakePolNewsNetDataFrame]:
    frame.fillna("", inplace=True)

firstandsecond = pd.DataFrame()
firstandsecond['Content'] = fakeAndTrueDataFrame['title'] + " " + fakeAndTrueDataFrame['text']
firstandsecond['Label']   = fakeAndTrueDataFrame['label'].map({'FAKE': 0, 'REAL': 1})

third          = pd.DataFrame()
third['Content'] = fakeDataFrame['title'] + " " + fakeDataFrame['text']
third['Label']   = 0

fourth         = pd.DataFrame()
fourth['Content'] = trueDataFrame['title'] + " " + trueDataFrame['text']
fourth['Label']   = 1

fifth          = pd.DataFrame()
fifth['Content'] = realNewsNetDataFrame['title'] + " " + realNewsNetDataFrame['text']
fifth['Label']   = 1

sixth          = pd.DataFrame()
sixth['Content'] = fakeNewsNetDataFrame['title'] + " " + fakeNewsNetDataFrame['text']
sixth['Label']   = 0

seventh        = pd.DataFrame()
seventh['Content'] = realPolNewsNetDataFrame['title'] + " " + realPolNewsNetDataFrame['text']
seventh['Label']   = 1

eighth         = pd.DataFrame()
eighth['Content'] = fakePolNewsNetDataFrame['title'] + " " + fakePolNewsNetDataFrame['text']
eighth['Label']   = 0

mainDataFrame = pd.concat(
    [firstandsecond, third, fourth, fifth, sixth, seventh, eighth],
    ignore_index=True
)
mainDataFrame.drop_duplicates(subset='Content', inplace=True)
mainDataFrame.dropna(inplace=True)
mainDataFrame.reset_index(drop=True, inplace=True)

print("Dataset shape:", mainDataFrame.shape)
print(mainDataFrame['Label'].value_counts())

os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_graphs', exist_ok=True)

matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':     0.8,
    'axes.grid':          True,
    'grid.linestyle':     '--',
    'grid.linewidth':     0.5,
    'grid.alpha':         0.4,
    'lines.linewidth':    1.5,
})

HATCHES    = ['', '///', '...', 'xxx', '\\\\\\', '+++']
GRAYS      = ['0.10', '0.30', '0.50', '0.65', '0.80', '0.90']
LINESTYLES = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(5,2))]
MARKERS    = ['o', 's', '^', 'D', 'v', 'P']


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>',          '', text)
    text = re.sub(r'\[.*?\]',        '', text)
    text = re.sub(r'[^a-z\s]',       '', text)
    text = re.sub(r'\s+',            ' ', text).strip()
    return text


df = mainDataFrame.copy()
df = df[['Content', 'Label']].dropna()
df['Content'] = df['Content'].apply(clean_text)
df = df[df['Content'].str.len() > 10]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

del mainDataFrame
gc.collect()

print(f"Dataset: {df.shape} | Labels: {df['Label'].value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    df['Content'], df['Label'],
    test_size=0.2, random_state=42, stratify=df['Label']
)
y_train = y_train.values
y_test  = y_test.values
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    min_df=2,
    dtype=np.float32
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)
print(f"Feature matrix: {X_train_vec.shape}")

pickle.dump(tfidf, open('saved_models/tfidf.pkl', 'wb'))

svd        = TruncatedSVD(n_components=100, random_state=42)
normalizer = Normalizer(copy=False)
X_train_lsa = normalizer.fit_transform(svd.fit_transform(X_train_vec))
X_test_lsa  = normalizer.transform(svd.transform(X_test_vec))
pickle.dump(svd,        open('saved_models/svd.pkl',        'wb'))
pickle.dump(normalizer, open('saved_models/normalizer.pkl', 'wb'))
gc.collect()

full_metrics   = {}
cv_results     = {}
trained_models = {}
cm_store       = {}


def run_cv(name, model, X_cv, y_cv, n_splits=5):
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X_cv, y_cv, cv=skf,
        scoring={
            'accuracy':  'accuracy',
            'precision': 'precision_weighted',
            'recall':    'recall_weighted',
            'f1':        'f1_weighted'
        },
        n_jobs=-1, return_train_score=False
    )
    result = {
        'CV_Acc_Mean':  scores['test_accuracy'].mean(),
        'CV_Acc_Std':   scores['test_accuracy'].std(),
        'CV_F1_Mean':   scores['test_f1'].mean(),
        'CV_F1_Std':    scores['test_f1'].std(),
        'CV_Prec_Mean': scores['test_precision'].mean(),
        'CV_Prec_Std':  scores['test_precision'].std(),
        'CV_Rec_Mean':  scores['test_recall'].mean(),
        'CV_Rec_Std':   scores['test_recall'].std(),
        'CV_F1_All':    scores['test_f1'].tolist(),
    }
    cv_results[name] = result
    print(f"  {name} CV F1 = {result['CV_F1_Mean']:.4f} +/- {result['CV_F1_Std']:.4f}")
    gc.collect()
    return result


def evaluate(name, model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_test,  y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test,    y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test,        y_pred, average='weighted', zero_division=0)

    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_te)[:, 1])
    except Exception:
        auc = None

    full_metrics[name]   = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC': auc}
    trained_models[name] = model
    cm_store[name]       = confusion_matrix(y_test, y_pred)

    print(f"  {name} -> Acc={acc:.4f} F1={f1:.4f}" + (f" AUC={auc:.4f}" if auc else ""))
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real'], zero_division=0))
    gc.collect()


print("\nLogistic Regression")
logreg = LogisticRegression(C=5, max_iter=1000, solver='lbfgs', n_jobs=-1)
run_cv("LogReg", logreg, X_train_vec, y_train)
evaluate("LogReg", logreg, X_train_vec, y_train, X_test_vec)

print("\nLinear SVM")
svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3)
run_cv("LinearSVM", svm, X_train_vec, y_train)
evaluate("LinearSVM", svm, X_train_vec, y_train, X_test_vec)

print("\nNaive Bayes")
nb = MultinomialNB(alpha=0.1)
run_cv("NaiveBayes", nb, X_train_vec, y_train)
evaluate("NaiveBayes", nb, X_train_vec, y_train, X_test_vec)

print("\nGradient Boosting")
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
run_cv("GBM", gbm, X_train_lsa, y_train)
evaluate("GBM", gbm, X_train_lsa, y_train, X_test_lsa)

del X_train_lsa, X_test_lsa
gc.collect()

print("\nVoting Ensemble")
lr2      = LogisticRegression(C=5, max_iter=1000, solver='lbfgs', n_jobs=-1)
svm2     = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3)
nb2      = MultinomialNB(alpha=0.1)
ensemble = VotingClassifier(estimators=[('lr', lr2), ('svm', svm2), ('nb', nb2)], voting='soft')
run_cv("Ensemble", ensemble, X_train_vec, y_train)
evaluate("Ensemble", ensemble, X_train_vec, y_train, X_test_vec)

for name, model in trained_models.items():
    pickle.dump(model, open(f'saved_models/{name}.pkl', 'wb'))
    print(f"  saved_models/{name}.pkl saved")

del X_train_vec, X_test_vec
gc.collect()

print("\nBiLSTM")

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

MAX_WORDS = 20000
MAX_LEN   = 100
EMBED_DIM = 64

tok = KerasTokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tok.fit_on_texts(X_train)

X_train_seq = pad_sequences(tok.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
X_test_seq  = pad_sequences(tok.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding='post')

bilstm_model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

bilstm_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy']
)
bilstm_model.summary()

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5)
]

history = bilstm_model.fit(
    X_train_seq, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

y_pred_dl = (bilstm_model.predict(X_test_seq, batch_size=256) > 0.5).astype(int).flatten()

acc_dl  = accuracy_score(y_test,  y_pred_dl)
prec_dl = precision_score(y_test, y_pred_dl, average='weighted', zero_division=0)
rec_dl  = recall_score(y_test,    y_pred_dl, average='weighted', zero_division=0)
f1_dl   = f1_score(y_test,        y_pred_dl, average='weighted', zero_division=0)

full_metrics['BiLSTM'] = {'Accuracy': acc_dl, 'Precision': prec_dl, 'Recall': rec_dl, 'F1': f1_dl, 'AUC': None}
cv_results['BiLSTM']   = {'CV_F1_Mean': None, 'CV_F1_Std': None, 'CV_Acc_Mean': None,
                           'CV_Acc_Std': None, 'CV_Prec_Mean': None, 'CV_Prec_Std': None,
                           'CV_Rec_Mean': None, 'CV_Rec_Std': None, 'CV_F1_All': None}
cm_store['BiLSTM']     = confusion_matrix(y_test, y_pred_dl)

print(f"BiLSTM -> Acc={acc_dl:.4f} F1={f1_dl:.4f}")

bilstm_model.save('saved_models/BiLSTM_model.keras')
with open('saved_models/bilstm_tokenizer.json', 'w') as f:
    json.dump(tok.to_json(), f)

gc.collect()

print("\nFINAL RESULTS")
metrics_df = pd.DataFrame(full_metrics).T.sort_values('F1', ascending=False)
print(metrics_df.round(4).to_string())
best = metrics_df['F1'].idxmax()
print(f"\nBest model: {best} | F1={metrics_df.loc[best,'F1']:.4f} | Acc={metrics_df.loc[best,'Accuracy']:.4f}")

print("\n5-FOLD CV RESULTS")
for m, r in cv_results.items():
    if r['CV_F1_Mean']:
        print(f"  {m:<12} CV F1 = {r['CV_F1_Mean']:.4f} +/- {r['CV_F1_Std']:.4f}")
    else:
        print(f"  {m:<12} CV F1 = N/A (validation split used)")

cv_df_rows = []
for m, r in cv_results.items():
    cv_df_rows.append({
        'Model':      m,
        'CV_F1_Mean': f"{r['CV_F1_Mean']:.4f}" if r['CV_F1_Mean'] else 'N/A',
        'CV_F1_Std':  f"{r['CV_F1_Std']:.4f}"  if r['CV_F1_Std']  else 'N/A',
        'Test_F1':    f"{full_metrics[m]['F1']:.4f}",
        'Test_Acc':   f"{full_metrics[m]['Accuracy']:.4f}",
    })
pd.DataFrame(cv_df_rows).to_csv('saved_models/cv_results.csv', index=False)

model_names   = list(full_metrics.keys())
graph_metrics = {
    'Accuracy':  [full_metrics[m]['Accuracy']    for m in model_names],
    'Precision': [full_metrics[m]['Precision']   for m in model_names],
    'Recall':    [full_metrics[m]['Recall']      for m in model_names],
    'F1-Score':  [full_metrics[m]['F1']          for m in model_names],
    'ROC-AUC':   [full_metrics[m]['AUC'] or 0.0  for m in model_names],
}
cv_f1_means = [cv_results[m]['CV_F1_Mean'] or 0.0 for m in model_names]
cv_f1_stds  = [cv_results[m]['CV_F1_Std']  or 0.0 for m in model_names]

bilstm_epochs     = list(range(1, len(history.history['accuracy']) + 1))
bilstm_train_acc  = history.history['accuracy']
bilstm_val_acc    = history.history['val_accuracy']
bilstm_train_loss = history.history['loss']
bilstm_val_loss   = history.history['val_loss']


def fig1():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(model_names))
    n = len(graph_metrics)
    w = 0.13
    for i, (metric, vals) in enumerate(graph_metrics.items()):
        offset = (i - n/2 + 0.5) * w
        ax.bar(x+offset, vals, w, label=metric,
               color=GRAYS[i], hatch=HATCHES[i], edgecolor='black', linewidth=0.6)
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Fig. 1: Performance Comparison of All Models')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig1_all_metrics.png')
    plt.close()
    gc.collect()


def fig2():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    f1  = graph_metrics['F1-Score']
    idx = np.argsort(f1)
    sm  = [model_names[i] for i in idx]
    sv  = [f1[i] for i in idx]
    for i, (m, v) in enumerate(zip(sm, sv)):
        ax.barh(m, v, color=GRAYS[i%len(GRAYS)], hatch=HATCHES[i%len(HATCHES)],
                edgecolor='black', linewidth=0.6, height=0.55)
        ax.text(v+0.001, i, f'{v:.4f}', va='center', ha='left', fontsize=9)
    ax.set_xlim(max(0, min(sv)-0.02), 1.01)
    ax.set_xlabel('Weighted F1-Score')
    ax.set_title('Fig. 2: Model Ranking by F1-Score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig2_f1_leaderboard.png')
    plt.close()
    gc.collect()


def fig3():
    n     = len(cm_store)
    ncols = 3
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, nrows*3.2))
    fig.suptitle('Fig. 3: Confusion Matrices', y=1.01, fontweight='bold')
    cmap      = plt.cm.Greys
    labels    = ['Fake', 'Real']
    axes_flat = list(axes.flat)
    for idx_c, (name, cm) in enumerate(cm_store.items()):
        ax     = axes_flat[idx_c]
        total  = cm.sum()
        thresh = cm.max() / 2.0
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]}\n({cm[i,j]/total*100:.1f}%)',
                        ha='center', va='center', fontsize=9,
                        color='white' if cm[i,j] > thresh else 'black')
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(name, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig3_confusion_matrices.png')
    plt.close()
    gc.collect()


def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
    ax1.plot(bilstm_epochs, bilstm_train_acc,  '-o',  ms=4, label='Train',      color='black')
    ax1.plot(bilstm_epochs, bilstm_val_acc,    '--s', ms=4, label='Validation', color='0.45')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Accuracy')
    ax1.set_ylim(max(0, min(bilstm_train_acc)-0.05), 1.01)
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.plot(bilstm_epochs, bilstm_train_loss, '-o',  ms=4, label='Train',      color='black')
    ax2.plot(bilstm_epochs, bilstm_val_loss,   '--s', ms=4, label='Validation', color='0.45')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('(b) Loss')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig.suptitle('Fig. 4: BiLSTM Training History', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig4_bilstm_training.png')
    plt.close()
    gc.collect()


def fig5():
    valid = [(model_names[i], graph_metrics['ROC-AUC'][i])
             for i in range(len(model_names)) if graph_metrics['ROC-AUC'][i] > 0]
    if not valid:
        return
    valid.sort(key=lambda x: x[1])
    sm, sv = zip(*valid)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(range(len(sm)), sv, '--', color='0.55', lw=1, zorder=3)
    for i, (m, v) in enumerate(zip(sm, sv)):
        ax.plot(i, v, marker=MARKERS[i%len(MARKERS)], color='black', ms=8, zorder=5)
        ax.text(i, v+0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(sm)))
    ax.set_xticklabels(sm, rotation=15, ha='right')
    ax.set_ylim(max(0, min(sv)-0.02), 1.01)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Fig. 5: ROC-AUC Score Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig5_roc_auc.png')
    plt.close()
    gc.collect()


def fig6():
    keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N    = len(keys)
    ang  = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    ang  = ang + ang[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for i, model in enumerate(model_names):
        vals  = [graph_metrics[k][i] for k in keys] + [graph_metrics[keys[0]][i]]
        shade = f'0.{max(10, min(85, 10+i*13))}'
        ax.plot(ang, vals, linestyle=LINESTYLES[i%len(LINESTYLES)],
                marker=MARKERS[i%len(MARKERS)], ms=4,
                color=shade, label=model, lw=1.4)
        ax.fill(ang, vals, alpha=0.05, color='0.3')
    ax.set_thetagrids(np.degrees(ang[:-1]), keys, fontsize=9)
    all_v = [v for k in keys for v in graph_metrics[k]]
    ax.set_ylim(max(0.80, min(all_v)-0.02), 1.0)
    ax.set_title('Fig. 6: Multi-Metric Radar Chart', pad=15, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.15), fontsize=8)
    ax.grid(True, linestyle='--', lw=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig6_radar.png')
    plt.close()
    gc.collect()


def fig7():
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    pv = graph_metrics['Precision']
    rv = graph_metrics['Recall']
    for i, m in enumerate(model_names):
        ax.scatter(rv[i], pv[i], marker=MARKERS[i%len(MARKERS)], s=90, color='black', zorder=5)
        ax.annotate(m, (rv[i], pv[i]), textcoords="offset points", xytext=(6, 3), fontsize=8)
    lo = max(0.85, min(min(pv), min(rv))-0.02)
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(lo, 1.01)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Fig. 7: Precision vs. Recall per Model')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig7_precision_recall.png')
    plt.close()
    gc.collect()


def fig8():
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    rows = []
    for m in model_names:
        auc_s = f"{full_metrics[m]['AUC']:.4f}" if full_metrics[m]['AUC'] else '-'
        rows.append([m,
            f"{full_metrics[m]['Accuracy']:.4f}",
            f"{full_metrics[m]['Precision']:.4f}",
            f"{full_metrics[m]['Recall']:.4f}",
            f"{full_metrics[m]['F1']:.4f}",
            auc_s])
    rows.sort(key=lambda x: x[4], reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.65)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor('0.15')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)+1):
        for j in range(len(cols)):
            tbl[i,j].set_facecolor('0.92' if i%2==0 else 'white')
            if i == 1:
                tbl[i,j].set_text_props(fontweight='bold')
    ax.set_title('Table I: Model Performance Metrics', pad=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('saved_graphs/fig8_results_table.png')
    plt.close()
    gc.collect()


def fig9():
    keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    fig.suptitle('Fig. 9: Individual Model Breakdown', fontweight='bold')
    for ax, model in zip(axes.flat, model_names):
        idx_m = model_names.index(model)
        vals  = [graph_metrics[k][idx_m] for k in keys]
        bars  = ax.bar(keys, vals,
                       color=[GRAYS[i] for i in range(len(keys))],
                       hatch=[HATCHES[i] for i in range(len(keys))],
                       edgecolor='black', lw=0.6, width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(max(0.80, min(vals)-0.05), 1.05)
        ax.set_title(model, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels(keys, rotation=10, ha='right', fontsize=8)
    for ax in list(axes.flat)[len(model_names):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig9_per_model_bars.png')
    plt.close()
    gc.collect()


def fig10():
    av  = graph_metrics['Accuracy']
    idx = np.argsort(av)
    sm  = [model_names[i] for i in idx]
    sv  = [av[i] for i in idx]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(range(len(sm)), sv, '-', lw=1.5, color='black', zorder=3)
    for i, (m, v) in enumerate(zip(sm, sv)):
        ax.scatter(i, v, marker=MARKERS[i%len(MARKERS)], s=75, color='black', zorder=5)
        ax.text(i, v+0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(sm)))
    ax.set_xticklabels(sm, rotation=15, ha='right')
    ax.set_ylim(max(0.85, min(sv)-0.02), 1.02)
    ax.set_ylabel('Accuracy')
    ax.set_title('Fig. 10: Accuracy - Ranked Progression')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig10_accuracy_line.png')
    plt.close()
    gc.collect()


def fig11():
    valid_names = [m for m in model_names if cv_results[m]['CV_F1_Mean']]
    means = [cv_results[m]['CV_F1_Mean'] for m in valid_names]
    stds  = [cv_results[m]['CV_F1_Std']  for m in valid_names]
    x     = np.arange(len(valid_names))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x, means,
           color=[GRAYS[i%len(GRAYS)] for i in range(len(valid_names))],
           hatch=[HATCHES[i%len(HATCHES)] for i in range(len(valid_names))],
           edgecolor='black', lw=0.6, width=0.55,
           yerr=stds,
           error_kw=dict(elinewidth=1.2, capsize=5, capthick=1.2, ecolor='black'))
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(x[i], mean+std+0.003, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=15, ha='right')
    ax.set_ylim(max(0.82, min(means)-0.05), 1.03)
    ax.set_ylabel('Weighted F1-Score')
    ax.set_xlabel('Model')
    ax.set_title('Fig. 11: 5-Fold Cross-Validation F1-Score (Mean +/- Std)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig11_cv_errorbars.png')
    plt.close()
    gc.collect()


def fig12():
    valid_names = [m for m in model_names if cv_results[m]['CV_F1_Mean']]
    cv_means    = [cv_results[m]['CV_F1_Mean'] for m in valid_names]
    cv_stds     = [cv_results[m]['CV_F1_Std']  for m in valid_names]
    te_f1       = [full_metrics[m]['F1']        for m in valid_names]
    x = np.arange(len(valid_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x-w/2, cv_means, w, label='5-Fold CV F1',
           color='0.20', hatch='///', edgecolor='black', lw=0.6,
           yerr=cv_stds, error_kw=dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor='black'))
    ax.bar(x+w/2, te_f1, w, label='Test F1',
           color='0.65', hatch='...', edgecolor='black', lw=0.6)
    for i in range(len(valid_names)):
        ax.text(x[i]-w/2, cv_means[i]+cv_stds[i]+0.003,
                f'{cv_means[i]:.3f}', ha='center', va='bottom', fontsize=7.5)
        ax.text(x[i]+w/2, te_f1[i]+0.003,
                f'{te_f1[i]:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=15, ha='right')
    ax.set_ylim(max(0.82, min(cv_means+te_f1)-0.05), 1.04)
    ax.set_ylabel('Weighted F1-Score')
    ax.set_title('Fig. 12: 5-Fold CV vs. Hold-out Test F1-Score')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('saved_graphs/fig12_cv_vs_test.png')
    plt.close()
    gc.collect()


def fig13():
    cols = ['Model', 'CV Acc (Mean+/-Std)', 'CV F1 (Mean+/-Std)', 'Test F1']
    rows = []
    for m in model_names:
        r     = cv_results[m]
        acc_s = f"{r['CV_Acc_Mean']:.4f}+/-{r['CV_Acc_Std']:.4f}" if r['CV_Acc_Mean'] else '-'
        f1_s  = f"{r['CV_F1_Mean']:.4f}+/-{r['CV_F1_Std']:.4f}"   if r['CV_F1_Mean']  else 'val-split'
        rows.append([m, acc_s, f1_s, f"{full_metrics[m]['F1']:.4f}"])
    rows.sort(key=lambda x: x[2], reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.65)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor('0.15')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)+1):
        for j in range(len(cols)):
            tbl[i,j].set_facecolor('0.92' if i%2==0 else 'white')
            if i == 1:
                tbl[i,j].set_text_props(fontweight='bold')
    ax.set_title('Table II: Cross-Validation vs. Test Results', pad=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('saved_graphs/fig13_cv_table.png')
    plt.close()
    gc.collect()


fig1();  fig2();  fig3();  fig4();  fig5()
fig6();  fig7();  fig8();  fig9();  fig10()
fig11(); fig12(); fig13()

print("\nAll figures saved to saved_graphs/")
print("All models saved to saved_models/")
print(f"Best model: {best} | F1={metrics_df.loc[best,'F1']:.4f} | Acc={metrics_df.loc[best,'Accuracy']:.4f}")