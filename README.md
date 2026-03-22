# Fake News Detection Using Machine Learning and Deep Learning

A comparative study of classical machine learning and deep learning approaches for automated fake news detection across multiple benchmark datasets. This repository contains the complete implementation used in the research paper including data preprocessing, feature extraction, model training, 5-fold cross-validation, and publication-ready figure generation.

---

## Abstract

This study evaluates six classification models — Logistic Regression, Linear SVM, Naive Bayes, Gradient Boosting, a Soft Voting Ensemble, and a Bidirectional LSTM — on a combined fake news corpus sourced from four publicly available datasets. Models are compared across accuracy, precision, recall, F1-score, and ROC-AUC. 5-fold stratified cross-validation is applied to all classical models to ensure robustness. TF-IDF with unigram and bigram features is used for classical models; the BiLSTM uses a learned embedding layer.

---

## Models

| Model | Category |
|---|---|
| Logistic Regression | Classical ML |
| Linear SVM | Classical ML |
| Naive Bayes | Classical ML |
| Gradient Boosting | Classical ML |
| Voting Ensemble (LR + SVM + NB) | Classical ML |
| Bidirectional LSTM | Deep Learning |

---

## Datasets

| Dataset | Source | Label |
|---|---|---|
| Fake or Real News | [Kaggle — LINK PLACEHOLDER] | FAKE / REAL |
| ISOT Fake News — Fake.csv | [Kaggle — LINK PLACEHOLDER] | 0 |
| ISOT Fake News — True.csv | [Kaggle — LINK PLACEHOLDER] | 1 |
| FakeNewsNet BuzzFeed Real | [GitHub — LINK PLACEHOLDER] | 1 |
| FakeNewsNet BuzzFeed Fake | [GitHub — LINK PLACEHOLDER] | 0 |
| FakeNewsNet PolitiFact Real | [GitHub — LINK PLACEHOLDER] | 1 |
| FakeNewsNet PolitiFact Fake | [GitHub — LINK PLACEHOLDER] | 0 |

A preprocessed combined version of all datasets is available here:
[Google Drive — LINK PLACEHOLDER]

---

## Repository Structure

```
.
├── fake_news_detection.py           
├── dataset_distribution_graphs.py   
├── requirements.txt                 
├── saved_models/                    
│   ├── LogReg.pkl
│   ├── LinearSVM.pkl
│   ├── NaiveBayes.pkl
│   ├── GBM.pkl
│   ├── Ensemble.pkl
│   ├── BiLSTM_model.keras
│   ├── bilstm_tokenizer.json
│   ├── tfidf.pkl
│   ├── svd.pkl
│   ├── normalizer.pkl
│   └── cv_results.csv
└── saved_graphs/                    
    ├── fig1_all_metrics.png
    ├── fig2_f1_leaderboard.png
    ├── fig3_confusion_matrices.png
    ├── fig4_bilstm_training.png
    ├── fig5_roc_auc.png
    ├── fig6_radar.png
    ├── fig7_precision_recall.png
    ├── fig8_results_table.png
    ├── fig9_per_model_bars.png
    ├── fig10_accuracy_line.png
    ├── fig11_cv_errorbars.png
    ├── fig12_cv_vs_test.png
    ├── fig13_cv_table.png
    ├── figD1_class_distribution_bar.png
    ├── figD2_class_balance_pie.png
    ├── figD3_wordcount_distribution.png
    └── figD4_dataset_stats_table.png
```

---

## Requirements

- Python 3.8 — 3.10
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

On Google Colab, only scikit-learn needs upgrading:

```bash
pip install scikit-learn==1.5.2 --upgrade -q
```

---

## Usage

### Google Colab (Recommended)

**1. Set runtime**

`Runtime → Change runtime type → T4 GPU`

**2. Mount Drive and place datasets**

Place raw dataset files at the following path in your Google Drive:

```
MyDrive/
└── Raw Data/
    ├── fake_or_real_news.csv
    ├── FAKE AND TRUE DATASET/
    │   ├── Fake.csv
    │   └── True.csv
    └── FakeNewsNet/
        ├── BuzzFeed_real_news_content.csv
        ├── BuzzFeed_fake_news_content.csv
        ├── PolitiFact_real_news_content.csv
        └── PolitiFact_fake_news_content.csv
```

**3. Run in order**

```python
# Cell 1 — upgrade scikit-learn
!pip install scikit-learn==1.5.2 --upgrade -q

# Cell 2 — dataset distribution figures (optional, run before main pipeline)
# paste contents of dataset_distribution_graphs.py

# Cell 3 — main pipeline
# paste contents of fake_news_detection.py
```

Outputs are saved to `saved_models/` and `saved_graphs/` automatically.

---

## Reload Saved Models

```python
import pickle, json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

logreg      = pickle.load(open('saved_models/LogReg.pkl',     'rb'))
svm         = pickle.load(open('saved_models/LinearSVM.pkl',  'rb'))
nb          = pickle.load(open('saved_models/NaiveBayes.pkl', 'rb'))
gbm         = pickle.load(open('saved_models/GBM.pkl',        'rb'))
ensemble    = pickle.load(open('saved_models/Ensemble.pkl',   'rb'))

tfidf       = pickle.load(open('saved_models/tfidf.pkl',      'rb'))
svd         = pickle.load(open('saved_models/svd.pkl',        'rb'))
normalizer  = pickle.load(open('saved_models/normalizer.pkl', 'rb'))

bilstm      = load_model('saved_models/BiLSTM_model.keras')
with open('saved_models/bilstm_tokenizer.json') as f:
    tok = tokenizer_from_json(json.load(f))
```

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Train / Test Split | 80% / 20% stratified |
| Cross-Validation | 5-fold stratified (classical models) |
| BiLSTM Validation | 15% held-out split |
| TF-IDF Max Features | 50,000 |
| TF-IDF N-gram Range | (1, 2) unigrams + bigrams |
| TF-IDF Sublinear TF | True |
| Logistic Regression C | 5 |
| SVM C | 1.0 |
| Naive Bayes Alpha | 0.1 |
| GBM Estimators | 100 |
| GBM Learning Rate | 0.1 |
| GBM Max Depth | 4 |
| BiLSTM Vocabulary Size | 20,000 |
| BiLSTM Max Sequence Length | 100 |
| BiLSTM Embedding Dimension | 64 |
| BiLSTM LSTM Units | 64 / 32 (two layers) |
| BiLSTM Dropout Rates | 0.3, 0.3, 0.2 |
| BiLSTM Batch Size | 128 |
| BiLSTM Optimizer | Adam |
| BiLSTM Learning Rate | 0.001 |
| BiLSTM LR Decay | ReduceLROnPlateau (factor=0.5, min=1e-5) |
| BiLSTM Early Stopping | Patience=3, monitor=val_accuracy |
| BiLSTM Max Epochs | 10 |

---

## Generated Figures

| File | Description |
|---|---|
| fig1_all_metrics.png | Grouped bar chart — all metrics per model |
| fig2_f1_leaderboard.png | Horizontal bar — F1 ranking |
| fig3_confusion_matrices.png | Confusion matrices for all models |
| fig4_bilstm_training.png | BiLSTM training and validation curves |
| fig5_roc_auc.png | ROC-AUC comparison |
| fig6_radar.png | Radar chart — multi-metric view |
| fig7_precision_recall.png | Precision vs Recall scatter |
| fig8_results_table.png | Summary results table |
| fig9_per_model_bars.png | Per-model metric breakdown |
| fig10_accuracy_line.png | Accuracy ranked progression |
| fig11_cv_errorbars.png | 5-fold CV F1 with error bars |
| fig12_cv_vs_test.png | CV vs hold-out test F1 |
| fig13_cv_table.png | CV results table |
| figD1_class_distribution_bar.png | Dataset class distribution |
| figD2_class_balance_pie.png | Class balance pie chart |
| figD3_wordcount_distribution.png | Word count distribution per class |
| figD4_dataset_stats_table.png | Dataset statistics summary |

All figures are saved at 300 DPI in grayscale, suitable for IEEE, Springer, and Elsevier submission.

---

## Citation

If you use this code or the methodology in your work, please cite:

```
[CITATION PLACEHOLDER — add after paper is published]
```

---

## License

[LICENSE PLACEHOLDER]