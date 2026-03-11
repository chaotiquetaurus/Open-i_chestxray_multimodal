# Beyond Pixels - Multimodal X-Ray Diagnosis

Multimodal approach to chest X-ray diagnosis, combining radiology report text analysis with image data from the [Open-I](https://openi.nlm.nih.gov/) dataset.

## Project Structure

```
Text classification/
  first_phases.ipynb   — Baseline text classification notebook
```

## Text Classification Baseline

`Text classification/first_phases.ipynb` implements a simple baseline for classifying radiology reports into 12 pathology labels (multi-label) using a logistic regression pipeline:

1. **Data loading** — Parses Open-I XML reports (indication + findings) and loads pathology labels via [TorchXRayVision](https://github.com/mlmed/torchxrayvision) (2314 reports, 12 labels including Normal)
2. **Preprocessing** — Anonymization token removal, stopword filtering, and lemmatization with spaCy
3. **Feature extraction** — TF-IDF vectorization on both indication and findings fields (1074 features)
4. **Exploratory analysis** — Label distribution, correlation heatmap, top TF-IDF words per pathology, t-SNE visualization
5. **Binary classification** — Normal vs. Pathology logistic regression with hyperparameter tuning (RandomizedSearchCV), achieving AUC = 0.96
6. **Multi-label classification** — OneVsRest logistic regression with ElasticNet regularization, achieving Macro AUC = 0.91

## How to Run

The notebook runs on **Google Colab** with Google Drive for dataset storage.

1. Upload the `NLMCXR_reports/ecgen-radiology` folder to your Google Drive under `Colab Notebooks/dataset/`
2. Open `Text classification/first_phases.ipynb` in Google Colab
3. Run all cells — dependencies (`torchxrayvision`, `spacy`, `en_core_web_sm`) are installed automatically

## Workflow git
Every user use a different branch while developping his part of the code, than merge it (and delete it) on main.
We still allow the right to main for quick fix, but one should not use it otherwise
