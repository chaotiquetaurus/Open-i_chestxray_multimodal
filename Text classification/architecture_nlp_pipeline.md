# Architecture -- Pipeline NLP Multi-label

**Projet :** Classification de rapports radiologiques (Indiana University Chest X-ray, ~3 955 rapports)

---

## Arborescence du projet

```
Text classification/
├── data/
│   ├── __init__.py           # exports: build_tokenizer, SP, MLMDataset, LabelDataset, pad_collate
│   ├── datasets.py           # MLMDataset, LabelDataset, pad_collate
│   ├── tokenizer.py          # build_tokenizer (BPE HuggingFace)
│   ├── labeliser.py          # XMLLoader, LabelMapper, preprocessing spaCy
│   ├── dataset_reports.csv   # rapports bruts (uid, indication, findings, impression)
│   └── dataset_labeled.csv   # rapports + 21 colonnes binaires de labels
├── models/
│   ├── __init__.py           # exports: Encoder, BERTForMLM, Classifier
│   ├── layers.py             # RoPE, MHA, Block (composants Transformer)
│   ├── encoder.py            # Encoder (stack de Blocks)
│   ├── bert_mlm.py           # BERTForMLM (encoder + tete MLM)
│   └── classifier.py         # Classifier (encoder + [CLS] pooling + tete multi-label)
├── scripts/
│   ├── pretrain.py           # Etape 3a : pre-entrainement MLM sur MIMIC-CXR
│   ├── finetune.py           # Etape 3b : fine-tuning multi-label
│   └── kfold.py              # Etape 3c : validation croisee 5-fold (Lightning)
├── notebooks/
│   ├── first_phases.ipynb    # Etape 1 : TF-IDF + sklearn
│   └── Deep_learning_Nlp_TFIDF.ipynb  # Etape 2 : TF-IDF + MLP PyTorch
├── checkpoints/              # tokenizer.json, bert_pretrained.pt, classifier_*.pt
└── outputs/                  # courbes d'entrainement (PNG)
```

---

## Pipeline de donnees (commun aux 3 etapes)

### Extraction et labellisation (`data/labeliser.py`)

| Composant | Role |
|-----------|------|
| `XMLLoader` | Parse ~3 955 fichiers XML Open-i → DataFrame (`uid`, `indication`, `findings`, `impression`) |
| `LabelMapper` | Mapping deterministe MeSH → **21 colonnes binaires** (dictionnaire Cohen et al. / TorchXRayVision) |
| `get_class_weights()` | Calcul des poids inverses de frequence pour `BCEWithLogitsLoss` |

### Preprocessing textuel (spaCy `en_core_web_sm`)

1. `removex()` -- supprime les placeholders (`xxxx`, `XXXX`, `year`, `years`, `old`)
2. `stopword()` -- supprime stop words et ponctuation
3. `lemmatizer()` -- lemmatisation

### Sortie

- `dataset_reports.csv` -- rapports bruts
- `dataset_labeled.csv` -- rapports + 21 labels binaires

---

## Etape 1 -- TF-IDF + sklearn

> **Notebook :** `notebooks/first_phases.ipynb`
>
> **Objectif :** Baseline classique ML. Vecteurs TF-IDF + classifieurs lineaires scikit-learn.

### 1.1 Vectorisation TF-IDF

Deux `TfidfVectorizer` independants concatenes via `scipy.hstack` :

| Champ | max_features | min_df | stop_words |
|-------|-------------|--------|------------|
| `indication` | 2 500 | 2 | english |
| `impression` | 2 500 | 2 | english |

Features effectives apres filtrage : **~1 432**.

### 1.2 Modeles

**Classification binaire (Normal vs Pathologie) :**

`MaxAbsScaler` → `LogisticRegression(solver='saga')`.
Optimisation : `RandomizedSearchCV` sur `C`, `penalty` (l1 / l2 / elasticnet), `l1_ratio`.

**Classification multi-label (21 pathologies) :**

`OneVsRestClassifier` wrappant la meme pipeline. Un classifieur binaire par pathologie.

### 1.3 Entrainement

- `StratifiedKFold` avec `scoring='roc_auc'`
- `RandomizedSearchCV` -- grille separee l1/l2 (sans `l1_ratio`) et elasticnet (avec `l1_ratio`)

> **Note :** `l1_ratio` ne doit apparaitre dans la grille qu'avec la penalite `elasticnet`, sinon tous les fits echouent.

### 1.4 Resultats

| Metrique | Valeur |
|----------|--------|
| AUC binaire | **0.96** |
| Accuracy binaire | **0.90** |
| AUC multi-label | Variable par pathologie |

---

## Etape 2 -- TF-IDF + Deep Learning (MLP PyTorch)

> **Notebook :** `notebooks/Deep_learning_Nlp_TFIDF.ipynb`
>
> **Objectif :** Remplacer le classifieur sklearn par un MLP PyTorch, en gardant la meme representation TF-IDF.

### 2.1 Vectorisation

Identique a l'etape 1 (TF-IDF indication + impression, ~1 432 features).

### 2.2 DataModule (`NlpDataModule`)

Conversion sparse → tenseurs PyTorch et creation des DataLoaders :

- Split : 80/10/10 via `train_test_split` (seed=42)
- `_to_tensor()` : `sparse.toarray()` → `torch.float32`
- Tailles : Train = 2 847 | Val = 317 | Test = 791

### 2.3 Modele MLP

**Baseline :** `Linear(1432, 256)` → `ReLU` → `Linear(256, 21)` (sans dropout, 10 epochs)

**Modele optimise :** `Linear(1432, hidden)` → `ReLU` → `Dropout(0.3)` → `Linear(hidden, 21)`

- Loss : `BCEWithLogitsLoss` avec `pos_weight` (poids inverses de frequence, clippes a `pw_clip`)
- Optimizer : `Adam`

### 2.4 Entrainement

**Early stopping :**

```
train_with_early_stopping(model, loss_fn, optimizer, patience=5, max_epochs=100)
```

Sauvegarde du meilleur modele (`best.pt`) selon la validation loss.

**Grid search :** 45 combinaisons (3 `lr` x 3 `hidden` x 5 `pw_clip`), chacune entrainee avec early stopping (max 60 epochs). Le meilleur modele est reentraine avec patience=10, max_epochs=100.

| Hyperparametre | Espace de recherche | Meilleure config |
|----------------|---------------------|------------------|
| `hidden` | {64, 128, 256} | **64** |
| `lr` | {1e-4, 5e-4, 1e-3} | **1e-3** |
| `pw_clip` | {1, 2, 5, 10, 20} | **20** |

### 2.5 Resultats

| Metrique | Baseline | Optimise |
|----------|----------|----------|
| F1 macro | 0.4238 | 0.4133 |
| F1 micro | 0.5859 | 0.5874 |
| AUC macro | 0.8601 | **0.8643** |
| Hamming loss | 0.0692 | 0.0694 |
| Seuil optimal | 0.55 | 0.55 |

Le MLP optimise ameliore legerement l'AUC macro (+0.004) mais le F1 macro baisse. Le ranking est bon (AUC 0.86), le bottleneck reste le desequilibre de classes sur les pathologies rares.

---

## Etape 3 -- Custom BERT (pre-entrainement + fine-tuning)

> **Scripts :** `scripts/pretrain.py`, `scripts/finetune.py`, `scripts/kfold.py`
> **Modules :** `models/`, `data/`
>
> **Objectif :** Remplacer TF-IDF par un encodeur Transformer entraine from scratch sur un corpus medical, puis fine-tune sur le dataset IU X-Ray.

### 3.1 Tokenizer BPE (`data/tokenizer.py`)

Tokenizer BPE entraine from scratch via HuggingFace `tokenizers` :

```python
build_tokenizer(texts, vocab_size=8192, max_len=256)
```

- Pre-tokenizer : `ByteLevel`
- Tokens speciaux : `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- Post-processing : insertion automatique `[CLS]` / `[SEP]` via `TemplateProcessing`
- Sauvegarde : `checkpoints/tokenizer.json`

### 3.2 Architecture de l'encodeur (`models/`)

Encodeur Transformer from scratch, inspire BERT avec des composants modernes :

| Module | Fichier | Description |
|--------|---------|-------------|
| `RoPE` | `layers.py` | Rotary Position Embedding (encodage positionnel relatif, applique aux Q et K) |
| `MHA` | `layers.py` | Multi-Head Attention (8 tetes, `scaled_dot_product_attention`, FlashAttention-compatible) |
| `Block` | `layers.py` | Pre-norm Transformer block : `RMSNorm → MHA → residuel → RMSNorm → FFN(SiLU) → residuel` |
| `Encoder` | `encoder.py` | Embedding + stack de 6 `Block` + `RMSNorm` finale |
| `BERTForMLM` | `bert_mlm.py` | Encoder + tete MLM : `Linear(d,d) → GELU → RMSNorm → Linear(d,V)` |
| `Classifier` | `classifier.py` | Encoder + `[CLS]` pooling (position 0) + `Dropout(0.1)` + `Linear(d, n_labels)` |

**Configuration :**

| Parametre | Valeur |
|-----------|--------|
| `d` (dimension) | 256 |
| `h` (tetes) | 8 |
| `N` (couches) | 6 |
| `d_ff` (FFN) | 512 |
| `V` (vocabulaire) | 8 192 |
| Params total | ~4.9M |

**Choix architecturaux :**
- **RoPE** au lieu de positional embeddings appris → meilleure generalisation aux longueurs variables, pas de parametres supplementaires
- **RMSNorm + SiLU** (style LLaMA) au lieu de LayerNorm + GELU → meilleure stabilite d'entrainement
- **Pre-norm** (normalisation avant l'attention) au lieu de post-norm

### 3.3 Datasets (`data/datasets.py`)

**`MLMDataset`** -- pour le pre-entrainement :
- Masquage aleatoire de 15% des tokens (hors tokens speciaux)
- Remplacement : 80% `[MASK]`, 10% token aleatoire, 10% inchange
- Labels non masques = `-100` (ignores par `cross_entropy`)

**`LabelDataset`** -- pour le fine-tuning :
- Encode les textes via le tokenizer pre-entraine
- Labels : tenseur `float` multi-label (vecteur binaire)

**`pad_collate()`** -- collate function avec padding dynamique au max du batch

### 3.4 Etape 3a -- Pre-entrainement MLM (`scripts/pretrain.py`)

Pre-entrainement auto-supervise sur MIMIC-CXR (~227k rapports radiologiques) via Masked Language Modeling.

```
python scripts/pretrain.py
```

| Parametre | Valeur |
|-----------|--------|
| Dataset | MIMIC-CXR (HuggingFace, ~227k rapports) |
| Epochs | 20 |
| Learning rate | 3e-4 |
| Batch size | 32 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Gradient clipping | max_norm=1.0 |

**Sortie :**
- `checkpoints/tokenizer.json`
- `checkpoints/bert_pretrained.pt`
- `outputs/mlm_curves.png`

### 3.5 Etape 3b -- Fine-tuning multi-label (`scripts/finetune.py`)

Fine-tuning du `Classifier` sur IU X-Ray avec les poids pre-entraines.

```
python scripts/finetune.py [5|14|21]
```

**Trois modes de labels :**

| Mode | Labels | Description |
|------|--------|-------------|
| 5 | Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion | CheXpert (benchmark SOTA) |
| 14 | 14 pathologies NIH | NIH ChestX-ray14 |
| 21 | Tous les labels IU X-Ray | Jeu complet incluant Normal |

| Parametre | Valeur |
|-----------|--------|
| Texte d'entree | `findings` |
| Epochs | 30 (max) |
| Learning rate | 2e-4 |
| Batch size | 32 |
| Early stopping | patience=5 |
| pos_weight | neg/pos clampe a 5 |
| Freeze encoder | Configurable (`FREEZE_ENCODER`) |
| Split | 80/20 (train/val) |

**Sortie :**
- `checkpoints/classifier_{mode}.pt`
- `outputs/ft_{mode}.png`

### 3.6 Etape 3c -- K-fold Cross-Validation (`scripts/kfold.py`)

Validation croisee 5-fold avec PyTorch Lightning pour une evaluation robuste.

```
python scripts/kfold.py [5|14|21]
```

**`LitClassifier`** (module Lightning) :
- `training_step` / `validation_step` : `BCEWithLogitsLoss` + pos_weight
- `on_validation_epoch_end` : calcul AUC macro sur le fold
- `configure_optimizers` : `AdamW` + `CosineAnnealingLR`

| Parametre | Valeur |
|-----------|--------|
| Texte d'entree | `indication` + `findings` (concatenes) |
| K folds | 5 |
| Early stopping par fold | patience=5 sur `val_loss` |
| pos_weight | neg/pos clampe a 20 (cas pathologiques uniquement) |
| Encodeur | Copie fraiche par fold (`copy.deepcopy`) |

**Sortie :**
- `outputs/kfold_{mode}.png`

---

## Evaluation (commune aux 3 etapes)

Recherche de seuil optimal + metriques standardisees :

| Fonctionnalite | Description |
|----------------|-------------|
| Seuil optimal | Recherche exhaustive sur [0.1, 0.55] par pas de 0.05 (maximise F1 macro) |
| F1 macro / micro / samples | F1-score sous trois moyennes |
| AUC macro / micro | ROC-AUC (classes avec support > 1 uniquement) |
| AUC par label | AUC individuel par pathologie (barplot horizontal) |
| Hamming loss | Erreur label par label |
| Plots | Loss curves, metriques globales (barplot), AUC par label |

---

## Recapitulatif des 3 etapes

| | Etape 1 | Etape 2 | Etape 3 |
|---|---------|---------|---------|
| **Representation** | TF-IDF (1 432 features) | TF-IDF (1 432 features) | BPE tokenizer (8 192 tokens) |
| **Modele** | LogisticRegression / OneVsRest | MLP PyTorch (Linear-ReLU-Dropout-Linear) | Transformer 6 couches (~4.9M params) |
| **Framework** | scikit-learn | PyTorch | PyTorch + Lightning |
| **Texte utilise** | indication + impression | indication + impression | findings (finetune) / indication + findings (kfold) |
| **Pre-entrainement** | -- | -- | MLM sur MIMIC-CXR (~227k rapports) |
| **Validation** | StratifiedKFold + RandomizedSearchCV | Grid search + early stopping | 5-fold CV (Lightning) |
| **AUC macro** | Variable | 0.86 | A evaluer |

---

## Infrastructure GPU (Telecom)

Le passage de Google Colab aux GPU de Telecom implique :

- **Paradigme batch job** : scripts `.py` soumis a un ordonnanceur (type SLURM) au lieu de notebooks interactifs
- **Portabilite GPU/CPU** : `torch.device("cuda" if torch.cuda.is_available() else "cpu")` dans tous les scripts
- **Chemins parametres** : variables `CKPT_DIR`, `OUT_DIR`, `DATA_DIR` en haut de chaque script
- **Bloc 1** (sklearn) : CPU uniquement. **Bloc 2** (MLP) : GPU optionnel. **Bloc 3** (BERT) : GPU recommande
