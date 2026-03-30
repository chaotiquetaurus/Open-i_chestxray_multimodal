# Architecture Logicielle — Pipeline NLP Multi-label

**Projet Open-i · Chest X-ray Report Classification**
Dataset : Indiana University Chest X-ray Collection (~3 955 rapports)

---

## 1. Vue d'ensemble

L'architecture est organisée en **quatre couches** indépendantes. Le pipeline suit deux blocs expérimentaux : TF-IDF + Régression Logistique (Bloc 1) puis TF-IDF + MLP PyTorch (Bloc 2).

| Couche | Responsabilité | Classes principales |
|--------|---------------|---------------------|
| Data Layer | Chargement, labellisation, préparation | `XMLLoader`, `LabelMapper`, `TFIDFVectorizer`, `NlpDataModule` |
| Model Layer | Définition des modèles | `SklearnMultiLabel`, `nn.Sequential` (MLP) |
| Training Layer | Entraînement et optimisation | `SklearnTrainer`, `train_with_early_stopping()` |
| Evaluation | Métriques et visualisation | `evaluate_model()` |

### Diagramme d'architecture

```mermaid
graph LR
    subgraph DATA["Data Layer"]
        direction TB
        XMLLoader["XMLLoader\n---\nparse_xml()\nbuild_dataframe()"]
        LabelMapper["LabelMapper\n---\nmesh_to_labels()\nget_class_weights()"]
        Preproc["Preprocessing\n---\nremovex()\nstopword()\nlemmatizer()\nspaCy en_core_web_sm"]
        TFIDF["TFIDFVectorizer ×2\n---\nindication: 2500\nimpression: 2500\nmin_df=2\nscipy.hstack"]
        DataModule["NlpDataModule\n---\nsparse→tensor\ntrain/val/test split\nDataLoader"]
    end

    subgraph MODELS["Model Layer"]
        direction TB
        SKModel["SklearnMultiLabel\n---\nOneVsRestClassifier\nLogisticRegression\nMaxAbsScaler"]
        MLPModel["nn.Sequential (MLP)\n---\nLinear→ReLU→Dropout→Linear\nBCEWithLogitsLoss + pos_weight"]
    end

    subgraph TRAIN["Training Layer"]
        direction TB
        SKTrainer["SklearnTrainer\n---\ncross_validate()\nRandomizedSearchCV"]
        TorchTrainer["train_with_early_stopping()\n---\nearly stopping (patience)\ngrid search (lr, hidden, pw_clip)"]
    end

    subgraph EVAL["Evaluation"]
        direction TB
        Evaluator["evaluate_model()\n---\nf1_macro/micro/samples\nauc_roc macro/micro/per_label\nhamming_loss\nthreshold optimization\nplots"]
    end

    XMLLoader -->|"DataFrame"| LabelMapper
    LabelMapper -->|"df + 21 labels"| Preproc
    Preproc -->|"texte nettoyé"| TFIDF
    TFIDF -->|"sparse matrix"| SKModel
    TFIDF -->|"sparse matrix"| DataModule
    DataModule -->|"tenseurs + DataLoaders"| MLPModel

    SKModel -->|"utilisé par"| SKTrainer
    MLPModel -->|"utilisé par"| TorchTrainer

    SKTrainer -->|"résultats"| Evaluator
    TorchTrainer -->|"résultats"| Evaluator
```

---

## 2. Data Layer

### 2.1 XMLLoader

Parse les ~3 955 fichiers XML Open-i. Produit un DataFrame (`uid`, `indication`, `findings`, `impression`).

- **`parse_xml()`** — extrait INDICATION, FINDINGS et IMPRESSION.
- **`build_dataframe()`** — construit le DataFrame consolidé avec préfixe UID (`"CXR" + uid`) pour compatibilité TorchXRayVision.

### 2.2 LabelMapper

Mapping déterministe MeSH → pathologies (dictionnaire Cohen et al. / TorchXRayVision). Produit **21 colonnes binaires**.

- **`mesh_to_labels()`** — convertit les termes MeSH en vecteurs multi-label binaires.
- **`get_class_weights()`** — calcule les poids inverses de fréquence pour `BCEWithLogitsLoss`.

### 2.3 Preprocessing (spaCy)

Pipeline de nettoyage textuel appliqué à `indication` et `impression` :

1. **`removex()`** — supprime les placeholders (`xxxx`, `XXXX`, `year`, `years`, `old`).
2. **`stopword()`** — supprime stop words et ponctuation via spaCy (`en_core_web_sm`).
3. **`lemmatizer()`** — lemmatisation via spaCy.

### 2.4 Vectorisation TF-IDF

Deux `TfidfVectorizer` indépendants, concaténés via `scipy.hstack` :

| Champ | max_features | min_df | stop_words |
|-------|-------------|--------|------------|
| `indication` | 2500 | 2 | english |
| `impression` | 2500 | 2 | english |

Résultat effectif : **1 432 features** (après filtrage `min_df=2`).

### 2.5 NlpDataModule (Bloc 2)

Gère la conversion sparse → tenseurs PyTorch et la création des DataLoaders :

```python
class NlpDataModule():
    def __init__(self, X_sparse, y, batch_size=64, test_size=0.2, val_size=0.1, random_state=42)
```

- Split : 80/10/10 via `train_test_split` (seed=42).
- `_to_tensor()` : `sparse.toarray()` → `torch.float32`.
- `get_dataloader(split)` : retourne un `DataLoader` avec `pin_memory` si GPU.
- Tailles : Train=2 847 | Val=317 | Test=791.

---

## 3. Model Layer

### 3.1 SklearnMultiLabel (Bloc 1)

**Classification binaire (Normal vs Pathologie) :**
Pipeline `MaxAbsScaler` → `LogisticRegression(solver='saga')`. Optimisation via `RandomizedSearchCV` sur `C`, `penalty` (l1/l2/elasticnet), `l1_ratio`.

**Multi-label (21 pathologies) :**
`OneVsRestClassifier` wrappant la même pipeline. Un classifieur binaire par pathologie avec son propre seuil optimal.

### 3.2 MLP (Bloc 2 — `nn.Sequential`)

**Baseline :** `Linear(1432, 256) → ReLU → Linear(256, 21)` — sans dropout, 10 epochs.

**Modèle optimisé (grid search) :** `Linear(1432, hidden) → ReLU → Dropout(0.3) → Linear(hidden, 21)`

| Hyperparamètre | Espace de recherche | Meilleure config |
|----------------|---------------------|------------------|
| `hidden` | {64, 128, 256} | **64** |
| `lr` | {1e-4, 5e-4, 1e-3} | **1e-3** |
| `pw_clip` | {1, 2, 5, 10, 20} | **20** |

- Loss : `BCEWithLogitsLoss` avec `pos_weight` (poids inverses de fréquence, clippés à `pw_clip`).
- Optimizer : `Adam`.

---

## 4. Training Layer

### 4.1 SklearnTrainer (Bloc 1)

- **`cross_validate()`** — StratifiedKFold, `scoring='roc_auc'`.
- **`RandomizedSearchCV`** — grille séparée pour l1/l2 (sans `l1_ratio`) et elasticnet (avec `l1_ratio`).

> `l1_ratio` ne doit apparaître dans la grille qu'avec la pénalité `elasticnet`. Le mélanger avec `l1`/`l2` fait échouer tous les fits.

### 4.2 train_with_early_stopping (Bloc 2)

```python
def train_with_early_stopping(model, loss_fn, optimizer, patience=5, max_epochs=100):
```

- Sauvegarde le meilleur modèle (`best.pt`) selon la validation loss.
- Early stopping : patience=5 (grid search), patience=10 (retrain final).
- Retourne `train_losses`, `val_losses` pour les plots.

**Grid search :** 45 combinaisons (3 lr × 3 hidden × 5 pw_clip), chacune entraînée avec early stopping (max 60 epochs). Évaluation sur le val set (F1 macro + AUC macro). Le meilleur modèle est réentraîné avec patience=10 et max_epochs=100.

---

## 5. Couche d'évaluation

Une seule fonction `evaluate_model()` partagée par les blocs :

| Fonctionnalité | Description |
|----------------|------------|
| Seuil optimal | Recherche exhaustive sur [0.1, 0.55] par pas de 0.05 |
| F1 macro/micro/samples | F1-score sous trois moyennes |
| AUC macro/micro | ROC-AUC (classes avec support > 1 uniquement) |
| AUC par label | AUC individuel par pathologie avec barplot |
| Hamming loss | Erreur label par label |
| Plots | Loss curves, métriques globales (barplot), AUC par label (barplot horizontal) |

---

## 6. Résultats

### Bloc 1 — Régression Logistique

- **Binaire (Normal vs Pathologie)** : AUC = 0.96, Accuracy = 0.90.
- **Multi-label** : AUC variable par pathologie, optimisé via `RandomizedSearchCV`.

### Bloc 2 — MLP

**Baseline (sans dropout, sans pos_weight clipping) :**

| Métrique | Valeur |
|----------|--------|
| F1 macro | 0.4238 |
| F1 micro | 0.5859 |
| AUC macro | 0.8601 |
| Hamming loss | 0.0692 |
| Seuil optimal | 0.55 |

**Modèle optimisé (lr=1e-3, hidden=64, pw_clip=20, dropout=0.3) :**

| Métrique | Valeur |
|----------|--------|
| F1 macro | 0.4133 |
| F1 micro | 0.5874 |
| AUC macro | 0.8643 |
| Hamming loss | 0.0694 |
| Seuil optimal | 0.55 |

Le modèle optimisé améliore légèrement l'AUC macro (+0.004) mais le F1 macro baisse. L'AUC macro de 0.86 confirme que le ranking est bon ; le bottleneck reste le déséquilibre de classes pour les pathologies rares.

---

## 7. Décisions de conception

**Vectorisation indication + impression** — Les deux champs textuels sont vectorisés séparément puis concaténés. `findings` n'est pas utilisé dans le Bloc 2.

**pos_weight avec clipping** — Les poids inverses de fréquence corrigent le déséquilibre mais sont clippés (`pw_clip`) pour éviter une instabilité numérique sur les classes très rares.

**Seuil global optimisé** — Un seuil unique est recherché sur la plage [0.1, 0.55] plutôt qu'un seuil par label, par simplicité.

**Séparation des paradigmes** — Deux approches distinctes : scikit-learn (`fit`/`predict`) pour le Bloc 1 et PyTorch (boucle manuelle) pour le Bloc 2.

**Extensibilité** — Un `ImageDataset` (Phase 3) et un `FusionClassifier` (Phase 4+) pourront s'ajouter sans modifier les couches actuelles.

---

## 8. Déploiement GPU — Infrastructure Télécom

Le passage de Google Colab aux GPU de Télécom implique plusieurs adaptations d'infrastructure, sans changement de code modèle.

### Environnement cible

L'infrastructure GPU de Télécom met à disposition des nœuds de calcul équipés de GPU NVIDIA accessibles via un ordonnanceur de jobs (type SLURM). Contrairement à Colab où l'exécution est interactive dans un notebook, le paradigme devient celui du **batch job** : on soumet un script, il est mis en file d'attente, puis exécuté quand les ressources sont disponibles.

### Transition Colab → Télécom

Le code développé en notebooks Colab doit être converti en **scripts Python autonomes** (`.py`) exécutables en ligne de commande. Les notebooks restent utilisables pour l'exploration et le prototypage, mais l'entraînement final se fait via des scripts soumis à l'ordonnanceur.

Les données Open-i, actuellement sur Google Drive, seront transférées vers le **stockage partagé** du cluster (accessible depuis tous les nœuds de calcul). Les chemins de fichiers seront paramétrés via des arguments CLI ou un fichier de configuration pour éviter les chemins en dur.

### Gestion des ressources GPU

Le Bloc 1 (TF-IDF + scikit-learn) n'a pas besoin de GPU et peut tourner sur CPU. Le Bloc 2 (MLP) bénéficie marginalement d'un GPU vu la taille du modèle. Le code utilise `torch.device("cuda" if torch.cuda.is_available() else "cpu")` pour la portabilité automatique entre les deux environnements. Les checkpoints du modèle (`best.pt` via early stopping) sont sauvegardés pour être récupérés après le job.

### Organisation pratique

Chaque expérience (combinaison d'hyperparamètres) correspond à un job soumis séparément. Les logs et métriques sont écrits dans des fichiers structurés pour permettre la comparaison post-hoc via `evaluate_model()`. Le grid search (45 combinaisons) peut être parallélisé sur plusieurs GPU si disponibles.

---

## 9. Inventaire complet des classes et fonctions

| Élément | Couche | Rôle |
|---------|--------|------|
| `XMLLoader` | Data | Parse les fichiers XML, construit le DataFrame brut |
| `LabelMapper` | Data | Mapping MeSH → 21 pathologies, calcul des poids de classes |
| `removex()` / `stopword()` / `lemmatizer()` | Data | Preprocessing textuel via spaCy |
| `TfidfVectorizer` ×2 | Data | Vectorisation indication + impression, 2500 features chacun |
| `NlpDataModule` | Data | Sparse → tenseurs, splits 80/10/10, DataLoaders PyTorch |
| `SklearnMultiLabel` | Model | OneVsRestClassifier + Pipeline LogisticRegression |
| `nn.Sequential` (MLP) | Model | Linear→ReLU→Dropout→Linear, BCEWithLogitsLoss + pos_weight |
| `SklearnTrainer` | Training | Cross-validation, RandomizedSearchCV |
| `train_with_early_stopping()` | Training | Boucle train/val, early stopping, sauvegarde best.pt |
| `evaluate_model()` | Evaluation | F1, AUC, Hamming, threshold search, plots |
