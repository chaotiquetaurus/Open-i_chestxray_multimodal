# MM-MIA — Multi-Modal Medical Imaging Analysis

**Projet proj104 2025-2026 — Télécom Paris**
Diagnostic multi-modal de radiographies thoraciques combinant **analyse de texte** (rapports radiologiques) et **analyse d'image** (DICOM/PNG) à partir du dataset [Open-I](https://openi.nlm.nih.gov/).

---

## 📂 Architecture du repo

```
MM-MIA/
├── pyproject.toml              # Un seul environnement Python pour tout le projet
├── requirements-lock.txt       # Versions exactes (reproductibilité)
├── .gitignore / .gitattributes # Configs Git + LFS (datasets et modèles)
├── .lfsconfig                  # LFS pointe vers GitLab ENST uniquement
│
├── SUIVI.md                    # Suivi équipe (à la racine pour visibilité)
│
├── src/mmmia/                  # Code Python (package installable)
│   ├── text_classification/    # NLP sur rapports radiologiques
│   │   ├── notebooks/          # Notebooks d'exploration et d'entraînement
│   │   ├── models/             # Architectures (BERT, classifier, encoder...)
│   │   └── scripts/            # Pipelines (finetune, pretrain, kfold...)
│   ├── image_classification/   # CV sur images PNG
│   │   ├── notebooks/          # cv_model_vit*, cv_model_01...
│   │   ├── own_model/          # Modèle TinyVGG custom (Lightning)
│   │   └── scripts/            # Scripts cluster (artishow_pytorch_*)
│   ├── dicom/                  # Pipeline DICOM (preprocessing, volumes 3D)
│   │   ├── notebooks/          # Preprocessing, fine-tuning DenseNet/3D...
│   │   └── scripts/            # CLI de preprocessing
│   ├── multimodal_fusion/      # Fusion image+texte (ViT + BERT, cross-attention)
│   ├── api/                    # API FastAPI (medical-api)
│   └── labeling/               # Outils de labellisation (Open-I XML)
│
├── data/                       # Datasets (Git LFS — voir data/README.md)
│   ├── text_classification/    # CSV labellisés
│   ├── image_classification/   # train.zip, test.zip
│   ├── dicom/
│   │   ├── raw/                # .dcm originaux
│   │   ├── preprocessed_multiwindow/
│   │   └── metadata/           # CSV de métadonnées
│   └── labeling/               # NLMCXR_reports + datasets
│
├── models/                     # Poids entraînés (Git LFS — voir models/README.md)
│   └── api/                    # model_full.pth (modèle servi par l'API)
│
├── docs/                       # Documentation projet
│   ├── planning.md
│   ├── cluster_ml_pipeline.md
│   ├── text_classification_architecture.md
│   ├── dicom_overview.md
│   ├── dicom_preprocessing.md
│   ├── api.md
│   ├── figures/                # Courbes ROC, training curves...
│   └── meetings/               # Comptes-rendus réunions superviseur
│
├── scripts/                    # Scripts shell (cluster, etc.)
└── tests/                      # Tests pytest
```

---

## 🚀 Installation

### Prérequis
- Python ≥ 3.10
- [Git LFS](https://git-lfs.com/) (`sudo apt install git-lfs && git lfs install`)
- CUDA (optionnel, pour entraînement GPU)

### Cloner le repo
```bash
git clone git@gitlab.enst.fr:proj104/2025-2026/MM-MIA.git
cd MM-MIA
git lfs pull   # télécharge datasets et modèles (peut prendre du temps)
```

### Setup environnement
```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev,notebook]   # installe mmmia en mode editable + dev tools + jupyter
```

Pour reproduire **exactement** les versions de l'équipe :
```bash
pip install -r requirements-lock.txt
```

---

## 🧪 Utilisation

### Lancer un notebook
```bash
jupyter lab src/mmmia/text_classification/notebooks/
```

### Lancer l'API
```bash
uvicorn mmmia.api.main:app --reload
```

### Entraîner un modèle (text classification)
```bash
python -m mmmia.text_classification.scripts.finetune
```

---

## 📊 Données & modèles

Tous les fichiers lourds (`*.dcm`, `*.zip`, `*.pth`, `*.ckpt`, PNG sous `data/`) sont gérés via **Git LFS sur GitLab ENST**.

- `git lfs pull` télécharge tout
- `git lfs pull --include="data/text_classification/**"` ne télécharge qu'un sous-ensemble

**📖 Lire impérativement avant de commit du code** : [`docs/LFS_WORKFLOW.md`](docs/LFS_WORKFLOW.md) — guide complet (ajouter un fichier, conventions, troubleshooting).

Voir aussi [`data/README.md`](data/README.md) et [`models/README.md`](models/README.md) pour la structure.

---

## 🤝 Workflow Git

- Une **branche par membre / par feature** (`feat/<name>`, `fix/<name>`).
- Merge via **Merge Request** sur GitLab après review.
- `main` est protégé : pas de push direct sauf quick fix urgent.
- Le suivi des avancées se fait dans [`SUIVI.md`](SUIVI.md).

### Remotes
- `origin` = `gitlab.enst.fr` (principal, avec LFS)
- `github` = miroir public read-only (code uniquement, pas les data LFS)

---

## 👥 Équipe

Voir [`SUIVI.md`](SUIVI.md) pour la répartition des rôles et l'avancement.

Superviseurs : voir [`docs/meetings/`](docs/meetings/).
