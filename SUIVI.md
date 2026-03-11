# Project Tracking

This document is used to record sessions, tasks, and observations. Simply fill in the corresponding sections.

## General Information

- **Project Name: Beyond Pixel**
- **Project Lead: AHMED SAID Djouhoud**
- **Start Date: 11/02**
- **Expected End Date: No fixed end date, continuous improvement approach**
- **Member 1 : enzo**
- **Member 2 : Djouhoud**
- **Member 3 : Hugo**
- **Member 3 : Aziz**
- **Member 4 : Elias**

---

## Objectives Checklist

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Objectives</title>
</head>
<body>

  <h3>Short-term Objectives (quick wins)</h3>
  <ul>
    <li><label><input type="checkbox"> Try to match the image to the problem</label></li> 
    <li><label><input type="checkbox">learn and understand how all  libraries work </label></li>
    <li><label><input type="checkbox"> </label></li>
  </ul>

  <h3>Achievable Objectives (reasonable commitment)</h3>
  <ul>
    <li><label><input type="checkbox">to be defined after the learning phase  </label></li>
    <li><label><input type="checkbox">to be defined </label></li>
    <li><label><input type="checkbox">to be defined </label></li>
  </ul>

  <h3>Advanced Objectives (long-term / complex)</h3>
  <ul>
    <li><label><input type="checkbox">to be defined </label></li>
    <li><label><input type="checkbox">to be defined </label></li>
    <li><label><input type="checkbox">to be defined </label></li>
  </ul>

</body>
</html>

---

## Session Tracking

Sessions are listed by date.

### Session 1 [20/02/26]

**Session Objectives:**

Complete the planning and set objectives for each member of the team, starting with the short term

**Activities Completed:**

- Planning
- Division of tasks

**Decisions / Results:**

- The current schedule is not final; After the training phase it may change if necessary.

**Next Steps for the Following Session:**

- Learning about Machine Learning and Deep Learning Libraries
- Setting up a planning schedule that should be effective. The lack of knowledge but especially the fact of not knowing to what extent we will be capable of doing things (a problem also mentioned by our supervisor who advised us to do things little by little even if we don't go all the way) pushed us to create a very progressive planning with ultimately few details on the pure practice of what we will do. In theory and ideas, the planning is however very precise.

**Decisions / Results:**

- Planning.md

**Next Steps for the Following Session:**

- Review all the documentation provided by our supervisor

### Session 2 [23/02/26]

**Session Objectives:**

- Start reviewing the documentation

**Activities Completed:**

In this session, each person focused on where they wanted to start. Overall, everything will need to be completed:

- Enzo: Started watching the 24-hour video on PyTorch, a deep learning library.
- Djouhoud : started to learn about the MONAI project
- Aziz: Started watching videos about scikit-learn library
- Hugo: A lot about Numpy and panda library. Started learning about basic ML, and application with scikit-learn, through https://inria.github.io/scikit-learn-mooc/index.html. 

**Decisions / Results:**

- Enzo: Video not finished but well underway
- Aziz: understood the basics of linear models, SVMs, and SGD.
- Hugo : Understood the basic of linear models. I now am able to manipulate panda and numpy a bit better. Started through scikit, to learn how to analyze a basic (here 2d) datasets.

**Next Steps for the Following Session:**

- Enzo: Continue the video and also review certain points from the presented book: "https://scikit-learn.org/stable/" on machine learning
- Aziz: continue reading more about the scikit-learn library
- Hugo: continue with basic Ml and scikit.

### Session 3 — Text Classification Baseline

**Session Objectives:**

- Build a first baseline model for classifying radiology reports from the Open-I dataset

**Activities Completed:**

- Built an end-to-end text classification pipeline in `Text classification/first_phases.ipynb`

**Pipeline:**

1. **Data loading**: Parsed 2314 XML radiology reports (indication + findings fields) and loaded 12 pathology labels using TorchXRayVision's Open-I loader.
2. **Preprocessing**: Removed anonymization tokens (XXXX, year, old), filtered stopwords and punctuation, then lemmatized using spaCy (`en_core_web_sm`).
3. **TF-IDF vectorization**: Vectorized indication and findings separately (max 2500 features each, min_df=2), concatenated into a single sparse matrix (1074 features, 98% sparsity).
4. **Binary classification (Normal vs. Pathology)**: Logistic regression with L1 regularization (C=0.83), tuned via RandomizedSearchCV. **AUC = 0.96**, accuracy = 0.90 (vs. 0.70 dummy baseline).
5. **Multi-label classification (12 pathologies)**: OneVsRest logistic regression with ElasticNet (C=5.4, l1_ratio=0.8). **Macro AUC = 0.91**. Per-pathology AUC ranges from 0.70 (Emphysema) to 0.98 (Granuloma).

**Results Analysis:**

- Pathologies with specific vocabulary (Granuloma, Atelectasis, Nodule) score very well (AUC > 0.94) because TF-IDF directly captures discriminant keywords like "granuloma", "atelectasis", "nodule".
- Pathologies with subtle or shared vocabulary (Emphysema AUC=0.70, Pneumonia AUC=0.81) perform worse — their reports use generic terms ("dyspnea", "opacity") shared across multiple conditions.
- Macro F1 = 0.53 is low: the model struggles with rare classes (Edema: 3 test samples, Nodule: 6) and produces many false positives when it lacks confident signal.
- The t-SNE visualization shows poor separation between Normal and Pathological reports in TF-IDF space, confirming the feature representation is limited.

**Why We Need BERT / Deep Learning:**

- **Negation handling**: TF-IDF treats "no pneumonia" and "pneumonia" identically — both contribute to the pneumonia keyword. A contextual model like BERT understands negation and would correctly distinguish these cases.
- **Semantic understanding**: "Cardiac silhouette is enlarged" means Cardiomegaly, but TF-IDF only sees individual words. BERT captures the full meaning of phrases.
- **Rare class performance**: With few training examples, TF-IDF features are too sparse to learn reliable patterns. Pre-trained language models bring prior medical/language knowledge that helps generalize from fewer examples.
- **Multi-word expressions**: Discriminant terms like "pleural effusion", "calcified granuloma", or "interstitial markings" are split by TF-IDF into independent tokens. BERT processes them as coherent concepts.

**Next Steps for the Following Session:**

- Implement a BERT-based text classifier (fine-tuning on the same dataset) to compare against this TF-IDF baseline
- Explore image-based classification to start the multimodal approach

---

## General Remarks

-

---
