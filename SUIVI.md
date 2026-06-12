# Project Tracking

This document is used to record sessions, tasks, and observations. Simply fill in the corresponding sections.

## General Information

!!!! When it said "Activities Completed before session" : it is what has been done between the seesion before and this one

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
    <li><label><input type="checkbox"> Understand the project goal and the</label></li> 
    <li><label><input type="checkbox">learn and understand how all  libraries work </label></li>
    <li><label><input type="checkbox"> Set up the working environment (Git structure, folders, notebooks) </label></li>
    <li><label><input type="checkbox"> Split learning topics: PyTorch, MONAI, scikit-learn, NumPy, pandas </label></li>
    <li><label><input type="checkbox">Read the documentation provided by the supervisor </label></li>

  </ul>

  <h3>Achievable Objectives (reasonable commitment)</h3>
  <ul>
    <li><label><input type="checkbox">to be defined after the learning phase  </label></li>
    <li><label><input type="checkbox">Build a clean dataset (match images with labels, handle NaN, normalize) </label></li>
    <li><label><input type="checkbox">Implement a first baseline model for image classification </label></li>
    <li><label><input type="checkbox">Build or Fine-tune a pretrained model </label></li>
    <li><label><input type="checkbox">Organize the code into reusable modules / scripts </label></li>
    <li><label><input type="checkbox">Document the current pipeline (preprocessing → model → evaluation) </label></li>
    <li><label><input type="checkbox"> Work with DICOM images using MONAI in a more advanced way</label></li>
    <li><label><input type="checkbox"> Explore multimodal models (image + text)</label></li>
  </ul>

  <h3>Advanced Objectives (long-term / complex)</h3>
  <ul>
    <li><label><input type="checkbox">to be defined </label></li>
    <li><label><input type="checkbox">Perform advanced hyperparameter optimization (Optuna, Ray Tune) </label></li>
    <li><label><input type="checkbox">Compare multiple architectures and write an in-depth analysis </label></li>
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

## General Remarks

-

---

### Session 3 [13/03/26]

**Activities Completed before session:**

- Enzo: Already done before (vacation): Video DL finisehd. Write a first DL model with a dataset already existed.
- Hugo: Built an end-to-end text classification pipeline in `Text classification/first_phases.ipynb`
- Djouhoud: Djouhoud: use of the DICOM dataset API, start the treatment of the DICOM images 
- Aziz: Continued reading the DL book, completed a HF course on CV.
**Session Objectives:**

- Enzo: Start the traitement of the data
- Hugo: Implement a BERT-based text classifier (fine-tuning on the same dataset) to compare against this TF-IDF baseline

**Activities Completed:**

In this session, each person focused on where they wanted to start. Overall, everything will need to be completed:

- Enzo: (focus on png images only): create our custom dataset, sorting images into test/train folders based on labels. Start normalizing the images
-Aziz: Created a dataset where he linked the images to their respective XML files. He also added in the dataset hugo labels which scores the patient on 12 different diseases. `image_preprocess/merged_df_meta.csv`


**Decisions / Results:**

- Enzo: organization of the dataset
- djouhoud : focus on the dicom images 


**Next Steps for the Following Session:**

- Building a first model to train

## General Remarks
-Hugo label classification method is not great and it has room for improvement: in fact, it assigned NaN values to more than 2000 rows.
- enzo : it takes too much time to run the model on all the data, for the next session, we have to find a solution 

---

### Session 4 [16/03/26]

**Activities Completed before session:** -> meaning between the two session



**Session Objectives:**

-meeting

**Activities Completed:**


**Decisions / Results:**
-We decided to continue on our program basicly.
-enzo:  need to implement fully the pipeline he beggined
-aziz:  try to solve overfittting problem





### Session 5 [27/03/26]

**Activities Completed before session:**


**Session Objectives:**


-Discuss the work each one did so far
Enzo - Complete the model from start to finish and strive to be efficient when working on these models.

**Activities Completed:**

-Aziz: Built the first version of a computer vision model for multi-label classification: Fine-tuned a DenseNet-121 on the png dataset (7470 images). Achieved a score of 0.78 AUC
The model can be found in `image classification (png)\cv_model_01.ipynb` with the documentation in `image classification (png)\readme.md`

Enzo - The model we created works, but it’s not very efficient, so we’re going to switch to a more optimized model: we’ll take a look at PyTorch 
Enzo - Transfer it as a python file so it can more easily be used (can be found in image classification (png)\own created model) and also work on the structure itseld (OO)
-Hugo : I started building my own bert model to understand better the transformer architectiure. For that i used d2l.ia very good step by step guide to build an attention mecanism and then a (quite modern ! transformer), i implemented (without always understanding everything of course but still) multiheadedattention, Rope, add and rms norme.

**Decisions / Results:**
-Hugo : I need to clean my repo

**Next Steps for the Following Session:**

Aziz: 

-Work on improving the CV model through changing the hyper-parameters;

-Fine-tuning an open source model already trained on medical images like the HF model: `codewithdark/vit-chest-xray`

- enzo : ask to the supervisor wether it is a good idea to continu developping our own model or focus on fine-tuning
## General Remarks

### Session 6 [03/04/26]

**Activities Completed before session:**
-hugo : continue and finish with the bert classifier
- enzo : did some research on how improving our own model

**Session Objectives:**

Djouhoud:
- Create a comprehensive mapping system linking images to their diagnostic labels.
- Reorganize the `projet_dicom/` module for better modularity and maintainability.
- Convert medical DICOM images to standard RGB PNG format for model training.
- Fix alignment and synchronization issues between images and their labels.
Enzo:
- Developing and improving our own model in an effort to create something that runs in a reasonable amount of time and works reasonably well. For now, the focus is primarily on simply optimizing the model to achieve satisfactory results.
Hugo:
-engineer the prtrain, find a bigger cxray datasets to pretrain 

**Activities Completed:**

- enzo : the model has been improved, more efficient with more layers and a more optimized augmentation
-hugo : Following once again the very good chap 15 of d2l.ai, i built a pretrain functiun on a portion of MIMICXR datasets (29k)
-hugo : Fine tuned the model on open-i

-Aziz: Worked on improving the vit model using MultilabelStratifiedShuffleSplit, Stronger data augmentation, using asymmetric loss, and a richer classification head. Achieved a score of AUC: 75%


-djouhoud :
- Applied Hounsfield unit scaling and windowing technique to preserve diagnostic information in chest X-rays
- Windowing parameters properly calibrated for pulmonary imaging (window center: 40 HU, window width: 400 HU)
- Batch converted part of  DICOM files in `fragment_data_set/` in the drive  to high-resolution PNG images
- Created logical folder structure: raw inputs → processed data → analysis → outputs
- Separated `fragment_data_set/` (medical DICOM files), `NLMCXR_reports/` (annotations), and output directories
- Implemented clear separation of concerns for preprocessing, analysis, and model training pipelines
- Added comprehensive documentation in README files
- Built `png_label_mapping.csv` establishing bidirectional relationships: image_id ↔ png_filename ↔ 21 pathology labels
- Integrated XML metadata from NLMCXR dataset containing clinical annotations
- Mapped 21 distinct pathologies: Atelectasis, Cardiomegaly, Effusion, Pneumonia, Pneumothorax, Edema, Emphysema, Fibrosis, Infiltration, Mass, Nodule, Hernia, Fracture, Pleural_Thickening, Opacity, Consolidation, Granuloma, Calcinosis, Scoliosis, Atherosclerosis, Normal
- Corrected filename mismatches between DICOM originals and PNG conversions
- Fixed label assignment errors from XML parsing
- Verified image-label correspondence across entire dataset
- Implemented validation checks to prevent future misalignments

**Decisions / Results:**
-Very very Satisfying result on the bert classifier, i obtain an auc score of .94 which is not so far from the .97 that we aim for. Although my model overfitt because i need to train all the layer during fine-tunning as my model is quite small (5m param/300mfor cxr-bert)

-Aziz: The model is no longer overfitting thanks to the new loss function. I got similar results as the previous model.


**Next Steps for the Following Session:**

- enzo : the model now needs to be optimize in a deeper way 
- Hugo : fine-tune CXR-bert (microsoft) and compare with my model 

---


### Session 6 [10/04/26]


**Activities Completed before session:**
Enzo: Implement torch.lightning to the codes so that it is easier to use the GPUs. exam revision

**Session Objectives:**
Enzo : nothing due to the exams


**Activities Completed:**
-Aziz: Worked on preparing the presentation for the intermediate evaluation.


**Decisions / Results:**

**Next Steps for the Following Session:**


### Session 7 [15/04/26]


**Activities Completed before session:**
Enzo: Try to implement finetuning (better) in our own created model (juste to see, btw we will switch to the model developped with Aziz) to compare and se the differences with a model not so good, exam revision

**Session Objectives:**
Enzo : nothing due to the exams


**Activities Completed:**



**Decisions / Results:**

**Next Steps for the Following Session:**

### Session 8 [05/05/26]

**Activities Completed before session:** 
Enzo : Watched the video about multimodel implementation : 
Stanford CS224N NLP with Deep Learning | 2023 | Lecture 16 - Multimodal Deep Learning, Douwe Kiela
Start reading the article mentionned in the video : Learning Transferable Visual Models From Natural Language Supervision

Aziz: Used also the learning material of Stanford CS224N to learn about multimodal implemebtatiob.
Read several blog posts about techniques we can use of the multimodal implementation
Djouhoud: Downloaded the dataset to the Telecom GPU server and verified the total number of DICOM files.


**Session Objectives:**
 - Put averything together and for the planning for the next steps
Djouhoud: Preprocessed DICOM files using the multi-window windowing method on the Telecom GPU server.
**Activities Completed:**
Enzo : discuss about what I've learned with the others and focus on the planning for the next sessions
Aziz: Read online blogs about multi-model implementation "Building a Multimodal Classifier in PyTorch: A Step-by-Step Guide".

**Decisions / Results:**
We decided to work one more week on the Deecom but give up if it still doesn't work, for the Fusion Phase,we shared what we know and organize what people are going to do 
Djouhoud: Encountered disk quota limitation during DICOM preprocessing. The output directory exceeded available storage on the Telecom GPU server.
**Next Steps for the Following Session:**
Enzo: try to think about a simple API
Aziz: Help Elias in the multi-model implementation.
Djouhoud: Explore alternative DICOM processing technique using 3D volumetric representation.
### Session 8 [13/05/26]

**Activities Completed before session:** 


Enzo : Made a simple API and tried if it works with enzo's model : https://test-api-2-production.up.railway.app/docs   . The Api take in consideration the model according to the dictionnary containing all the values each neuron has after training the model. To use the API with another model, we only need to change the `model_full.pth`. The API use `Dockers`, to containes things and the deployement has been mage using `Railway`
**Session Objectives:**
Enzo: work with the others with the multimodal goals.

Aziz: 
- Read papers on multimodal fusion via bidirectional cross-attention:
  - **LXMERT** (Tan & Bansal, EMNLP 2019): two separate encoders (text + image) that mutually enrich each other via cross-attention, with a final fusion layer
  - **ViLBERT** (Lu et al., NeurIPS 2019): dual-stream architecture with cross-attention between visual and text tokens, a precursor to bimodal approaches
  - **TransCheX** (cf. supervisor reference): adaptation of these principles to the chest X-ray radiology domain
  - Explored **MONAI (NVIDIA)** as a potential framework for medical data handling and multimodal training



**Activities Completed:**
djouhoud: -  fine-tuning of a DenseNet-121 model using a preprocessed dataset (windowing applied and reduced image size).
- Training was performed for half of the data set and 5 epochs to test whether the model can run and converge successfully.
- start to preprocess the dataset for the volume methode 
enzo : discuss about how we should do the multimodal implementation. Read about that

Aziz: 
- Implemented `multimodal_fusion/` module combining the custom text encoder (BERT MLM, d=256) and ViT (`codewithdark/vit-chest-xray`, d=768) via bidirectional cross-attention projected to d=512

**Decisions / Results:**
Djouhoud: The windowing method is not suitable for Google Colab, as it significantly increases training time even when using only half of the dataset, and the resulting metrics are not satisfactory.


**Next Steps for the Following Session:**
enzo : continue to think of ways to improve our multimodal implementation.
djouhoud : focusing in the volume methode

### Session 9 [27/05/26]

**Activities Completed before session:** 
enzo : improve the model : The code trains a multimodal model that combines a chest X‑ray processed by EfficientNet‑B0 with TF‑IDF text features from clinical reports. The two embeddings are concatenated and passed through a fusion MLP that predicts 14 thoracic pathologies. Training happens in two stages: first the fusion head is learned while the image branch is frozen, then the entire network is fine‑tuned ----> see the branch enzo_multimodal for now.

**Session Objectives:**
enzo : use the .pth of the other's model to check if everything works with the model done.

**Activities Completed:**


**Decisions / Results:**



**Next Steps for the Following Session:**


### Session 10 [10/06/26]

**Activities Completed before session:** 
enzo : use the .pth of the other's model to check if everything works with the model done. It works, not so good, AUC very low.

**Session Objectives:**
elias : again learning about single stream transformer in order to implement one multimodal head.
enzo: create the flyer for the presentation



**Activities Completed:**

Aziz: Built the `multimodal_fusion/` module that combines a custom BERT text encoder and a Vision Transformer (ViT) image encoder into a single multi-label classifier for 21 chest pathologies, using bidirectional cross-attention as the fusion mechanism.
enzo : the canva of the Flyer has been created and some information has already been written.

**`BidirectionalCrossAttention`**

Both modalities are projected to a common space (d=512) before attention:

- Text tokens `(B, L, 256)` → Linear projection → `(B, L, 512)`
- Image tokens `(B, 197, 768)` → Linear projection → `(B, 197, 512)`
- Two cross-attention directions computed **in parallel**:
  - Text → Image (`t2i`): text queries attend over image keys/values
  - Image → Text (`i2t`): image queries attend over text keys/values
- Each branch has a residual connection, FFN (512 → 1024 → 512), and LayerNorm

**`MultimodalFusion`**

| Component | Details |
|-----------|---------|
| Text encoder | Custom BERT (d=256, 6 layers, 8 heads, RoPE+RMSNorm+SiLU) — 4.3M params |
| Image encoder | `codewithdark/vit-chest-xray` ViT-B/16 (d=768, 197 tokens) — 86M params |
| Cross-attention | BidirectionalCrossAttention, 8 heads, d_model=512 |
| Text pooling | CLS token (index 0) of fused text sequence |
| Image pooling | Learned soft-attention over all 197 fused image tokens |
| Classification head | Linear(1024 → 512) + LayerNorm + GELU + Dropout + Linear(512 → 21) |
| **Total parameters** | **95,931,925** |


### Results

**Training curve (key epochs):**

| Epoch | Phase | Val AUC |
|-------|-------|---------|
| 1 | Frozen | 0.6247 |
| 3 | Frozen | 0.9206 |
| 4 | Unfrozen | 0.9596 |
| 8 | Unfrozen | 0.9879 ★ |
| 15 | Unfrozen | 0.9881 ★ best |
| 22 | Unfrozen | early stop |

**Per-class AUC on test set (968 samples):**

| Pathology | AUC |
|-----------|-----|
| Atelectasis | 0.9875 |
| Cardiomegaly | 0.9887 |
| Effusion | 0.9731 |
| Pneumonia | 0.9670 |
| Pneumothorax | 0.9969 |
| Edema | 0.9922 |
| Emphysema | 0.9604 |
| Fibrosis | 0.9550 |
| Infiltration | 0.9839 |
| Mass | 0.9946 |
| Nodule | 0.9918 |
| Hernia | 0.9997 |
| Fracture | 0.9922 |
| Pleural_Thickening | 1.0000 |
| Opacity | 0.9999 |
| Consolidation | 0.9936 |
| Granuloma | 0.9999 |
| Calcinosis | 0.9974 |
| Scoliosis | 0.9950 |
| Atherosclerosis | 1.0000 |
| Normal | 0.9920 |
| **Mean AUC** | **0.9886** |

**Best val AUC: 0.9881 — Test AUC: 0.9886**

Checkpoint saved at: `multimodal_fusion/checkpoints/multimodal_fusion.pt`
More documentation can be found on `multimodal_fusion/checkpoints/README.md`



**Decisions / Results:**



**Next Steps for the Following Session:**
elias : implementing this single stream transformer (see branch1 before merging) 
enzo : complete the flyer and finalise the API with the new model Aziz made.
