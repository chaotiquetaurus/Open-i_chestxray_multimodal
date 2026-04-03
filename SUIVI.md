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
-Aziz: Worked on the learning material.
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

**Activities Completed before session:**

- meeting with Nikolas

**Session Objectives:**



**Activities Completed:**


**Decisions / Results:**



**Next Steps for the Following Session:**


## General Remarks


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

**Decisions / Results:**

**Next Steps for the Following Session:**

Aziz: 

-Work on improving the CV model through changing the hyper-parameters;

-Fine-tuning an open source model already trained on medical images like the HF model: `codewithdark/vit-chest-xray`

- enzo : ask to the supervisor wether it is a good idea to continu developping our own model or focus on fine-tuning
## General Remarks

### Session 6 [03/04/26]

**Activities Completed before session:**


**Session Objectives:**


- Developing and improving our own model in an effort to create something that runs in a reasonable amount of time and works reasonably well. For now, the focus is primarily on simply optimizing the model to achieve satisfactory results.

**Activities Completed:**

- enzo : the model has been improved

**Decisions / Results:**

**Next Steps for the Following Session:**

- enzo : the model now needs to be optimize in a deeper way 




