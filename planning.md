```mermaid

gantt
    title Project Timeline – Multimodal X‑Ray Diagnosis
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Preparation
    Resources                 :a1, 2025-02-18, 2025-03-07
    Planning Check (All)      :a2, 2025-03-08, 1d

    section Week 1 Training Data
    Training (M1)             :m1a, 2025-03-09, 2025-03-15
    Training (M2)             :m2a, 2025-03-09, 2025-03-15
    Training (M3)             :m3a, 2025-03-09, 2025-03-15
    Training (M4)             :m4a, 2025-03-09, 2025-03-15
    Training (M5)             :m5a, 2025-03-09, 2025-03-15

    section Week 2 Data
    Raw Data__                  :m1b_clean, 2025-03-16, 2025-03-23
    Image Prep                :m2b_clean, 2025-03-16, 2025-03-23
    Text Prep__                 :m3b_clean, 2025-03-16, 2025-03-23
    Splits & Repro            :m4b_clean, 2025-03-16, 2025-03-23
    Pipeline Struct.          :m5b_clean, 2025-03-16, 2025-03-23

    section Week 3 Training Machine Learning
    Training CV Models        :m1c_train, 2025-03-24, 2025-03-30
    Training NLP Extraction   :m2c_train, 2025-03-24, 2025-03-30
    Planning Check (All)      :a3, 2025-03-30, 1d

    section Core Dev
    Computer Vision Models    :m1c_dev, 2025-03-31, 2025-04-24
    Extract Information       :m2c_dev, 2025-03-31, 2025-04-24

    section Debug Core Dev
    Debug CV Models           :m1c_debug, 2025-04-25, 2025-05-04
    Debug Extraction          :m2c_debug, 2025-04-25, 2025-05-04

    section Fusion
    Fusion Phase (All)        :fusion, 2025-05-05, 2025-05-24

    section Evaluation & Debug
    Debug + Evaluation (All)  :eval, 2025-05-25, 2025-06-11

    section Final
    Final Presentation (All)  :present, 2025-06-11, 2025-06-30
```


## Planning

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Responsable</th>
      <th>Details</th>
      <th>Date (Planned)</th>
      <th>Date (Completed)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Collective ML/DL Phase & Resources</td>
      <td>All</td>
      <td>Reading books/resources (StatLearning, scikit-learn, D2L, PyTorch videos, Hugging Face)</td>
      <td>Feb 18 – Mar 1</td>
      <td></td>
    </tr>
    <tr>
      <td>Check planning</td>
      <td>All</td>
      <td>Checking and uptading the planning if necessary</td>
      <td>Mar 2</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Raw Data Manager</td>
      <td>Member 1</td>
      <td>Organize label, Download dataset, perform raw cleaning, run global statistics, produce a first data quality report</td>
      <td>Mar 9 – Mar 15</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Image Preprocessing and image quality</td>
      <td>Member 2</td>
      <td>Organize label, Inspect image quality, compute image statistics : brightness histogram, pixel intensity distribution, image size distribution, Start designing the image preprocessing function</td>
      <td>Mar 9 – Mar 15</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Image Preprocessing and image quality</td>
      <td>Member 3</td>
      <td>Organize label, Inspect radiology report, clean text, compute text statistics : report length distribution, vocab size, term frequencies, identify incomplete ort inconsistent report</td>
      <td>Mar 9 – Mar 15</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Split, Distribution and Reproducibility</td>
      <td>Member 4</td>
      <td>Organize label, Analyse label distribution, identify class imbalance, check patient-level consistency prepare the project environment</td>
      <td>Mar 9 – Mar 15</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Pipeline Structure and Consistency Checks</td>
      <td>Member 5</td>
      <td>Organize label, Build skeleton of the training pipeline, Define expected intput formats (image tensors, text vectors), Create mock data for testing, start litteratur reviex on multimodal pipelines</td>
      <td>Mar 9 – Mar 15</td>
      <td></td>
    </tr>
    <tr>
      <td>Raw Data Manager</td>
      <td>Member 1</td>
      <td>Download dataset, perform raw cleaning, run global statistics, produce a first data quality report</td>
      <td>Mar 16 – Mar 23</td>
      <td></td>
    </tr>
    <tr>
      <td>Image Preprocessing and image quality</td>
      <td>Member 2</td>
      <td>Inspect image quality, compute image statistics : brightness histogram, pixel intensity distribution, image size distribution, Start designing the image preprocessing function</td>
      <td>Mar 16 – Mar 23</td>
      <td></td>
    </tr>
    <tr>
      <td>Image Preprocessing and image quality</td>
      <td>Member 3</td>
      <td>Inspect radiology report, clean text, compute text statistics : report length distribution, vocab size, term frequencies, identify incomplete ort inconsistent report</td>
      <td>Mar 16 – Mar 23</td>
      <td></td>
    </tr>
    <tr>
      <td>Split, Distribution and Reproducibility</td>
      <td>Member 4</td>
      <td>Analyse label distribution, identify class imbalance, check patient-level consistency prepare the project environment</td>
      <td>Mar 16 – Mar 23</td>
      <td></td>
    </tr>
    <tr>
      <td>Pipeline Structure and Consistency Checks</td>
      <td>Member 5</td>
      <td>Build skeleton of the training pipeline, Define expected intput formats (image tensors, text vectors), Create mock data for testing, start litteratur reviex on multimodal pipelines</td>
      <td>Mar 16 – Mar 23</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Implement computer vision models</td>
      <td>Member 1, 2, 4</td>
      <td>Implement computer vision models for medical image classification (data preprocessing, augmentation, and validation strategies). Investigate transfer learning methods. Everyone will have to participate because it is linked with what everyone worked at the last step.</td>
      <td>Mar 24 – Apr 30</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation extract information</td>
      <td>Member 3, 5</td>
      <td>Use NLP methods to extract information from medical reports.Everyone will have to participate because it is linked with what everyone worked at the last step.</td>
      <td>Mar 24 – Apr 30</td>
      <td></td>
    </tr>
    <tr>
      <td>Check planning</td>
      <td>All</td>
      <td>Checking and uptading the planning if necessary</td>
      <td>Apr 30</td>
      <td></td>
    </tr>
    <tr>
      <td>Implement computer vision models</td>
      <td>Member 1, 2, 4</td>
      <td>Implement computer vision models for medical image classification (data preprocessing, augmentation, and validation strategies). Investigate transfer learning methods. Everyone will have to participate because it is linked with what everyone worked at the last step.</td>
      <td>Mar 31 – Apr 24</td>
      <td></td>
    </tr>
    <tr>
      <td>extract information</td>
      <td>Member 3, 5</td>
      <td>Use NLP methods to extract information from medical reports.Everyone will have to participate because it is linked with what everyone worked at the last step.</td>
      <td>Mar 31 – Apr 24</td>
      <td></td>
    </tr>
    <tr>
      <td>debug Implement computer vision models</td>
      <td>Member 1, 2, 4</td>
      <td>time to debug, everyone check the part of the others, try to give another points of views</td>
      <td>Apr 25 - May 4</td>
      <td></td>
    </tr>
    <tr>
      <td>debug extract information</td>
      <td>Member 3, 5</td>
      <td>time to debug, everyone check the part of the others, try to give another points of views</td>
      <td>Apr 25 – May 4</td>
      <td></td>
    </tr>
    <tr>
      <td>Individual Finalization + Fusion</td>
      <td>All</td>
      <td>Finalize individual components, start multimodal integration. everyone will participate because everyone needs to implement his precedent task to the fusion</td>
      <td>May 5 – May 24</td>
      <td></td>
    </tr>
    <tr>
      <td>Debug / Evaluation & Interpretation</td>
      <td>All</td>
      <td>everyone also check the work of the other to test and participate of the debug, with another point of view</td>
      <td>May 25 – Jun 11</td>
      <td></td>
    </tr>
    <tr>
      <td>Final Presentation</td>
      <td>All</td>
      <td>Prepare slides, rehearsals, supervisor feedback</td>
      <td>Jun 11 – End</td>
      <td></td>
    </tr>
  </tbody>
</table>

---