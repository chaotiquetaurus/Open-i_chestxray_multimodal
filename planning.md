```mermaid


gantt
    title Project Timeline – Multimodal X‑Ray Diagnosis
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Preparation
    Resources                 :a1, 2025-02-18, 2025-03-7
    Planning Check (All)      :a2, 2025-03-08, 1d

    section Week 1 Training Data
    Training (M1)             :m1a, 2025-03-09, 2025-03-15
    Training (M2)             :m2a, 2025-03-09, 2025-03-15
    Training (M3)             :m3a, 2025-03-09, 2025-03-15
    Training (M4)             :m4a, 2025-03-09, 2025-03-15
    Training (M5)             :m5a, 2025-03-09, 2025-03-15

    section Week 2  Data
    Raw Data__                  :m1b, 2025-03-16, 2025-03-23
    Image Prep                :m2b, 2025-03-16, 2025-03-23
    Text Prep__                   :m3b, 2025-03-16, 2025-03-23
    Splits & Repro            :m4b, 2025-03-16, 2025-03-23
    Pipeline Struct.          :m5b, 2025-03-16, 2025-03-23

    section Week 3 Training machine learning
    Raw Data__                  :m1b, 2025-03-24, 2025-03-30
    Image Prep                :m2b, 2025-03-24, 2025-03-30
    Text Prep__                :m3b, 2025-03-24, 2025-03-30
    Splits & Repro            :m4b, 2025-03-24, 2025-03-30
    Pipeline Struct.          :m5b, 2025-03-24, 2025-03-30

    section Core Dev
    computer vision models         :m1c, 2025-03-31, 2025-04-25
    extract information       :m2c, 2025-03-31, 2025-04-25

    section Fusion
    Fusion Phase (All)        :fusion, 2025-04-26, 2025-05-14

    section Evaluation
    Evaluation (All)          :eval, 2025-05-15, 2025-06-11

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
      <td>Mar 3 – Mar 14</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Image Preprocessing and image quality</td>
      <td>Member 2</td>
      <td>Organize label, Inspect image quality, compute image statistics : brightness histogram, pixel intensity distribution, image size distribution, Start designing the image preprocessing function</td>
      <td>Mar 3 – Mar 14</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Image Preprocessing and image quality</td>
      <td>Member 3</td>
      <td>Organize label, Inspect radiology report, clean text, compute text statistics : report length distribution, vocab size, term frequencies, identify incomplete ort inconsistent report</td>
      <td>Mar 3 – Mar 14</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Split, Distribution and Reproducibility</td>
      <td>Member 4</td>
      <td>Organize label, Analyse label distribution, identify class imbalance, check patient-level consistency prepare the project environment</td>
      <td>Mar 3 – Mar 14</td>
      <td></td>
    </tr>
    <tr>
      <td>Formation Pipeline Structure and Consistency Checks</td>
      <td>Member 5</td>
      <td>Organize label, Build skeleton of the training pipeline, Define expected intput formats (image tensors, text vectors), Create mock data for testing, start litteratur reviex on multimodal pipelines</td>
      <td>Mar 3 – Mar 14</td>
      <td></td>
    </tr>
    <tr>
      <td>Check planning</td>
      <td>All</td>
      <td>Checking and uptading the planning if necessary</td>
      <td>Mar 15</td>
      <td></td>
    </tr>
    <tr>
      <td>Raw Data Manager</td>
      <td>Member 1</td>
      <td>Download dataset, perform raw cleaning, run global statistics, produce a first data quality report</td>
      <td>Mar 16 – Mar 29</td>
      <td></td>
    </tr>
    <tr>
      <td>Image Preprocessing and image quality</td>
      <td>Member 2</td>
      <td>Inspect image quality, compute image statistics : brightness histogram, pixel intensity distribution, image size distribution, Start designing the image preprocessing function</td>
      <td>Mar 16 – Mar 29</td>
      <td></td>
    </tr>
    <tr>
      <td>Image Preprocessing and image quality</td>
      <td>Member 3</td>
      <td>Inspect radiology report, clean text, compute text statistics : report length distribution, vocab size, term frequencies, identify incomplete ort inconsistent report</td>
      <td>Mar 16 – Mar 29</td>
      <td></td>
    </tr>
    <tr>
      <td>Split, Distribution and Reproducibility</td>
      <td>Member 4</td>
      <td>Analyse label distribution, identify class imbalance, check patient-level consistency prepare the project environment</td>
      <td>Mar 16 – Mar 29</td>
      <td></td>
    </tr>
    <tr>
      <td>Pipeline Structure and Consistency Checks</td>
      <td>Member 5</td>
      <td>Build skeleton of the training pipeline, Define expected intput formats (image tensors, text vectors), Create mock data for testing, start litteratur reviex on multimodal pipelines</td>
      <td>Mar 16 – Mar 29</td>
      <td></td>
    </tr>
    <tr>
      <td>Implement computer vision models</td>
      <td>Member 1, 2, 4</td>
      <td>Implement computer vision models for medical image classification (data preprocessing, augmentation, and validation strategies). Investigate transfer learning methods. Everyone will have to participate because it is linked with what everyone worked at the last step.</td>
      <td>Mar 30 – Apr 25</td>
      <td></td>
    </tr>
    <tr>
      <td>extract information</td>
      <td>Member 3, 5</td>
      <td>Use NLP methods to extract information from medical reports.Everyone will have to participate because it is linked with what everyone worked at the last step.</td>
      <td>Mar 30 – Apr 25</td>
      <td></td>
    </tr>
    <tr>
      <td>Individual Finalization + Fusion</td>
      <td>All</td>
      <td>Finalize individual components, start multimodal integration. everyone will participate because everyone needs to implement his precedent task to the fusion</td>
      <td>Apr 26 – May 14</td>
      <td></td>
    </tr>
    <tr>
      <td>Evaluation & Interpretation</td>
      <td>All</td>
      <td>Systematic model comparison, metrics, explainability, vision/text relationship</td>
      <td>May 15 – Jun 11</td>
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