```mermaid
%%{init: {
    "theme": "default",
    "themeVariables": {
        "fontSize": "9px",
        "fontFamily": "Inter",
        "textColor": "#e0e0e0",
        "taskTextColor": "#ffffff",
        "taskColor": "#3a7bd5",
        "taskBorderColor": "#2a5fa8",
        "lineColor": "#90caf9",
        "sectionBkgColor": "#0d47a1",
        "sectionBkgColor2": "#0a3d91"
    }
}}%%

gantt
    title Project Timeline – Multimodal X‑Ray Diagnosis
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Preparation
    Resources                 :a1, 2025-02-18, 2025-03-01
    Planning Check (All)      :a2, 2025-03-02, 1d

    section Week 1–2: Cleaning & Analysis
    Training (M1)             :m1a, 2025-03-03, 2025-03-15
    Training (M2)             :m2a, 2025-03-03, 2025-03-15
    Training (M3)             :m3a, 2025-03-03, 2025-03-15
    Training (M4)             :m4a, 2025-03-03, 2025-03-15
    Training (M5)             :m5a, 2025-03-03, 2025-03-15

    section Week 3–4: Deep Cleaning
    Raw Data                  :m1b, 2025-03-16, 2025-03-29
    Image Prep                :m2b, 2025-03-16, 2025-03-29
    Text Prep                 :m3b, 2025-03-16, 2025-03-29
    Splits & Repro            :m4b, 2025-03-16, 2025-03-29
    Pipeline Struct.          :m5b, 2025-03-16, 2025-03-29

    section Core Dev
    ML Pipeline (M1)          :m1c, 2025-03-30, 2025-04-25
    Vision Models (M2)        :m2c, 2025-03-30, 2025-04-25
    NLP Reports (M3)          :m3c, 2025-03-30, 2025-04-25
    Support (M4)              :m4c, 2025-03-30, 2025-04-25
    Support (M5)              :m5c, 2025-03-30, 2025-04-25

    section Fusion
    Fusion Phase (All)        :fusion, 2025-04-26, 2025-05-14

    section Evaluation
    Evaluation (All)          :eval, 2025-05-15, 2025-06-11

    section Final
    Final Presentation (All)  :present, 2025-06-11, 2025-06-30
```

![alt text](image.png)

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
      <td>ML Pipeline Construction</td>
      <td>Member 1</td>
      <td>Dataset download, preprocessing, training, evaluation, interpretation</td>
      <td>Mar 30 – Apr 25</td>
      <td></td>
    </tr>
    <tr>
      <td>Vision Models (Classification)</td>
      <td>Member 2</td>
      <td>Image preprocessing, augmentation, validation, transfer learning</td>
      <td>Mar 30 – Apr 25</td>
      <td></td>
    </tr>
    <tr>
      <td>NLP Report Extraction</td>
      <td>Member 3</td>
      <td>Tokenization, information extraction, language models</td>
      <td>Mar 30 – Apr 25</td>
      <td></td>
    </tr>
    <tr>
      <td>Support & Resilience</td>
      <td>Members 4</td>
      <td>Support other members in case of issues, begin integration work</td>
      <td>Mar 30 – Apr 25</td>
      <td></td>
    </tr>
    <tr>
      <td>Support & Resilience</td>
      <td>Members 5</td>
      <td>Support other members in case of issues, begin integration work</td>
      <td>Mar 30 – Apr 25</td>
      <td></td>
    </tr>
    <tr>
      <td>Individual Finalization + Fusion</td>
      <td>All</td>
      <td>Finalize individual components, start multimodal integration</td>
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