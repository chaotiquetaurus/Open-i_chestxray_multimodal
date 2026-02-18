```mermaid
gantt
    title Project Planning - Beyond Pixel
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m

    section Initial Phase
    Collective ML/DL Phase & Resources         :done, p1, 2026-02-18, 2026-03-01

    section Analysis
    Exploratory Analysis - Open-I Chest X-ray  :p2, 2026-03-16, 2026-03-29

    section Parallel Development
    ML Pipeline Construction                   :p3, 2026-03-30, 2026-04-25
    Vision Models (Classification)             :p4, 2026-03-30, 2026-04-25
    NLP Report Extraction                      :p5, 2026-03-30, 2026-04-25
    Support & Resilience                       :p6, 2026-03-30, 2026-04-25

    section Integration
    Finalization + Multimodal Fusion           :p7, after p3, 19d

    section Evaluation
    Evaluation & Interpretation                :p8, after p7, 27d

    section Presentation
    Final Presentation                         :p9, after p8, 10d
```
