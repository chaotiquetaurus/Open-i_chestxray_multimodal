```mermaid
gantt
    title Planning du projet - Beyond Pixel
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m

    section Phase initiale
    Phase collective ML/DL & ressources        :done, p1, 2026-02-18, 2026-03-01

    section Analyse
    Analyse exploratoire Open-I Chest X-ray    :p2, 2026-03-16, 2026-03-29

    section Développements parallèles
    Construction pipeline ML                   :p3, 2026-03-30, 2026-04-25
    Modèles vision (classification)            :p4, 2026-03-30, 2026-04-25
    Extraction NLP des rapports                :p5, 2026-03-30, 2026-04-25
    Aide & résilience                          :p6, 2026-03-30, 2026-04-25

    section Intégration
    Finalisation + fusion multimodale          :p7, after p3, 19d

    section Évaluation
    Évaluation & interprétation                :p8, after p7, 27d

    section Présentation
    Présentation finale                        :p9, after p8, 10d
```