# Interpretability Analysis Results

## 1. Attention Heatmaps (Image Pooling & Text→Image Attention)

### Overall Conclusion
The attention maps show some variation between pathologies, indicating that the model does not always focus on exactly the same image regions. However, several high-attention areas appear near image borders or outside clinically relevant anatomy. Therefore, the attention mechanism provides only limited evidence of pathology-specific localization and should be interpreted with caution.

### Pneumothorax
The strongest attention appears near the left image border and partially outside the lung field. Although the model predicts Pneumothorax with moderate confidence, the highlighted region is not anatomically consistent with the expected location of a pneumothorax, raising concerns about border-related artifacts.

### Cardiomegaly
Attention is distributed across several regions, including lower thoracic areas. Some attention is located closer to relevant anatomy than in the Pneumothorax case, but the cardiac silhouette is not clearly emphasized as the dominant region.

### Effusion
The attention pattern differs noticeably from the frontal-view cases and follows the lateral projection. Some highlighted regions overlap with lower thoracic areas where pleural effusions are commonly observed, although the localization remains diffuse.

### Normal
Multiple hotspots are present despite the absence of pathology. The highlighted regions do not correspond to a specific anatomical structure and include border areas, suggesting that attention is not exclusively focused on disease-related features.

---

## 2. Grad-CAM on Individual Images

### Overall Conclusion
Grad-CAM produces more pathology-dependent patterns than raw attention maps and is generally a more direct indicator of which image features influence the prediction. However, the localization remains weak and inconsistent across samples, with some maps showing little signal and others highlighting regions that are not clearly related to the pathology.

### Pneumothorax
The Grad-CAM map is nearly uniform with very little localized activation. This suggests that the prediction is not strongly driven by a specific image region for this sample.

### Cardiomegaly
A small localized hotspot is visible, indicating that a limited region contributes to the prediction. However, the highlighted area is not clearly centered on the enlarged cardiac silhouette, making the explanation difficult to interpret clinically.

### Effusion
The heatmap shows widespread activation across the image rather than a single focused region. The resulting pattern appears noisy and lacks a clear anatomical target, reducing confidence in the localization.

### Normal
Several localized hotspots appear near image borders and peripheral regions. Since no pathology is present, these activations may reflect general image characteristics rather than disease-specific findings.

---

## 3. Averaged Norm-Corrected Grad-CAM (15 Samples per Pathology)

### Overall Conclusion
After averaging Grad-CAM maps from multiple positive cases, no pathology exhibits a strong and reproducible hotspot. The averaged signals remain weak and diffuse, suggesting that different samples do not consistently activate the same image region. This indicates limited evidence for a stable pathology-specific localization strategy.

### Pneumothorax
The averaged map contains only weak activations scattered across the grid. No dominant region emerges across patients, indicating low agreement in localization.

### Cardiomegaly
The averaged signal is very weak and broadly distributed. No clear concentration appears around the heart region, suggesting that the model does not consistently rely on the same cardiac area across cases.

### Effusion
A slightly stronger pattern is visible near the lower portion of the grid, which could be loosely consistent with pleural fluid accumulation. However, the overall signal remains weak and lacks a sharply defined hotspot.

### Normal
The strongest averaged activations occur near image borders and lower edge regions. These patterns are more suggestive of image-position effects than of meaningful anatomical localization.

---

# Final Takeaway

Across all three analyses, the model shows **some sensitivity to image content**, since the highlighted regions are not completely identical across pathologies. However, the explanations do not consistently align with expected disease locations, and several activations occur near image borders or background regions. Overall, the results suggest that while the image branch contributes to the predictions, the model does not appear to rely on a clear and reproducible anatomical region for each pathology. This may indicate reliance on distributed visual cues, dataset-specific patterns, or information provided by the text modality in the multimodal fusion process.