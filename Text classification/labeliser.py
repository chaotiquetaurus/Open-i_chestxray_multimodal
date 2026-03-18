import os
import re
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# ── Mapping complet basé sur le vocabulaire officiel NLM Open-i ─────────────
MESH_MAP = {
    "Atelectasis": [
        "pulmonary atelectasis", "atelectasis", "atelectases",
        "collapsed lung", "collapse; pulmonary", "lobar collapse"
    ],
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "enlarged heart",
        "enlargement heart", "heart size increased", "megalocardia"
    ],
    "Effusion": [
        "pleural effusion", "fluid in the chest", "pleural cavity effusion",
        "pericardial effusion", "effusion; pericardium", "pericardial fluid"
    ],
    "Pneumonia": [
        "pneumonia", "inflammation lung", "pulmonary inflammation",
        "pneumoniae", "pneumonitides", "pneumonitis"
    ],
    "Pneumothorax": [
        "pneumothorax", "free air in the chest outside the lung",
        "pleural air collection", "hemopneumothorax", "hydropneumothorax"
    ],
    "Edema": [
        "pulmonary edema", "edema lung", "wet lungs", "subpleural edema",
        "pulmonary congestion", "pulmonary vascular congestion",
        "vascular redistribution", "central vascular congestion"
    ],
    "Emphysema": [
        "emphysema", "pulmonary emphysema", "lung emphysema",
        "centrilobular emphysema", "panacinar emphysema",
        "bullous emphysema", "abnormal collection of air in tissues"
    ],
    "Fibrosis": [
        "fibrosis", "pulmonary fibrosis", "fibroses", "fibroplasia",
        "fibrous thickening", "cirrhosis of lung", "fibrosis of lung",
        "idiopathic pulmonary fibrosis"
    ],
    "Infiltration": [
        "infiltrate"
    ],
    "Mass": [
        "mass"
    ],
    "Nodule": [
        "nodule", "nodulus"
    ],
    "Hernia": [
        "hernia, diaphragmatic", "diaphragmatic hernia",
        "hernia, hiatal", "hiatal hernia", "esophageal hiatal hernia"
    ],
    "Fracture": [
        "fractures, bone", "fracture", "broken", "bone fractures"
    ],
    "Pleural_Thickening": [
        "thickening", "thickened", "cuffing"
    ],
    "Opacity": [
        "opacity", "decreased translucency", "increased density",
        "opaque", "haziness", "airspace disease",
        "air space lung disease", "alveolar lung disease"
    ],
    "Consolidation": [
        "consolidation"
    ],
    "Granuloma": [
        "granuloma", "calcified granuloma", "calcifying granuloma",
        "granulomatous calcifications", "granulomatous disease"
    ],
    "Calcinosis": [
        "calcinosis", "calcification", "calcified", "macrocalcification",
        "calcium deposits"
    ],
    "Scoliosis": [
        "scoliosis", "scoliotic", "spine curvature",
        "lateral curvature of the spine"
    ],
    "Atherosclerosis": [
        "atherosclerosis", "atherosclerotic vascular disease", "atheromatosis"
    ],
}


# ── Parsing XML → une ligne par image ────────────────────────────────────────
def parse_xml_dir(xml_dir):
    samples = []
    for filepath in glob.glob(os.path.join(xml_dir, "*.xml")):
        tree = ET.parse(filepath)
        root = tree.getroot()

        # MeSH labels (major + automatic)
        labels_major = [n.text.lower() for n in root.findall(".//MeSH/major") if n.text]
        labels_auto  = [n.text.lower() for n in root.findall(".//MeSH/automatic") if n.text]
        all_labels   = "|".join(np.unique(labels_major + labels_auto))

        # One row per parentImage (= per image in the CSV)
        for img in root.findall(".//parentImage"):
            img_id = img.attrib.get("id", "")
            if img_id:
                samples.append({
                    "image_id": img_id + ".png",
                    "mesh_all": all_labels,
                })

    return pd.DataFrame(samples)


# ── Labellisation ────────────────────────────────────────────────────────────
def label_from_mesh(mesh_df):
    labels = {}
    for pathology, keywords in MESH_MAP.items():
        pattern = "|".join([re.escape(k) for k in keywords])
        mask = mesh_df["mesh_all"].str.contains(pattern, case=False, na=False, regex=True)
        labels[pathology] = mask.values.astype(int)

    labels_df = pd.DataFrame(labels, index=mesh_df.index)
    labels_df["Normal"] = (labels_df[list(MESH_MAP.keys())].sum(axis=1) == 0).astype(int)
    labels_df.insert(0, "image_id", mesh_df["image_id"].values)
    return labels_df


# ── Pipeline ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    XML_DIR  = os.path.join(BASE_DIR, "NLMCXR_reports", "ecgen-radiology")
    CSV_PATH = os.path.join(BASE_DIR, "Datasets", "new_dataset.csv")
    OUT_PATH = os.path.join(BASE_DIR, "dataset_labeled.csv")

    print("Parsing XMLs...")
    mesh_df = parse_xml_dir(XML_DIR)
    print(f"  {len(mesh_df)} images parsées depuis les XML")

    print("Labellisation...")
    labels_df = label_from_mesh(mesh_df)

    print("Merge avec le dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} lignes dans le CSV")

    df_final = df.merge(labels_df, on="image_id", how="left")

    LABEL_COLS = list(MESH_MAP.keys()) + ["Normal"]
    # Images sans XML correspondant → 0
    df_final[LABEL_COLS] = df_final[LABEL_COLS].fillna(0).astype(int)

    print(f"\nMatched: {df_final[LABEL_COLS[0]].notna().sum()} / {len(df_final)}")
    print("\nLabel distribution:")
    print(df_final[LABEL_COLS].sum().sort_values(ascending=False).to_string())
    print(f"\nShape final: {df_final.shape}")

    df_final.to_csv(OUT_PATH, index=False)
    print(f"\nSauvegardé : {OUT_PATH}")
