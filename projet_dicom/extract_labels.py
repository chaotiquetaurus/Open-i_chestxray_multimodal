import pydicom
from pathlib import Path
import pandas as pd

def extract_dicom_labels():
    """Extrait les métadonnées DICOM pour la classification chest"""
    dataset_path = Path(".") / "fragment_data_set"
    
    labels_info = []
    
    for dcm_file in sorted(dataset_path.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(dcm_file)
            patient_id = dcm_file.parent.parent.name
            
            # Extraction des champs importants
            info = {
                "patient_id": patient_id,
                "filename": dcm_file.name,
                "series_description": str(getattr(ds, "SeriesDescription", "N/A")),
                "study_description": str(getattr(ds, "StudyDescription", "N/A")),
                "series_uid": str(getattr(ds, "SeriesInstanceUID", "N/A")),
                "modality": str(getattr(ds, "Modality", "N/A")),
                "patient_age": str(getattr(ds, "PatientAge", "N/A")),
                "patient_sex": str(getattr(ds, "PatientSex", "N/A")),
                "rows": int(getattr(ds, "Rows", 0)),
                "columns": int(getattr(ds, "Columns", 0)),
            }
            
            labels_info.append(info)
            
        except Exception as e:
            print(f"Erreur lecture {dcm_file}: {e}")
    
    if not labels_info:
        print("❌ Aucun fichier DICOM trouvé!")
        return
    
    # Créer DataFrame
    df = pd.DataFrame(labels_info)
    
    # Affichage des labels uniques
    print("\n=== MÉTADONNÉES DICOM EXTRAITES ===\n")
    print(df.to_string(index=False))
    
    print("\n=== DESCRIPTIONS UNIQUES ===")
    print("\nSeriesDescription:")
    print(df['series_description'].unique())
    
    print("\nStudyDescription:")
    print(df['study_description'].unique())
    
    # Sauvegarder en CSV
    output_path = Path("dicom_labels.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Labels sauvegardés dans: {output_path}")
    
    return df

if __name__ == "__main__":
    df = extract_dicom_labels()
