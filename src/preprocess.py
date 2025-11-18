import os
import pandas as pd
from PIL import Image
from tqdm import tqdm # A library for progress bars! Run !pip install tqdm in your notebook if you don't have it.

# -----------------------------------------------------------------------------
# --- CONFIGURATION ---
# -----------------------------------------------------------------------------

# >> TODO: Update these paths to match your system <<
# Base directory where you downloaded all the raw datasets
RAW_DATA_BASE_DIR = 'C:/Users/nipun/Project_Datasets/'

# Directory where the final, clean images will be saved
PROCESSED_DATA_DIR = 'processed_data/'

# Standard image size we will resize everything to
IMG_SIZE = (224, 224)

# --- Define paths to each raw dataset ---
NIH_DIR = os.path.join(RAW_DATA_BASE_DIR, 'NIH_ChestX-ray14')
RSNA_DIR = os.path.join(RAW_DATA_BASE_DIR, 'RSNA_Pneumonia')
COVID_KAGGLE_DIR = os.path.join(RAW_DATA_BASE_DIR, 'COVID-19_Radiography_Database')


# -----------------------------------------------------------------------------
# --- HELPER FUNCTIONS ---
# -----------------------------------------------------------------------------
def find_nih_image_path(base_dir, image_filename):
    """
    Searches through all 'images_001' to 'images_012' folders,
    and then inside the nested subfolder, to find the image.
    """
    for i in range(1, 13):
        folder_name = f"images_{i:03}"
        numbered_folder_path = os.path.join(base_dir, folder_name)

        if os.path.isdir(numbered_folder_path):
            subfolders = os.listdir(numbered_folder_path)
            if len(subfolders) > 0:
                image_subfolder_name = subfolders[0]
                potential_path = os.path.join(numbered_folder_path, image_subfolder_name, image_filename)
                if os.path.exists(potential_path):
                    return potential_path
    return None

# -----------------------------------------------------------------------------
# --- DATASET PROCESSING FUNCTIONS ---
# -----------------------------------------------------------------------------

def process_nih_dataset():
    """
    Processes the NIH ChestX-ray14 dataset.
    It extracts images for 'Pneumonia', 'Tuberculosis', and 'Normal' (No Finding).
    """
    print("Processing NIH Dataset...")
    
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'Pneumonia'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'Tuberculosis'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'Normal'), exist_ok=True)
    
    df = pd.read_csv(os.path.join(NIH_DIR, 'Data_Entry_2017.csv'))
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_filename = row['Image Index']
        labels = row['Finding Labels']
        
        target_class = None
        if 'Pneumonia' in labels:
            target_class = 'Pneumonia'
        elif 'Tuberculosis' in labels:
            target_class = 'Tuberculosis'
        elif 'No Finding' in labels:
            target_class = 'Normal'
        
        if target_class:
            # --- THIS IS THE MODIFIED LOGIC ---
            dest_path = os.path.join(PROCESSED_DATA_DIR, target_class, f"nih_{image_filename}")
            
            # Only process the image if it hasn't been saved already
            if not os.path.exists(dest_path):
                src_path = find_nih_image_path(NIH_DIR, image_filename)
                
                if src_path:
                    try:
                        img = Image.open(src_path).convert('RGB').resize(IMG_SIZE)
                        img.save(dest_path)
                    except Exception as e:
                        print(f"Warning: Could not process file {src_path}. Error: {e}")

def process_rsna_dataset():
    """
    Processes the RSNA Pneumonia dataset.
    It extracts images for 'Pneumonia' and 'Normal'.
    """
    print("\nProcessing RSNA Dataset...")
    
    import pydicom
    
    df = pd.read_csv(os.path.join(RSNA_DIR, 'stage_2_train_labels.csv'))
    df_unique = df.drop_duplicates('patientId')
    
    for index, row in tqdm(df_unique.iterrows(), total=df_unique.shape[0]):
        patient_id = row['patientId']
        label = row['Target']
        
        target_class = 'Pneumonia' if label == 1 else 'Normal'
        
        # --- THIS IS THE MODIFIED LOGIC ---
        dest_path = os.path.join(PROCESSED_DATA_DIR, target_class, f"rsna_{patient_id}.png")

        # Only process the image if it hasn't been saved already
        if not os.path.exists(dest_path):
            src_path = os.path.join(RSNA_DIR, 'stage_2_train_images', f"{patient_id}.dcm")
            
            if os.path.exists(src_path):
                try:
                    dcm_data = pydicom.dcmread(src_path)
                    image_array = dcm_data.pixel_array
                    img = Image.fromarray(image_array).convert('RGB').resize(IMG_SIZE)
                    img.save(dest_path)
                except Exception as e:
                    print(f"Warning: Could not process file {src_path}. Error: {e}")
def process_covid_kaggle_dataset():
    """
    Processes the Kaggle COVID-19 Radiography Database.
    Extracts images for 'COVID', 'Normal', and 'Pneumonia' (from Viral Pneumonia).
    """
    print("\nProcessing Kaggle COVID-19 Dataset...")
    
    # --- THIS IS THE FIX ---
    # Ensure the destination folder for COVID images exists before we start
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'COVID'), exist_ok=True)
    # ----------------------

    source_classes = ['COVID', 'Normal', 'Viral Pneumonia']
    
    for source_class in source_classes:
        if source_class == 'Viral Pneumonia':
            target_class = 'Pneumonia'
        else:
            target_class = source_class

        source_folder = os.path.join(COVID_KAGGLE_DIR, source_class, 'images')
        
        if os.path.exists(source_folder):
            image_files = os.listdir(source_folder)
            
            for image_filename in tqdm(image_files, desc=f'Processing {source_class}'):
                
                dest_path = os.path.join(PROCESSED_DATA_DIR, target_class, f"covid_{image_filename}")
                
                if not os.path.exists(dest_path):
                    src_path = os.path.join(source_folder, image_filename)
                    try:
                        img = Image.open(src_path).convert('RGB').resize(IMG_SIZE)
                        img.save(dest_path)
                    except Exception as e:
                        # This will now catch real errors, not the FileNotFoundError
                        print(f"Warning: Could not process file {src_path}. Error: {e}")
def process_tb_dataset():
    """
    Processes the Kaggle Tuberculosis (TB) Chest X-ray Database.
    """
    print("\nProcessing Kaggle Tuberculosis Dataset...")
    
    # Use the exact folder name after unzipping
    TB_DIR = os.path.join(RAW_DATA_BASE_DIR, 'Tuberculosis-Chest-X-ray-Database') 
    
    source_classes = ['Tuberculosis', 'Normal']
    
    for source_class in source_classes:
        if source_class == 'Tuberculosis':
            source_folder = os.path.join(TB_DIR, source_class)
            
            # --- DEBUGGING PRINT STATEMENTS ---
            print(f"DEBUG: Checking for folder at this exact path: '{source_folder}'")
            if os.path.exists(source_folder):
                print(f"DEBUG: Success! Folder found.")
                # --- END DEBUGGING ---
                
                image_files = os.listdir(source_folder)
                
                for image_filename in tqdm(image_files, desc=f'Processing {source_class}'):
                    dest_path = os.path.join(PROCESSED_DATA_DIR, 'Tuberculosis', f"tb_{image_filename}")
                    
                    if not os.path.exists(dest_path):
                        src_path = os.path.join(source_folder, image_filename)
                        try:
                            img = Image.open(src_path).convert('RGB').resize(IMG_SIZE)
                            img.save(dest_path)
                        except Exception as e:
                            print(f"Warning: Could not process file {src_path}. Error: {e}")
            else:
                # --- DEBUGGING PRINT STATEMENTS ---
                print(f"DEBUG: FAILED! The folder was not found at the path above.")
                print(f"DEBUG: Please check that the folder name in the script matches your computer exactly.")
                # --- END DEBUGGING ---

# -----------------------------------------------------------------------------
# --- MAIN EXECUTION BLOCK ---
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("--- Starting Dataset Preprocessing ---")
    
    # We will call each function here
    process_nih_dataset()
    process_rsna_dataset()
    process_covid_kaggle_dataset()
    process_tb_dataset()
    print("--- Preprocessing Complete ---")