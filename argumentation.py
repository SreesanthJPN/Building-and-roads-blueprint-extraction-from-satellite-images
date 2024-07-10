import os
import numpy as np
from PIL import Image

def load_and_preprocess_data(image_dir, mask_dir, img_size=(512, 512)):
    images = []
    masks = []
    
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        try:
          
            img = Image.open(img_path)
            img = img.convert('RGB')  
            img = img.resize(img_size)
            img = np.array(img)
            
            mask = Image.open(mask_path)
            mask = mask.convert('RGB')  
            mask = mask.resize(img_size)
            mask = np.array(mask)
            
            binary_mask = np.all(mask == (0, 0, 255), axis=-1).astype(np.uint8)
            
            images.append(img)
            masks.append(binary_mask)
        
        except Exception as e:
            print(f"Error processing {img_file} or {mask_file}: {e}")
    
    return np.array(images), np.expand_dims(np.array(masks), axis=-1)