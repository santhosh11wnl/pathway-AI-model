import shutil
import os
import cv2
import torch
import numpy as np
from retinanet.dataloader import Resizer

def classify_image(image, model):
    model.eval()
    sample = {'img': image, 'annot': np.array([[0., 0., 1., 1., 0.]])}
    resize_img = Resizer()
    out = resize_img(sample)
    input_img = out['img'].numpy()
    input_img = np.expand_dims(input_img, 0)
    input_img = np.transpose(input_img, (0, 3, 1, 2))
    
    with torch.no_grad():
        image_tensor = torch.from_numpy(input_img)
        image_tensor = image_tensor.cpu()  # Use CPU for processing
        scores, classification = model(image_tensor.float())
    return scores, classification

def process_images(image_folder, model_path, output_folder):
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load model to CPU
    model = model.cpu()  # Ensure model is using CPU

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(image_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, img_name)
            image = cv2.imread(image_path)
            scores, classification = classify_image(image, model)
            if classification[0] == 'pathway':
                shutil.copy(image_path, output_folder)

# if __name__ == '__main__':
#     image_folder = r'C:\Users\saimi\deepproject\extract_img\21715183'
#     model_path = r'C:\Users\saimi\deepproject\RetinaNet(pathway_identifier)\RetinaNet(pathway_identifier)\model\csv_retinanet_2.pt'
#     output_folder = r'C:\Users\saimi\deepproject\pathway_images'
    
#     process_images(image_folder, model_path, output_folder)
