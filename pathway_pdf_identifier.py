import shutil
import os
import numpy as np
import fitz  # PyMuPDF
import cv2
import torch
from retinanet.dataloader import Resizer

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
            images.append(image)
    doc.close()
    return images

def image_identifier(image, model):
    sample = {'img': image, 'annot': np.array([[0., 0., 1., 1., 0.]])}
    resize_img = Resizer()
    out = resize_img(sample)
    input_img = out['img'].numpy()
    input_img = np.expand_dims(input_img, 0)
    input_img = np.transpose(input_img, (0, 3, 1, 2))

    with torch.no_grad():
        image = torch.from_numpy(input_img)
        scores, classification = model(image.float())
        return classification[0] == 'pathway'

def process_pdfs(pdf_dir, model_path, output_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.training = False
    model.eval()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for pdf_name in os.listdir(pdf_dir):
        if pdf_name.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_name)
            images = extract_images_from_pdf(pdf_path)
            contains_pathway = any(image_identifier(image, model) for image in images)
            if contains_pathway:
                shutil.copy(pdf_path, output_path)

if __name__ == '__main__':
    pdf_dir = r'C:\Users\saimi\deepproject\pdf files'
    model_path = r'C:\Users\saimi\deepproject\RetinaNet(pathway_identifier)\RetinaNet(pathway_identifier)\model\csv_retinanet_2.pt'
    output_path = r'C:\Users\saimi\deepproject\paper'
    process_pdfs(pdf_dir, model_path, output_path)
