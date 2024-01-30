import base64
import json
import os
import requests
import torch
from dotenv import load_dotenv
from openai import OpenAI
from get_text_image_from_pdfs import extract_information, Path 
from pathway_pdf_identifier import process_pdfs
from Pathway_image_identifier import process_images

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def select_image_from_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            return os.path.join(directory, filename)
    return None

def process_pdfs_images_and_chatgpt():
    # PDF Identification
    pdf_dir = r'C:\Users\saimi\deepproject\pdf files'
    model_path = r'C:\Users\saimi\deepproject\RetinaNet(pathway_identifier)\RetinaNet(pathway_identifier)\model\csv_retinanet_2.pt'
    output_path = r'C:\Users\saimi\deepproject\paper'
    process_pdfs(pdf_dir, model_path, output_path)

    # Extracting text and images from PDFs
    pdf_path = 'paper'
    pdf_file_path = Path(pdf_path)
    for pdf_file in pdf_file_path.glob("*.pdf"):
        pdf_name = os.path.split(pdf_file)[1].split('.')[0]
        extract_information(pdf_file)

    # Image Extraction and Identification
    image_folder = r'C:\Users\saimi\deepproject\extract_img\21715183'
    model_path = r'C:\Users\saimi\deepproject\RetinaNet(pathway_identifier)\RetinaNet(pathway_identifier)\model\csv_retinanet_2.pt'
    output_folder = r'C:\Users\saimi\deepproject\pathway_images'
    process_images(image_folder, model_path, output_folder)

    # ChatGPT API
    model_path = r'C:\Users\saimi\deepproject\RetinaNet(pathway_identifier)\RetinaNet(pathway_identifier)\model\csv_retinanet_2.pt'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model = model.cpu()

    image_directory = r"C:\Users\saimi\deepproject\pathway_images"
    image_path = select_image_from_directory(image_directory)

    if image_path:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        base64_image = encode_image(image_path)
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Function as the greatest route gene extractor, accurately extracting each gene link based on the relation provided in the image, here genes are enclosed in circles and relations are by arrows and T-bars. the relations are two types inhibition and activation .Here inhibition is represented by T-bar and dashed T-bar represents Indirect Inhibition and activation is by arrow symbols and dashed arrow symbols for indirect activation. The arrow line to arrowhead represents the direction of the relation, and arrow one is like T-bar. Please remove every gene relationship from the image, avoid confusing them with one another, and refer to the relationships as gene1 (startor) and relationship as gene2 (receptor)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()

            try:
                messages = response_json['choices'][0]['message']['content']
                if isinstance(messages, list):
                    messages = "\n".join(messages)
                print(messages)
                with open('output.json', 'w') as json_file:
                    json.dump(messages, json_file)
            except KeyError:
                print("Error: Key not found in response JSON.")
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
    else:
        print("No image found in the directory")

if __name__ == '__main__':
    process_pdfs_images_and_chatgpt()
