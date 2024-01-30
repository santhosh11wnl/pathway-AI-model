from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Function as the greatest route gene extractor, accurately extracting each gene link based on the relation provided in the image, here genes are enclosed in circles and relations are by arrows and T-bars. the relations are two types inhibition and activation .Here inhibition is represented by T-bar and dashed T-bar represents Indirect Inhibition and activation is by arrow symbols and dashed arrow symbols for indirect activation. The arrow line to arrowhead represents the direction of the relation, and arrow one is like T-bar. Please remove every gene relationship from the image, avoid confusing them with one another, and refer to the relationships as gene1 (startor) and relationship as gene2 (receptor)"},
                {
                    "type": "path",
                    "path": r"C:\Users\saimi\deepproject\PMC3932947__2040-2392-5-9-7.jpg"  
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)