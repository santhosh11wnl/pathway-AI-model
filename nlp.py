import os
import requests
import json
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import PyPDF2
from dotenv import load_dotenv
import nltk
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

pdf_path = r"C:\Users\saimi\deepproject\pdf files\21715183.pdf"  

pdf_text = extract_text_from_pdf(pdf_path)

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

processed_text = preprocess(pdf_text)

dictionary = corpora.Dictionary([processed_text])
corpus = [dictionary.doc2bow(processed_text)]

lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

topics = lda_model.print_topics(num_words=5)

url = "https://api.openai.com/v1/engines/davinci-codex/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "prompt": f"Summarize the following document:\n{' '.join(processed_text)}",
    "max_tokens": 200, 
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()
pdf_summary = result.get('text', '').strip()

ner_tags = ne_chunk(pos_tag(word_tokenize(pdf_text)))

output_data = {
    "LDA_Topics": topics,
    "PDF_Summary": pdf_summary,
    "NER_Tags": str(ner_tags), 
}

with open("output_k.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

