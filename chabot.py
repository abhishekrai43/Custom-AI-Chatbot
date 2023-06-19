from flask import Flask, render_template, request
from flask import Markup
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import os
import requests
from bs4 import BeautifulSoup
import re
app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = ''

def sanitize_file_name(file_name):
    # Remove any special characters from the file name
    sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", file_name)
    return sanitized_name

def clean_html_tags(text):
    # Remove HTML tags using regular expressions
    clean_text = re.sub(r"<.*?>", "", text)
    return clean_text

def scrape_url(url, directory_path):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        cleaned_text = clean_html_tags(soup.get_text())
        sanitized_url = sanitize_file_name(url)
        filename = os.path.join(directory_path, f"{sanitized_url}.txt")
        with open(filename, "w", encoding="utf-8") as text_file:
            text_file.write(cleaned_text)
        return filename
    except Exception as e:
        print(f"Failed to scrape URL: {url}")
        print(e)
        return None

def process_urls(directory_path):
    urls_done = []

    with open(os.path.join(directory_path, "urls.txt"), "r",encoding='utf-8') as file:
        urls = file.readlines()

    for url in urls:
        url = url.strip()
        if not url.endswith("-done"):
            text_file = scrape_url(url, directory_path)
            if text_file:
                urls_done.append(f"{url}-done")

    with open(os.path.join(directory_path, "urls.txt"), "a") as file:
        for url_done in urls_done:
            file.write(url_done + "\n")

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 1024
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    process_urls(directory_path)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        input_text = request.form['input_text']
        response = chatbot(input_text)
        return render_template('chat.html', input_text=input_text, response=response)
    return render_template('chat.html')

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    response_text = response.response.replace("\n", "<br>")
    return Markup(response_text)

if __name__ == '__main__':
    index = construct_index("docs")
    app.run(host='0.0.0.0')
