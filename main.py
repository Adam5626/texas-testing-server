from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import urllib.request
import fitz
import numpy as np
import tensorflow_hub as hub
from lcserve import serving
from sklearn.neighbors import NearestNeighbors
import requests

recommender = None

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i : (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings
    

def load_recommender(path, start_page=1):
    global recommender
    if recommender is None:
        recommender = SemanticSearch()
        print("Done")

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'



def generate_answer(question):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += (
        "Give a summarised answer from the information provided, if the answer to the question does not seem to be found in the information provided, simple say Text not found in document. Write answer in a paragraph with no line breaks."
    )

    prompt += f"Query: {question} \nAnswer: "
    answer = requests.get("https://sf98l4weqe.execute-api.us-west-2.amazonaws.com/llama2-lambda", {"query": prompt})
    print(len(answer.text.split('Answer: ')))
    htmlContent = answer.text.split('Answer: ')[1].replace('\n', "<br />")
    return htmlContent



@serving
def ask_url(url: str, question: str):
    download_pdf(url, 'corpus.pdf')
    load_recommender('corpus.pdf')
    return generate_answer(question)

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Enable CORS for all routes and origins



@app.route('/embeddings', methods=['POST'])
def process_data():
        data = request.get_json()
        link = data["link"]
        download_pdf(link, 'corpus.pdf')
        load_recommender('corpus.pdf')
        return jsonify({"message": "Successfully made embeddings"})

@app.route('/results', methods=['POST'])
def get_results():
    data = request.get_json()
    question = data["question"]
    answer = generate_answer(question)

    return answer

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
