import json
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY1')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

embd = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
from pinecone import Pinecone

def combining_json_data(filename):
    """
    Combining all the json data into one json file
    """
    with open(f'Extracted_Data/{filename}.json', 'r') as file:
        data = json.load(file)
    
    all_content = " ".join(data.values())
    with open('Report_Content/{}.txt'.format(filename), 'w') as f:
        f.write(all_content)

    text_loader = TextLoader('{}.txt'.format(filename))
    docs = text_loader.load()

    return docs

def text_splitting_file(docs):
    """
    Splitting the text into pages
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=300
    )

    texts = text_splitter.split_documents(docs)

    return texts

def create_embeddings(texts):
    embeddings_list = []
    for text in texts:
        # print(text)
        res = embd.embed_query(text.page_content)
        embeddings_list.append(res)
    return embeddings_list

def upsert_embeddings_to_pinecone(index, embeddings):
    index.upsert(vectors=[(str(id), embedding) for id, embedding in enumerate(embeddings)])
    
def push_data_to_pinecone(filename, index_name):
    try:
        docs = combining_json_data(filename)
        texts = text_splitting_file(docs)
        embeddings = create_embeddings(texts)
        pc = Pinecone(api_key="d9bf6f8a-6525-4821-9c64-c4a92083ee73")
        index = pc.Index(index_name)

        # Upserting_data
        upsert_embeddings_to_pinecone(index, embeddings)
    except Exception as e:
        print("Error Occured: ",e.message)