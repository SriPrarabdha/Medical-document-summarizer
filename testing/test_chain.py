from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
# from langchain.chains.summarize import load_summarize_chain
# from langchain.prompts import PromptTemplate

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage
import google.generativeai as genai
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY1')

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

embd = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0.8)

# Doc texts split
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('Extracted_Data/Medical_Record_File_1.json', 'r') as file:
    data = json.load(file)

text = ""
# print(data)
for key in data:
    # print(key)
    text= text + " " + data[key]
    # print(data[key])

chunk_size_tok = 2000
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_tok, chunk_overlap=0
)
texts_split = text_splitter.split_text(text)
print(texts_split)


template = """Here is a medical record file of some patient your task is to just create in detail
                summaries that has every thing related to the given context. Just provide me with summary do not add anything extra from your side.
    
    Context:
    "{text}"
    """
# map_prompt_template = PromptTemplate(template=template, input_variables=["text"])
map_prompt_template = PromptTemplate.from_template(template)
llm_chain = LLMChain(llm=model, prompt=map_prompt_template)

stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# map_chain = load_summarize_chain(llm=model,
#                             chain_type="stuff",
#                             prompt=map_prompt_template)
# prompt = ChatPromptTemplate.from_template(template)
# chain = prompt | model | StrOutputParser()

# Format text within each cluster for summarization
summaries = []

for chunks in texts_split:
    response = stuff_chain.invoke([chunks])
    print(response)
