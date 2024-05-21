import asyncio
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

get_or_create_eventloop()

import streamlit as st
from summary_chains.stuff_summarization import stuff_summarize
from summary_chains.refine_summarization import refine_summarize
from summary_chains.raptor_summarization import raptor_summarize
from summary_chains.map_reduce_summarization import map_reduce_summary

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage
import google.generativeai as genai
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("LLM Assistant")
input_prompt = st.text_area("Enter your prompt")

data = st.selectbox("Choose a pdf", ["Medical_Record_File_1", "Medical_Record_File_2", "Medical_Record_File_3"])
method = st.selectbox("Choose a method", ["Stuff", "Map_Reduce", "Refine", "Raptor"])
chunk_size = st.slider("Chunk Size", 1000, 5000, value=4000, step=100)

# os.environ["GOOGLE_API_KEY"] = "AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDx7cfKeqr0YK0TE8767lnMz6G5NmeXJBI"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0)

def get_output(filename, chunk_size, method):
    with open(f'Extracted_Data/{filename}.json', 'r') as file:
        data = json.load(file)

    text = ""
    # print(data)
    for key in data:
        # print(key)
        text= text + " " + data[key]
        # print(data[key])

    chunk_size_tok = chunk_size
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=80
    )
    texts_split = text_splitter.split_text(text)

    docs = [Document(page_content=text) for text in texts_split]

    if method == "Stuff":
        return stuff_summarize(docs, model)
    elif method == "Map_Reduce":
        return map_reduce_summary(docs, model)
    elif method == "Refine":
        return refine_summarize(docs, model)
    elif method == "Raptor":
        return raptor_summarize(docs)

if st.button("Submit"):
    output_text = get_output(data,chunk_size,method)
    st.markdown(output_text)