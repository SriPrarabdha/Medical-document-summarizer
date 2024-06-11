import streamlit as st
from summary_chains.stuff_summarization import stuff_summarize
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
from dotenv import load_dotenv

load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY1')
os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY2')

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0)

def get_data(intput_prompt, filename):
    prompt = f"You are given a sentence your task is to just return the name of the doctor or physician present in that sentence and if you find no doctor/physician then in that case return None . SENTENCE: {input_prompt}"
    result = model.invoke(prompt).content

    if result == "None":
        with open(f'/teamspace/studios/this_studio/Medical-document-summarizer/Extracted_Data/{filename}.json', 'r') as file:
            data = json.load(file)

        text = ""
        # print(data)
        for key in data:
            # print(key)
            text= text + " " + data[key]
            # print(data[key])

    else:
        pass
def get_output(filename, chunk_size, method):
    

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=80
    )
    texts_split = text_splitter.split_text(text)

    docs = [Document(page_content=text) for text in texts_split]

    if method == "Stuff":
        return stuff_summarize(docs)
    elif method == "Map_Reduce":
        return map_reduce_summary(docs)
    elif method == "Refine":
        return stuff_summarize(docs)
    elif method == "Raptor":
        return stuff_summarize(docs)

st.title("LLM Assistant")

input_prompt = st.text_area("Enter your prompt")

data = st.selectbox("Choose a pdf", ["Medical_Record_File_1", "Medical_Record_File_2", "Medical_Record_File_3"])
method = st.selectbox("Choose a method", ["Stuff", "Map_Reduce", "Refine", "Raptor"])
chunk_size = st.slider("Chunk Size", 1000, 5000, value=4000, step=100)

if st.button("Submit"):
    output_text = get_output(input_prompt, method)
    st.markdown(output_text)