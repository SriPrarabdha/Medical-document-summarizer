import asyncio
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
import google.generativeai as genai
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from report_summary_extractor import report_summary_extractor
from report_wise_json_segregator import report_wise_json_data_extraction

import time
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

get_or_create_eventloop()

loader = st.empty()

def simulate_long_process():
    # Display the loader
    with loader.container():
        # Load the circular loader image
        loader_image = Image.open('loader.gif')
        
        # Display the circular loader image
        st.image(loader_image, caption='Loading...', use_column_width=True)
        
        # Loop for 10 seconds
        for i in range(100):
            time.sleep(0.1)

st.title("LLM Assistant")
input_prompt = st.text_area("Enter your prompt")

data = st.selectbox("Choose a pdf", ["Medical_Record_File_1", "Medical_Record_File_2", "Medical_Record_File_3"])
method = st.selectbox("Choose a method", ["Stuff", "Map_Reduce", "Refine", "Raptor", "Summary"])
chunk_size = st.slider("Chunk Size", 1000, 5000, value=4000, step=100)

os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY1')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0)

if data:
    # Simulate a long-running process
    simulate_long_process()

prompt_template = """Context:
"{text}"

Given this medical report of a patient give me a concise summary for all this document in 
multiple points where each points follows a template like
{Type of form/Diagnosis/examination or pharmacy/medical prescription reports/follow-up evaluations for a patient}
dated {date of that form/Diagnosis/examination or pharmacy/medical prescription reports/follow-up evaluations} 
by {name of physician /attorney if present} - Impression : {any impression/inference from form/Diagnosis/examination or pharmacy/medical prescription reports/follow-up evaluations}.
Strictly follow this format
"""
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(llm=model, prompt=prompt)

def get_output(filename, chunk_size, method):
    if method == "Summary":
        return report_summary_extractor(filename)
    else:
        with open(f'Extracted_Data/{filename}.json', 'r') as file:
            data = json.load(file)
        text = ""
        for key in data:
            text += " " + data[key]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=80
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
    output_placeholder = st.empty()
    if method == "Summary":
        for partial_summary in get_output(data, chunk_size, method):
            output_placeholder.markdown(partial_summary)
    else:
        output_text = get_output(data, chunk_size, method)
        output_placeholder.markdown(output_text)
