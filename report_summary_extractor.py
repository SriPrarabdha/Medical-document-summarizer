from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from report_wise_json_segregator import report_wise_json_data_extraction

os.environ["GOOGLE_API_KEY"] = "AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0.8)

prompt_template = """Given this medical report of a patient give me a concise summary for all this document in multiple points where each points follows a template like
*Type of form/Diagnosis/examination or pharmacy/medical presciption reports/follow-up evaluations for a patient* dated *date of that form/Diagnosis/examination or pharmacy/medical presciption reports/follow-up evaluations* by *name of physician /attorney if present* - Impression : *any impression/inference from form/Diagnosis/examination or pharmacy/medical presciption reports/follow-up evaluations*
    
Context:
"{text}"
"""
prompt = PromptTemplate.from_template(prompt_template)

llm_chain = LLMChain(llm=model, prompt=prompt)

# filename = 'Medical_Record_File_1'

def report_summary_extractor(filename):
    report_extracted_data = report_wise_json_data_extraction(filename)

    complete_sum = ""
    for i in range(1,len(report_extracted_data)+1):
        complete_sum += llm_chain.invoke(report_extracted_data[i])['text']
        print('{} / {} Completed || {}%'.format(i, len(report_extracted_data),(i*100)/len(report_extracted_data)))

    with open('Report_Wise_Summary/{}.txt'.format(filename), 'w') as f:
        f.write(complete_sum)
        
    return complete_sum
