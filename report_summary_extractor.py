from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from report_wise_json_segregator import report_wise_json_data_extraction
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY1')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0.7)

prompt_template = """Given this medical report of a patient give me a concise summary for all this document in multiple points where each points follows a template like
Type of form/Diagnosis/examination or pharmacy/medical prescription reports/follow-up evaluations for a patient dated date of that form/Diagnosis/examination or pharmacy/medical prescription reports/follow-up evaluations by name of physician /attorney if present - Impression : any impression/inference from form/Diagnosis/examination or pharmacy/medical prescription reports/follow-up evaluations
Strictly follow the given template and don not anything beside the short yet comprehensive summary
Context:
"{text}"
"""

prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(llm=model, prompt=prompt)

# Prompt for checking the relevance of the summary
checker_prompt_template = """ Analyze the context, and return a threshold score based on the criticality of the report and impression returned with respect to the health of a person.
Strictly just return the threshold score only, and the threshold score should be an integer.
Context:
"{text}"
"""

checker_prompt = PromptTemplate.from_template(checker_prompt_template)
checker_llm_chain = LLMChain(llm=model, prompt=checker_prompt)

def report_summary_extractor(filename):
    report_extracted_data = report_wise_json_data_extraction(filename)
    complete_sum = ""
    for i in range(1, len(report_extracted_data) + 1):
        partial_sum = llm_chain.invoke(report_extracted_data[i][2])['text']

        llm_threshold = checker_llm_chain.invoke(partial_sum)['text']

        if int(llm_threshold)>=7:
            complete_sum += "\n\n || REFERENCES PAGE NOS: {}, {} ||".format(report_extracted_data[i][0],report_extracted_data[i][1])
            complete_sum += partial_sum
        
        print('{} / {} Completed || {}% || Validation: {}'.format(i, len(report_extracted_data),(i*100)/len(report_extracted_data), int(llm_threshold)>= 7))
        yield complete_sum
    with open('Report_Wise_Summary/{}.txt'.format(filename), 'w') as f:
        f.write(complete_sum)
