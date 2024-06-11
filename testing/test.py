import pdf2image
import io
import base64
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY1')

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

models = [m for m in genai.list_models()]
print(models)

model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0.8)

filepath = "./page_15.jpg"
with open(filepath, "rb") as image_file:
    image_data = image_file.read()
    encoded_string = base64.b64encode(image_data).decode("utf-8")

# print(encoded_string)

# text_message = {
#     "type": "text",
#     "text": "Your task is to return me each and every thing written in this image as a plain simple text and nothing else. In some lines we find 'With/Without' and there is circle on one of the text to be selected . If 'Without' is circled then just return 'Without' in your generated answer and drop 'With' . also like if you find 'Y/N' or 'Yes/No' and one is circled then just inclued the circled one in your answer. these are just examples in general if anywhere in a line you see multiple texts separated by '/' and one of the texts is circled then just return that one and drop others."
# }

text_message = {
    "type": "text",
    "text": "Your task is to return me each and every thing written in this image as a plain simple text and nothing else. In some lines we find that one of the word is circled while others are not like in Thoracic/ lumbar/ back pain , back pain is majorly circled , in Y/N on of them is circled , in With/Without one is circled while other is not your task is to only write the circled one and drop the other ones in your generated answer"
}

image_message = {
    "type": "image_url",
    "image_url" : {"url" : f"data:image/jpeg;base64,{encoded_string}"}
}

message = HumanMessage(content=[text_message , image_message])

answer = model.invoke([message])

print(answer.content)

"""
Boulevard Medical Care PC
6269 99 Street
Rego Park, NY 11374
EVALUATION DATE: 1-19-24
PATIENT NAME: Yan
DOB: 7-6-87 SEX: M DATE OF LOSS: 10-23
ELECTRODIAGNOSTIC HISTORY AND PHYSICAL 
EVALUATION
Patient was referred for an electrodiagnostic evaluation
Chart and available imaging records were reviewed
HPI: The above-named patient presented with and/or reported the following:
s/p low back mva in Nup w/ iga film sohm
Pems Luz NCS/EME she (c)(u)(c)
Chief Complaint(s):
• Cervical neck pain/without radiation down (right/left/bilateral)upper 
extremity(ies) to
• Without numbness/ tingling/ weakness/ pain/ stiffness in
(right/left/bilateral)
• Thoracic/ lumbar/ back pain/without radiation down (right/left/bilateral) lower
extremity(ies) to
• With/without numbness/ tingling/ weakness/ pain/ stiffness in 
(right/left/bilateral)
• Without bladder/ bowel problems/ fevers/ night sweats/ recent weight loss
Region Pain Quality Alleviating Exacerbating Setting/Timing Pain
Factor Factor Intensity
Head
Cervical 
Thoracic
Lumbar 
Sharp rest twisting rand 7/10
Sharp led rand ully Sharp 8/10

Past Medical History: Diabetes: N Pacemaker: Y/N
Hypertension: N Other: Y/N

ECN: 2024022002989 Received Date: 2024-02-19
"""