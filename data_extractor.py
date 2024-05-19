from pdf2image import convert_from_path
import json
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from langchain_core.messages import HumanMessage

# Prarabdh's API Key : AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU
# Krishan's API Key : AIzaSyDx7cfKeqr0YK0TE8767lnMz6G5NmeXJBI
os.environ["GOOGLE_API_KEY"] = "AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0.8)

file_to_extract = "Medical_Record_File_3"

# Store Pdf with convert_from_path function
images = convert_from_path('data\{}.pdf'.format(file_to_extract))

encoded_json = {}

for i in range(len(images)): 
    # Save pages as images from the pdf
    images[i].save('all_images/{}/page{}.jpg'.format(file_to_extract,str(i+1)), 'JPEG')


    with open('all_images/{}/page{}.jpg'.format(file_to_extract,str(i+1)), "rb") as image_file:
        image_data = image_file.read()
        encoded_string = base64.b64encode(image_data).decode("utf-8")

    text_message = {
    "type": "text",
    "text": "Your task is to return me each and every thing written in this image as a plain simple text and nothing else. In some lines we find 'With/Without' and there is circle on one of the text to be selected . If 'Without' is circled then just return 'Without' in your generated answer and drop 'With' . also like if you find 'Y/N' or 'Yes/No' and one is circled then just inclued the circled one in your answer. these are just examples in general if anywhere in a line you see multiple texts separated by '/' and one of the texts is circled then just return that one and drop others. Also the content given in tables are corresponding to that particular row and add them to that only. Remove unnecessary spaces and line breaks"
    }

    image_message = {
        "type": "image_url",
        "image_url" : {"url" : f"data:image/jpeg;base64,{encoded_string}"}
    }

    message = HumanMessage(content=[text_message , image_message])
    answer = model.invoke([message])

    encoded_json[i+1] = str(answer.content)
    print("{}/{} || {} percentage completed".format(i+1,len(images), ((i+1)*100)/len(images)))

save_file = open("Extracted_Data/{}.json".format(file_to_extract), "w")
json.dump(encoded_json, save_file, indent = 4)
save_file.close()
