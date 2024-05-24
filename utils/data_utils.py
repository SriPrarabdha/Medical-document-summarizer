from langchain_community.document_loaders import PyPDFLoader
import re 

import fitz 
import json

abb = ['MD', 'DO', 'PT', 'DC', 'P.C.']

def get_doctor_page_number(filename):
    loader = PyPDFLoader("data/Medical_Record_File_1.pdf")
    pages = loader.load_and_split()

    data = {}
    # text = pages[0].page_content
    # lines = text.splitlines()
    pattern = r'(.*?)(\d+)$'
    retry = 3
    data_found = True
    for i in range(len(pages)):
        if not data_found:
            retry -=1
            print(f'No doctor found on the current page {i}. Retrying agian')
            print("Total Retrys left.... ", retry)
        
        if retry==0:
            break

        text = pages[i].page_content
        lines = text.splitlines()
        data_found = False

        for line in lines:
            if any(substr in line for substr in abb):
                match = re.search(pattern, line)

                if match:
                    doctor_name = match.group(1)
                    doctor_page = match.group(2)
                    data[doctor_name] = int(doctor_page)
                    data_found = True

    print(data)
    return data



def extract_toc(pdf_path):
    document = fitz.open(pdf_path)

    # Extract the table of contents
    toc = document.get_toc()

    # Create a dictionary for the TOC
    toc_dict = {}

    for item in toc:
        title = item[1]
        page = item[2]
        toc_dict[title] = page
    
    #Sorting the dictionary based on the values
    toc_dict = {k: v for k, v in sorted(toc_dict.items(), key=lambda item: item[1])}
    
    print(toc_dict)
    return toc_dict

def slice_doctor_content(pages_json):
    
    pages = list(pages_json.values())
    pages.sort()
    pages.insert(0,1)

    docs = []

    for i in range(len(pages)-1):
        text = ''
        for j in  range(pages[i], pages[i+1]):
            text= text+ " " + data[str(j)]

        docs.append(text)

    print(docs)
    return docs

# table of content 
!pip install PyMuPDF
import fitz  # PyMuPDF
import json
import re

def extract_toc(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    results = {}

    for entry in toc:
        level, title, page = entry
        results[title] = page
    
    doc.close()
    return results

def extract_doctors_info_from_toc(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doctor_info = {}

    for entry in toc:
        level, title, page = entry

        
        doctor_names = re.findall(r'\bDr\.\s[A-Z][a-zA-Z]+\b', title)

        for doctor in doctor_names:
            if doctor not in doctor_info:
                doctor_info[doctor] = {
                    "starting page": page,
                    "ending page": page,
                    "date of starting": None,
                    "date of ending": None
                }
            else:
                doctor_info[doctor]["ending page"] = page

        
            dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', title)
            if dates:
                if doctor_info[doctor]["date of starting"] is None:
                    doctor_info[doctor]["date of starting"] = dates[0]
                doctor_info[doctor]["date of ending"] = dates[-1]

    doc.close()

    formatted_info = {}
    for doctor, info in doctor_info.items():
        date_range = None
        if info["date of starting"] and info["date of ending"]:
            date_range = f'{info["date of starting"]} - {info["date of ending"]}'
            formatted_info[date_range] = info["starting page"]
        else:
            formatted_info[doctor] = info["starting page"]

    return formatted_info

def main(pdf_path):
    toc_info = extract_toc(pdf_path)
    doctor_info = extract_doctors_info_from_toc(pdf_path)

    combined_info = {**toc_info, **doctor_info}

    # Convert results to JSON
    json_output = json.dumps(combined_info, indent=4)
    print(json_output)

    # Save the JSON output to a file
    with open('toc_output.json', 'w') as f:
        f.write(json_output)

# Example usage
pdf_path = '/Copy of Medical Record File 1.pdf'  
main(pdf_path)
