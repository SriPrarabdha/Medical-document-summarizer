from langchain_community.document_loaders import PyPDFLoader
import re 

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