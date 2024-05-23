from utils.data_utils import extract_toc
import json 
import pypdfium2 as pdfium

def report_wise_json_data_extraction(filename):
    json_data = None 
    # filename = 'Medical_Record_File_3'

    with open('Extracted_Data/{}.json'.format(filename), 'r') as file:
        json_data = json.load(file)

    toc_items = list(extract_toc('data/{}.pdf'.format(filename)).items())

    pdf = pdfium.PdfDocument("data/{}.pdf".format(filename))
    n_pages = len(pdf)
    counter = 1

    send_chunk = {}
    for i in range(len(toc_items)):
        spage = toc_items[i][1]
        epage = toc_items[i+1][1] if i+1 < len(toc_items) else n_pages

        temp = ""
        
        if epage-spage > 15:
            temps = spage 
            tempe = spage+15
            while(tempe < epage):
                temp = ""
                print(temps , "end=", tempe)
                for page in range(temps, tempe):
                    if (temps == tempe):
                        continue
                    # print(spage , "end=", epage)
                    # send_chunk[i] = json_data[str(page)]
                    temp = "\n".join([temp, json_data[str(page)]])
                
                if (tempe+15 < epage):
                    temps = tempe
                    tempe += 15
                else:
                    temps = tempe
                    tempe = epage
                    
                if temp != "":
                    send_chunk[counter] = temp
                    counter +=1
        else:
            print(spage , "end=", epage)
            for page in range(spage, epage):
                if (spage == epage):
                    continue
                # send_chunk[i] = json_data[str(page)]
                # print(spage , "end=", epage)
                temp = "\n".join([temp, json_data[str(page)]])

            if temp != "":
                send_chunk[counter] = temp
                counter +=1

    return send_chunk