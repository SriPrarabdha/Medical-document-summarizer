# from pdf2image import convert_from_path
# import json
# import base64

# # Store Pdf with convert_from_path function
# # while True:
# images = convert_from_path('Air Grills with VCD.pdf',last_page= 5)

# encoded_json = {}

# for i in range(len(images)):
#     # Save pages as images in the pdf
#     images[i].save('testpage'+ str(i) +'.jpg', 'JPEG')
#     with open("testpage{}.jpg".format(i), "rb") as image2string:
#         converted_string = base64.b64encode(image2string.read())

#     encoded_json[i] = str(converted_string)

# save_file = open("testsavedata.json", "w")
# json.dump(encoded_json, save_file, indent = 4)
# save_file.close()

import pypdfium2 as pdfium

pdf = pdfium.PdfDocument("data\Medical_Record_File_1.pdf")
print('hello')
n_pages = len(pdf)
print(n_pages )
for page_number in range(n_pages):
    page = pdf.get_page(page_number)
    bitmap = page.render(
        scale=3,
    )
    pil_image = bitmap.to_pil()
    pil_image.save(f"all_images/Medical_Record_File_1/page{page_number+1}.png")