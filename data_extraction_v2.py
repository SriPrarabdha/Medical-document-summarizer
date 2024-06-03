import os
import base64
import zipfile
import logging
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_path

# Set up Google API key for Generative AI
os.environ["GOOGLE_API_KEY"] = "AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU"

# Configure logging
logging.basicConfig(level=logging.INFO)

class PDFExtractor:
    def __init__(self, pdf_path, output_folder="extracted_images"):
        self.pdf_path = pdf_path
        self.output_folder = output_folder

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract_text_from_image(self, img_base64):
        prompt = "Extract the text data from the image."
        model = ChatGoogleGenerativeAI(model='gemini-pro-vision', temperature=0.8)
        msg = model.invoke([HumanMessage(content=[{"type": "text", "text": prompt},
                                                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}])])
        return msg.content

    def extract_images_from_pdf(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        images = convert_from_path(self.pdf_path)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(self.output_folder, f"page_{i+1}.jpg")
            image.save(image_path, "JPEG")
            image_paths.append(image_path)
        return image_paths

    def generate_img_data(self, image_paths):
        img_data_list = []
        img_text_data = []
        for img_path in image_paths:
            base64_image = self.encode_image(img_path)
            img_data_list.append({
                "image_file": os.path.basename(img_path),
                "page_number": None
            })
            text_data = self.extract_text_from_image(base64_image)
            img_text_data.append({
                "content": text_data,
                "page_number": None
            })
        return img_data_list, img_text_data

    def extract_data_from_pdf(self):
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        extracted_data = []
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata
            page_number = metadata.get("page", None)
            extracted_data.append({
                "content": content,
                "page_number": page_number
            })
        return extracted_data

    def extract(self):
        extracted_data = self.extract_data_from_pdf()
        image_paths = self.extract_images_from_pdf()
        img_data_list, img_text_data = self.generate_img_data(image_paths)
        all_extracted_data = extracted_data + img_text_data
        output_data = {
            "texts": all_extracted_data,
            "images": img_data_list
        }
        return output_data
