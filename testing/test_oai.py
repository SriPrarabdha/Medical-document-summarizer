import base64
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# def convert_pdf_to_base64(pdf_path):
    
def convert_pdf_chunk_to_base64(pdf_path, first_page=4, last_page=24, output_folder="converted_images"):

  images = pdf2image.convert_from_path(pdf_path=pdf_path, first_page=first_page, last_page=last_page)
  encoded_images_json = {}

  for i, image in enumerate(images):
    with io.BytesIO() as output:
      
      image.save(output, format="JPEG")
      image_data = output.getvalue()
     
      encoded_image = base64.b64encode(image_data).decode("utf-8")

      filename = f"page_{i+4}.jpg"  
      encoded_images_json[f"{i+4}": encoded_image]
      image.save(os.path.join("/teamspace/studios/this_studio/images", filename), format="JPEG")
  
  try:
    json_string = json.dumps(encoded_images_json, indent=4)  

    with open(pdf_path, 'w') as f:
        json.dump(data, f)
  except: 
    pass
  return encoded_images


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4-turbo")

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries


# Image summaries
img_base64_list, image_summaries = generate_img_summaries(fpath)