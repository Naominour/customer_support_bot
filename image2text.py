import shutil
import uuid
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import io  



def extract_passenger_id_from_image(image_path_or_url):
    # Check if the input is a URL
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")

    # Initialize processor and model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

    # Preprocess the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate text from the image
    generated_ids = model.generate(pixel_values, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().rstrip('.')

    return generated_text