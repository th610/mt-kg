import base64
import io
from PIL import Image

def encode_numpy_image(image):
    if image.shape[2] != 3:
        image = image.transpose(1, 2, 0)
    image_out = Image.fromarray(image)
    buffer = io.BytesIO()
    image_out.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')