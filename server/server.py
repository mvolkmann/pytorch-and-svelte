from io import BytesIO
from typing import cast, List, Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F  # for softmax
from torchvision import transforms  # type: ignore

Classification = Tuple[float, str]
Result = List[Classification]

def remove_alpha(src: Image) -> Image:
    # Convert the image to a NumPy array.
    rgba = np.array(src)

    row, col, channels = rgba.shape
    if channels == 3:
        return rgba
    assert channels == 4, 'RGBA images have 4 channels.'

    # Create a new NumPy array to hold the RGB representation.
    rgb = np.zeros((row, col, 3), dtype='float32')

    # Split out the data from each channel.
    # Red values are in channel 0, green in 1, blue in 2, and alpha in 3.
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    # Convert the alpha values from a range of 0-255 to 0-1.
    a = np.asarray(a, dtype='float32') / 255.0

    # Multiply all red, green, and blue values
    # by the corresponding alpha value.
    bg_red, bg_green, bg_blue = 0, 0, 0  # black
    rgb[:, :, 0] = r * a + (1.0 - a) * bg_red  # channel 0
    rgb[:, :, 1] = g * a + (1.0 - a) * bg_green  # channel 1
    rgb[:, :, 2] = b * a + (1.0 - a) * bg_blue  # channel 2

    # Convert the new color values to unsigned int values.
    rgb_ints = rgb.astype(np.uint8)

    return Image.fromarray(rgb_ints, 'RGB')

# Load an instance of the ResNet neural network.
model = torch.hub.load('pytorch/vision:master', 'resnet101', pretrained=True)

# Put the network into "eval" mode because we want to
# evaluate input rather than perform training.
model.eval()

# Create a function that prepares an image
# for input to the ResNet network
# so it matches the images used for training.
preprocess = transforms.Compose([
    # Resize the image to reduce the number of pixels to be processed.
    transforms.Resize(256),

    # Crop the image to a smaller size about its center,
    # removing possibly unnecessary pixels at the edges.
    transforms.CenterCrop(224),

    # Convert the image data to a tensor object.
    transforms.ToTensor(),

    # Normalize the red/green/blue values of the pixels to
    # have the same mean and standard deviation values
    # that were used when the model was trained.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # [red, green, blue]
        std=[0.229, 0.224, 0.225]  # [red, green, blue]
    )
])

# Get the 1000 possible labels used to
# train the ResNet network from a text file.
with open('./imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

app = FastAPI()

# Enable web apps from other domains to send requests.
app.add_middleware(CORSMiddleware, allow_origins='*')

@app.post('/classify', response_model=Result)
async def classify_image(file: UploadFile = File(...)) -> Result:
    # Convert the image in the request body to the
    # proper format to use as input to the ResNet network.
    data = cast(bytes, await file.read())
    pil_image = Image.open(BytesIO(data))

    if pil_image.mode == 'RGBA':
        pil_image = remove_alpha(pil_image)
        # TODO: Is this better or worse?
        # pil_image = pil_image.convert('RGB')
        # plt.imshow(pil_image)
        # plt.show()

    image_t = preprocess(pil_image)

    # Create a 1D tensor object from the image
    # with the data starting at index zero.
    # Why isn't zero the default?
    batch_t = torch.unsqueeze(image_t, 0)

    # Perform inference to get predicted classes.
    # "out" is set to a tensor that contains percentage predictions
    # for each of the 1000 possible ImageNet classes.
    out = model(batch_t)

    # Compute the percentage certainty for each tensor value.
    percentages = F.softmax(out, dim=1)[0] * 100

    # Get the top predictions.
    _, indices = torch.sort(out, descending=True)
    result = []
    n = 5
    for i in indices[0][:n]:
        label = labels[i]
        confidence = percentages[i].item()
        result.append((confidence, label))
    print(result)

    return result
