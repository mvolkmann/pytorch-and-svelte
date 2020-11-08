from io import BytesIO
from typing import cast, List, Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image  # type: ignore
from pydantic import BaseModel
import torch
import torch.nn.functional as F  # for softmax
from torchvision import transforms  # type: ignore

Classification = Tuple[float, str]
Result = List[Classification]

class Model(BaseModel):
    result: Result

class FileUpload(BaseModel):
    file: UploadFile = File(...)

# Load a neural network model.
model = torch.hub.load('pytorch/vision:master', 'resnet101', pretrained=True)

# Put the network into "eval" mode because we want to
# evaluate input rather than perform training.
model.eval()

# Prepare the image for input to the network.
preprocess = transforms.Compose([
    # Resize the image to reduce the number of pixels to be
    # processed and match the image sizes used for training.
    transforms.Resize(256),

    # Crop the image to a smaller size about its center,
    # removing unnecessary pixels at the edges.
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

# JSON in request bodies of POST and PUT requests
# is validated against this type definition.
# When validation fails, the response status
# is set to 422 Unprocessable Entity.
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins='*')

# @app.post('/classify', response_model=FileUpload)
@app.post('/classify')
# async def classify_image(file: UploadFile = File(...)) -> Result:
# async def classify_image(file: UploadFile = File(...)) -> Model:
async def classify_image(file: UploadFile = File(...)) -> Result:
    data = cast(bytes, await file.read())
    pil_image = Image.open(BytesIO(data))
    image_t = preprocess(pil_image)

    # Create a 1D tensor object from the image
    # with the data starting at index zero.
    # Why isn't zero the default?
    batch_t = torch.unsqueeze(image_t, 0)

    # Perform inference to get predicted classes.
    # "out" is set to a tensor that contains percentage predictions
    # for each of the 1000 possible ImageNet classes.
    out = model(batch_t)

    # Get the 1000 possible labels from a text file.
    with open('./imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

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
    return result
