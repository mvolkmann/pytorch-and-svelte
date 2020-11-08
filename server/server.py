from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple

Classification = Tuple[float, str]
Result = List[Classification]

class Model(BaseModel):
    result: Result

class FileUpload(BaseModel):
    file: UploadFile = File(...)

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
async def classify_image(image: UploadFile = File(...)):
    print('server.py classify_image: image =', image)
    result = [(0.9, 'whippet'), (0.05, 'greyhound')]
    # return {'result': result}
    return result
