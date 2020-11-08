# pytorch-and-svelte

This project includes a Svelte web app and a Python FastAPI server.
The Svelte web app allows users to select an image file
to be classified using a PyTorch neural network.
The Python FastAPI server supports image file upload
and returns classification information about the image.

The server requires the following installs:

- `pip install fastapi pydantic python-multipart uvicorn`

To run the server:

- `cd server`
- `./start`

To run the client:

- `cd ui`
- `npm run dev`
- browse localhost:5000
- select an image file
- classification of the image will be displayed
