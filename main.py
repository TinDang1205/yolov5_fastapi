import shutil
import uvicorn

from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile

from predict import predict_model
import os
# Set your Cloudinary credentials
# ==============================
from dotenv import load_dotenv

load_dotenv()

# Import the Cloudinary libraries
# ==============================
import cloudinary
import cloudinary.uploader
import cloudinary.api

config = cloudinary.config(secure=True)


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
file_upload = "static"
path_to_model = 'weights/best.pt'


@app.post("/api/predict")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    if uploaded_file:
        # Lưu file
        path_to_save = os.path.join(file_upload, uploaded_file.filename)
        print("Save = ", path_to_save)
        file_location = f"{file_upload}/{uploaded_file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(uploaded_file.file.read())
        url, data, filename = predict_model(weights=path_to_model,
                                            source=path_to_save)  # http://server.com/static/path_to_save
        var = filename.split('.')[0]
        cloudinary.uploader.upload(url, public_id=var, unique_filename=False, overwrite=True)
        # Build the URL for the image and save it in the variable 'srcURL'
        srcURL = cloudinary.CloudinaryImage(filename).build_url()
        data = {
            "image": srcURL,
            "data": data,
        }
        return jsonable_encoder(data)

    return 'Upload file to detect'
# Start Backend
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
