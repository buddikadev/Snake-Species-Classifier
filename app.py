import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Important: Enable Legacy Keras support to handle models trained in R/older Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Try importing from tf_keras first (fix for Keras 3 compatibility issues)
try:
    from tf_keras.models import load_model
    from tf_keras.preprocessing.image import img_to_array
except ImportError:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array

# Initialize the FastAPI Application
app = FastAPI()

# Get the current directory path for reliable file access on Cloud Servers
current_dir = os.path.dirname(os.path.realpath(__file__))

# Mount Static files (CSS, JS, Images)
# os.path.join ensures the path works on both Windows and Linux (Render)
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Initialize Jinja2 Templates for rendering HTML
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Load the trained AI model (.h5 file)
MODEL_PATH = os.path.join(current_dir, "snake_species_classifier_Finetuned.h5")
model = load_model(MODEL_PATH)

# List of Snake Species (In the same order as the model's output layer)
CLASS_NAMES = [
    "Ceylon Krait - මුදු කරවලා",
    "Common Krait - තෙල් කරවලා",
    "Common Rat Snake - ගැරඬියා",
    "Green Pit Viper - පළා පොළඟා",
    "Green Vine Snake - ඇහැටුල්ලා",
    "Hypnale hypnale - පොලොන් තෙලිස්සා",
    "Indian Cobra - නාගයා",
    "Indian Python - පිඹුරා",
    "Russell’s Viper - තිත් පොළඟා",
    "Saw-scaled Viper - වැලි පොළඟා"
]

def prepare_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess the uploaded image before passing it to the AI model.
    1. Convert to RGB
    2. Resize to 224x224
    3. Normalize pixels (0-1)
    4. Expand dimensions for batch processing
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route to serve the Home Page (index.html)
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    # Context must include 'request' for Jinja2 compatibility with FastAPI
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )

# API Route to handle Image Upload and Prediction
@app.post("/predict")
async def predict_snake(file: UploadFile = File(...)):
    try:
        # Read the uploaded file bytes and open as an Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess and run the model prediction
        processed_image = prepare_image(image)
        predictions = model.predict(processed_image)
        
        # Get the index of the highest probability
        predicted_idx = np.argmax(predictions[0])
        
        # Fetch class name and calculate confidence percentage
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[0][predicted_idx] * 100)
        
        return JSONResponse({
            "success": True,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        # Return error message if something goes wrong
        return JSONResponse({"success": False, "error": str(e)})

# Start the Server
if __name__ == "__main__":
    import uvicorn
    # Cloud services like Render provide a dynamic PORT via environment variables
    # Default to 8000 if no environment variable is set
    port = int(os.environ.get("PORT", 8000))
    
    # Use 0.0.0.0 to allow external access (Required for Hosting)
    uvicorn.run(app, host="0.0.0.0", port=port)