import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# අලුත් tf_keras පැකේජය භාවිතයෙන් මොඩලය ලෝඩ් කිරීම
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array

# FastAPI App එක ආරම්භ කිරීම
app = FastAPI()

# Static Files (CSS, JS, Images) සම්බන්ධ කිරීම
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML Templates සම්බන්ධ කිරීම
templates = Jinja2Templates(directory="templates")

# AI Model එක ලෝඩ් කිරීම (File name එක හරියටම තියෙනවද බලන්න)
MODEL_PATH = "snake_species_classifier_Finetuned.h5"
model = load_model(MODEL_PATH)

# සර්ප වර්ග 10 ලැයිස්තුව (A-Z පිළිවෙලට)
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

# පින්තූරය Model එකට ගැළපෙන පරිදි සකස් කිරීම
def prepare_image(image: Image.Image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main Page එක (index.html) පෙන්වීම සඳහා Route එක
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    # අලුත් FastAPI සංස්කරණයට ගැළපෙන පරිදි context ලබා දීම (Internal Server Error එකට විසඳුම)
    context = {"request": request}
    return templates.TemplateResponse(request=request, name="index.html", context=context)

# පින්තූරය අඳුරගැනීම සඳහා API Route එක
@app.post("/predict")
async def predict_snake(file: UploadFile = File(...)):
    try:
        # Upload කරපු පින්තූරය කියවීම
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # පින්තූරය සකස් කර අනුමාන කිරීම (Prediction)
        processed_image = prepare_image(image)
        predictions = model.predict(processed_image)
        predicted_idx = np.argmax(predictions[0])
        
        # ප්‍රතිඵලය ලබා ගැනීම
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[0][predicted_idx] * 100)
        
        return JSONResponse({
            "success": True,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

# සර්වර් එක Run කිරීම
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)