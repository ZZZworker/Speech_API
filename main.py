from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from model import (
    #load_noise_reduction_model,
    load_speech_to_text_model,
    load_gemini_model,
    #reduce_noise,
    transcribe_audio,
    correct_text_with_keys,
    parse_to_dict,
)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
#noise_reduction_model = load_noise_reduction_model()
speech_to_text_model = load_speech_to_text_model()
gemini_model = load_gemini_model(api_key="AIzaSyCRkvVUVU6ryDvWbxSfOPwQba09kETuh0o")

# Keys สำหรับการแยกข้อความ
Mykeys = [
    'ชื่อแพทย์', 
    'เลขบิ้บนักวิ่ง', 
    'อาการ', 
    'อัตราการเต้นหัวใจ', 
    'อัตราการหายใจ', 
    'อุณหภูมิ', 
    'หลักกิโลเมตร'
]

TEMP_FOLDER = "./temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

@app.get("/")
def index():
    return {"message": "Welcome to the Combined Model API"}

@app.post("/process-audio")
async def process_audio(file: UploadFile):
    if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload .wav or .mp3 files.")

    input_path = os.path.join(TEMP_FOLDER, file.filename)
    output_path = os.path.join(TEMP_FOLDER, "filtered_" + file.filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        # ลดเสียงรบกวน
        #reduce_noise(noise_reduction_model, input_path, output_path)

        # แปลงเสียงเป็นข้อความ
        transcription = transcribe_audio(speech_to_text_model, output_path)

        # แก้ไขข้อความและแบ่งตาม Keys
        corrected_text = correct_text_with_keys(gemini_model, transcription, Mykeys)

        # แปลงข้อความเป็น Dictionary
        result_dict = parse_to_dict(corrected_text)

        return JSONResponse(result_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# รัน Uvicorn ภายใน main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
