#from deepfilter import DeepFilterNet
from transformers import pipeline
import google.generativeai as genai
import torch

# # โหลดโมเดล DeepFilterNet
# def load_noise_reduction_model():
#     # เลือกโมเดล เช่น deepfilter2, deepfilter2lite
#     # หรือ path ไปยัง checkpoint
#     return DeepFilterNet(model_name_or_path="deepfilter2")

# โหลดโมเดล Speech-to-Text (Thonburian Whisper)
def load_speech_to_text_model():
    MODEL_NAME = "biodatlab/whisper-th-medium-combined"
    device = 0 if torch.cuda.is_available() else "cpu"
    return pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )
    

# โหลดโมเดล Gemini 1.5 Flash
def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# ฟังก์ชันสำหรับลดเสียงรบกวน
# def reduce_noise(model, input_audio_path, output_audio_path):
#     model.enhance_file(input_audio_path, output_audio_path)


# ฟังก์ชันสำหรับแปลงเสียงเป็นข้อความ
def transcribe_audio(model, audio_path):
    transcription = model(
        audio_path,
        batch_size=16,
        return_timestamps=False,
        generate_kwargs={"language": "<|th|>", "task": "transcribe"},
    )["text"]
    return transcription.replace(" ", "")

# ฟังก์ชันสำหรับแก้ไขข้อความด้วย Gemini
def correct_text(model, text):
    prompt = f"แก้ไขข้อความนี้ให้ถูกต้อง: {text}"
    response = model.generate_content(prompt)
    return response.text

# ฟังก์ชันสำหรับแก้ไขข้อความและแบ่งตาม Key
def correct_text_with_keys(model, text, keys):
    prompt = (
        f"จงแบ่งประโยคต่อไปนี้ให้อยู่ในแต่ละหัวข้อตามที่อยู่ใน key ที่กำหนดไว้: {keys} "
        "หากมีสาเหตุให้ใส่เข้าไปในช่อง key ที่เป็นอาการด้วย หากไม่มีข้อมูลไม่ต้องใส่ค่า "
        f"ไม่ต้องใส่หัวข้อมาว่านี่คือการแบ่ง: {text}"
    )
    response = model.generate_content(prompt)
    return response.text

# ฟังก์ชันแปลงข้อความที่แบ่งแล้วให้เป็น Dictionary
def parse_to_dict(text):
    result_dict = {}
    lines = text.split("\n")
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            result_dict[key.strip()] = value.strip()
    return result_dict
