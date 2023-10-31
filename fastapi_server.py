import io
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import speech_recognition as sr

from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)

#import os

#openai_api_key = os.environ.get("OPENAI_API_KEY")
#if not openai_api_key:
#    raise ValueError("OpenAI API key not found in environment variables.")


app = FastAPI()


class TextModel(BaseModel):
    text: str

def audio_to_text_using_speechrecognition(audio_file_bytes):
    recognizer = sr.Recognizer()
    
    # Use io.BytesIO to create a file-like object from the audio bytes
    file_like = io.BytesIO(audio_file_bytes)
    
    with sr.AudioFile(file_like) as source:
        audio_data = recognizer.record(source)
        try:
            # Using Google Web Speech API (requires internet connection)
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Google Web Speech API could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Web Speech API; {e}"

@app.get('/')
async def index():
    return {"test": 'test'}

@app.post("/audio_to_text/")
async def audio_to_text(audio: UploadFile = None):
    if not audio:
        return {"error": "No audio file provided"}

    audio_bytes = audio.file.read()
    recognized_text = audio_to_text_using_speechrecognition(audio_bytes)
    return {"text": recognized_text}

@app.post("/get_response/")
async def get_response(data: TextModel):

    message = data.text # "Hi, ChatGPT!"  # あなたの質問をここに書く
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=message)
    ]

    llm = ChatOpenAI()  # ChatGPT APIを呼んでくれる機能

    response = llm(messages)
    response_txt = response.content[:]
    #response = f"Generated response for: {data.text}"
    #response_api = f"Generated response for: {response}"
    return {"response": response_txt}

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": "Error occurred."},
    )

