import gradio as gr
import os
import numpy as np
from pydub import AudioSegment
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transformers import pipeline
from openai import OpenAI


class AudioProcessor:
    def __init__(self):
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=0)
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_message = """You are a NLP Agent. You analyze conversation between Agent and customer and provide insights on, 
        - Sentiment (Positive, Negative, Neutral)
        - Intent/Topic List
        - Entity List
        - Issue List
        - Resolution Summary
        - Customer Satisfaction
        """

    def transcribe(self, audio):
        sr, y = audio
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        return self.transcriber({"sampling_rate": sr, "raw": y})["text"]

    def convert_mp3_to_wav(self, mp3_path, wav_path):
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        return wav_path

    def get_conversation_from_audio(self, file_path):
        ext = os.path.splitext(file_path)[1]
        if ext == ".mp3":
            wav_file_path = file_path.replace(".mp3", ".wav")
            wav_path = self.convert_mp3_to_wav(file_path, wav_file_path)
        else:
            wav_path = file_path
            
        print("wav path", wav_path)
        diarization = self.diarization_pipeline(wav_path)
        audio_meta = diarization.to_lab().split('\n')
        conversation_text = []

        for meta in audio_meta:
            try:
                starttime, endtime, speaker = meta.split(' ')
                audio = AudioSegment.from_file(wav_path)
                extracted_segment = audio[float(starttime) * 1000:float(endtime) * 1000]
                text = self.transcribe((extracted_segment.frame_rate, np.array(extracted_segment.get_array_of_samples())))
                conversation_text.append(f"Speaker {speaker}: {text}")
            except:
                pass
        return conversation_text

    def get_completion(self, conversation_text):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": conversation_text}
            ]
        )
        return completion.choices[0].message.content

processor = AudioProcessor()
# conversation_text = processor.get_conversation_from_audio("audio.wav")
# print(processor.get_completion("\n".join(conversation_text)))

def analyze_audio(audio_path):
    # save the audio file 

    # Placeholder for actual audio analysis logic
    conversation = processor.get_conversation_from_audio(audio_path)
    print(conversation)
    result = processor.get_completion("\n".join(conversation))
    return result

title = "Conversation Analysis for Customer Call Agent Calls"
description = """
This application allows you to upload an audio file of a customer call with an agent.
It will analyze the conversation and provide insights such as sentiment, keywords, and more.
"""

demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio( type="filepath"),
    outputs=gr.Markdown(),
    title=title,
    description=description
)

demo.launch()