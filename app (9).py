# app.py
import os
import re
from pathlib import Path
import gradio as gr
from groq import Groq

# Load API key (set in HF Spaces "Variables and secrets")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Max chars for TTS
MAX_TTS_CHARS = 400

def voice_chat(audio_file):
    try:
        # 1️⃣ Speech-to-Text (Transcription)
        with open(audio_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        user_text = transcription.text.strip() if transcription.text else "No speech detected."

        # 2️⃣ AI Response
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": "You are a helpful AI voice assistant."},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_completion_tokens=512,
        )

        ai_text = completion.choices[0].message.content.strip()

        # 🚨 Remove <think>...</think> or similar tags
        ai_text = re.sub(r"<.*?>.*?</.*?>", "", ai_text, flags=re.DOTALL).strip()

        # 3️⃣ Limit text length for TTS
        if len(ai_text) > MAX_TTS_CHARS:
            ai_text = ai_text[:MAX_TTS_CHARS] + " ... (truncated)"

        # 4️⃣ Text-to-Speech
        speech_path = Path("response.wav")
        response = client.audio.speech.create(
            model="playai-tts",
            voice="Basil-PlayAI",
            response_format="wav",
            input=ai_text,
        )
        with open(speech_path, "wb") as f:
            f.write(response.read())

        return user_text, ai_text, str(speech_path)

    except Exception as e:
        return "❌ Error during processing", f"Error: {str(e)}", None


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Voice-to-Voice AI Assistant (Groq)")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="🎤 Record or Upload Audio"
            )
            submit_btn = gr.Button("Send to AI 🚀")

        with gr.Column():
            user_text = gr.Textbox(label="📝 Transcribed Input")
            ai_text = gr.Textbox(label="🤖 AI Response (Text)")
            audio_output = gr.Audio(label="🔊 AI Response (Audio)", type="filepath")

    submit_btn.click(fn=voice_chat, inputs=[audio_input], outputs=[user_text, ai_text, audio_output])

demo.launch()
