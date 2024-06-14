from flask import Flask, request, render_template_string, send_file
import torch
from transformers import AutoTokenizer, set_seed, AutoFeatureExtractor
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import os
import shutil
from pydub import AudioSegment
import uuid
import subprocess
import nltk

nltk.download('punkt')

app = Flask(__name__)

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")
feature_extractor = AutoFeatureExtractor.from_pretrained("parler-tts/parler-tts-mini-expresso")

SEED = 4294967295

def gen_tts(text, description):
    global SEED
    inputs = tokenizer(description, return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)

    set_seed(SEED)
    generation = model.generate(
        input_ids=inputs.input_ids, prompt_input_ids=prompt.input_ids, do_sample=True, temperature=1.0
    )
    audio_arr = generation.cpu().numpy().squeeze()

    return model.config.sampling_rate, audio_arr

def get_temp_name(file_name):
    ori_name = os.path.basename(file_name)
    file_name, file_extension = os.path.splitext(ori_name)
    random_uuid = str(uuid.uuid4())[:6]
    new_name = f'{file_name}_{random_uuid}{file_extension}'
    return new_name

def remove_silence_from_audio(audio_file, margin):
    base_path = os.path.dirname(audio_file)
    random_uuid = str(uuid.uuid4())[:6]
    new_save_path = None
    logs = None
    try:
        ffprobe_command = f"ffprobe -i {audio_file} -show_entries format=duration -v quiet -of csv=p=0"
        duration_string = subprocess.check_output(ffprobe_command, shell=True, text=True)
        duration = float(duration_string)

        ffmpeg_command = f"ffmpeg -f lavfi -t {duration} -i color=c=black:s=160x90 -c:v libx264 -strict experimental {base_path}/blank.mp4 -y"
        white_var = os.system(ffmpeg_command)

        if white_var == 0:
            response_var = os.system(f"ffmpeg -i {base_path}/blank.mp4 -i {audio_file} -map 0 -map 1 -c:v copy -c:a aac -strict experimental {base_path}/temp.mp4 -y")
            if response_var == 0:
                var = os.system(f"auto-editor {base_path}/temp.mp4 --margin {float(margin)}sec")
                if var == 0:
                    new_save_path = f"./results/remove_silence_{get_temp_name(audio_file)}"
                    final_var = os.system(f"ffmpeg -i {base_path}/temp_ALTERED.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 {new_save_path} -y")
                    if final_var == 0:
                        ffprobe_command = f"ffprobe -i {new_save_path} -show_entries format=duration -v quiet -of csv=p=0"
                        after_duration_string = subprocess.check_output(ffprobe_command, shell=True, text=True)
                        after_duration = float(after_duration_string)
                        logs = f"Before: {duration} sec\nAfter: {after_duration} sec"
                        print(f"Audio saved")
                    else:
                        print("Failed to remove silence from audio")
                else:
                    print("Failed to edit video")
            else:
                print("Can't add audio to blank video")
        else:
            print("Can't create blank video")

    except Exception as e:
        print(e)
        print("Can't find audio duration")
    return new_save_path, logs

def merge_audio_files(temp_list, output_path):
    merged_audio = AudioSegment.empty()
    for audio_file in temp_list:
        audio = AudioSegment.from_wav(audio_file)
        merged_audio += audio
    merged_audio.export(output_path, format="wav")

def name_from_text(text):
    random_uuid = uuid.uuid4().hex[:6]
    if len(text) == 0:
        raise ValueError("Text must be at least 20 characters long.")
        filename = f"{random_uuid}.wav"
    else:
        truncated_text = text[:20]
        sanitized_text = truncated_text.replace("'", "_").replace(".", "_").replace('"', "_")
        filename = f"{sanitized_text}_{random_uuid}.wav"
    return './results/' + filename.replace(" ", "_")

def voice_design(text, description, large=False, remove_silence_from_tts=False, silence_margin='0.2'):
    save_path = name_from_text(text)
    if os.path.exists("./chunks_audio"):
        shutil.rmtree("./chunks_audio")
    os.mkdir("./chunks_audio")
    if large:
        sentences = nltk.sent_tokenize(text)
        temp_list = []
        for i, sentence in enumerate(sentences):
            try:
                temp_path = f"./chunks_audio/{i}.wav"
                SAMPLE_RATE, audio_arr = gen_tts(sentence, description)
                sf.write(temp_path, audio_arr, SAMPLE_RATE)
                temp_list.append(temp_path)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        if len(temp_list) == 1:
            shutil.copy(temp_list[0], save_path)
        elif len(temp_list) > 1:
            merge_audio_files(temp_list, save_path)
        else:
            print("No audio")
    else:
        SAMPLE_RATE, audio_arr = gen_tts(text, description)
        sf.write(save_path, audio_arr, SAMPLE_RATE)
    if remove_silence_from_tts:
        remove_silence_save_path, log_data = remove_silence_from_audio(save_path, silence_margin)
        return remove_silence_save_path, remove_silence_save_path, log_data
    return save_path, save_path, ''

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Text to Speech Model</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
                .form-container {
                    display: flex;
                    flex-direction: column;
                    align-items: flex-start;
                }
                .form-container label {
                    margin: 10px;
                }
                .form-container input[type="text"] {
                    margin: 10px;
                }
                #input_text {
                    width: 450px;
                    height: 100px;
                }
                #prompt_text {
                    width: 450px;
                    height: 30px;
                }
                .form-container input[type="submit"] {
                    margin: 10px;
                    height: 30px;
                }
            </style>
        </head>
        <body>
            <h1>Text to Speech Model</h1>
            <p>Enter your input text and prompt:</p>
            <form action="/generate_audio" method="post">
                <div class="form-container">
                    <label for="input_text">Input:</label>
                    <input type="text" id="input_text" name="input_text">
                    <label for="prompt_text">Prompt:</label>
                    <input type="text" id="prompt_text" name="prompt_text">
                    <input type="submit" value="Submit">
                </div>
            </form>
        </body>
        </html>
    ''')

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        input_text = request.form['input_text']
        prompt_text = request.form['prompt_text']

        # Generate TTS
        save_path, _, logs_data = voice_design(input_text, prompt_text, large=True, remove_silence_from_tts=False)

        # Serve the generated audio file
        return send_file(save_path, as_attachment=True)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    if not os.path.exists('./results'):
        os.mkdir('./results')
    app.run(debug=True)
