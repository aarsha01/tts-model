from flask import Flask, render_template, request, jsonify, send_file
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed,AutoFeatureExtractor   
import soundfile as sf

app = Flask(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")
feature_extractor = AutoFeatureExtractor.from_pretrained("parler-tts/parler-tts-mini-expresso")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['inputText1']
    description = data['inputText2']
    
    try:
        print("generating audio")
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        SAMPLE_RATE = feature_extractor.sampling_rate
        
        set_seed(42)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, do_sample=True, temperature=1.0)
        audio_arr = generation.cpu().numpy().squeeze()
        
        # Save the generated audio to a file
        audio_filename = "parler_tts_out.wav"
        sf.write(audio_filename, audio_arr, model.config.sampling_rate)
        
        
        # Return a JSON response with the filename or status
        return jsonify({'status': 'success', 'message': 'Audio generated successfully', 'audio_filename': audio_filename})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
@app.route('/<path:filename>')
def download_file(filename):
    return send_file(filename)
if __name__ == '__main__':
    app.run(debug=True)
