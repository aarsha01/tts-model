import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed,AutoFeatureExtractor   
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")
feature_extractor = AutoFeatureExtractor.from_pretrained("parler-tts/parler-tts-mini-expresso")
prompt = "Why do you make me do these examples? They're *so* generic."
description = "Thomas speaks moderately slowly in a sad tone with emphasis and high quality audio."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
SAMPLE_RATE = feature_extractor.sampling_rate
print(SAMPLE_RATE)
set_seed(42)
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, do_sample=True, temperature=1.0)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
