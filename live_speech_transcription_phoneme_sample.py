import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import speech_recognition as sr
import time
from datetime import timedelta
import io
import soundfile as sf


# load model and processor
tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft",legacy=False, clean_up_tokenization_spaces=True)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

r = sr.Recognizer()

val = str()
val = 'yes'

#start time of application
start_time = time.monotonic()

with sr.Microphone(sample_rate=16000) as source:
	while val == 'yes':
		print('start speaking please!')
		audio = r.listen(source) #pyaudio object
		data = io.BytesIO(audio.get_wav_data()) # list of bytes
		clip = AudioSegment.from_file(data) # numpy array
		x = torch.FloatTensor(clip.get_array_of_samples()) # tensor

		inputs = tokenizer(x,sampling_rate=16000,return_tensors = 'pt', padding = 'longest').input_values
		logits = model(inputs).logits
		tokens = torch.argmax(logits, axis = -1) # get the distribution of each time stamp
		text = tokenizer.batch_decode(tokens) # convert tokens into a string


		print('You said (in phonemes): ', str(text).lower())
		val = input('Do you want to keep going? ')
		val = str(val)
	print('all done!')



#runtime
end_time = time.monotonic()
print('runtime: ',timedelta(seconds=end_time - start_time))