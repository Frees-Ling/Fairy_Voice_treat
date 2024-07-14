from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

def text_to_speech(text, output_file='output.wav'):
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')

    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split = 'validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]['xvector']).unsqueeze(0)
