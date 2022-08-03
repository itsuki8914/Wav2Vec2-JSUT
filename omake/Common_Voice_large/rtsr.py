import pyaudio
import numpy as np
import torch 
import torch.nn as nn
import torchaudio
from datasets import load_dataset, load_metric, Audio
import os
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
import torchaudio as ta

def load_tkn_hira(path='./vocab_roman.json'):
    tokenizer = Wav2Vec2CTCTokenizer(path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    return tokenizer

def load_fe(sr=16000):
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sr, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    return feature_extractor

def load_processor(fe, tkn):
    processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tkn)
    return processor

def load_w2v2model(path, processor):
    model = Wav2Vec2ForCTC.from_pretrained(
    'facebook/wav2vec2-large-xlsr-53',
    attention_dropout=0.2,
    hidden_dropout=0.2,
    feat_proj_dropout=0.2,
    mask_time_prob=0.1,
    layerdrop=0.2,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    diversity_loss_weight=100
    )

    model.lm_head = nn.Linear(1024, len(processor.tokenizer))
    model.config.vocab_size=len(processor.tokenizer)
    model.load_state_dict(torch.load(path))
    #model.half()
    return model

def hira_predict(x, processor, model, sr=16000):
    input_values = processor(x, return_tensors="pt",sampling_rate=sr).input_values
    input_values = torch.tensor(input_values, requires_grad=False, dtype=torch.float32)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription


def main():

    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    THRESHOLD = 0.1

    waitseconds = 1.5 #待ち時間

    vocab_path='./vocab_hira.json'
    w2v2_model_path='./wav2vec2_xlsr_ja_hira/checkpoint-53450/pytorch_model.bin'
    p = pyaudio.PyAudio()
    
    tokenizer_hira = load_tkn_hira(vocab_path)
    feature_extractor = load_fe(RATE)
    processor = load_processor(feature_extractor, tokenizer_hira)
    w2v2 = load_w2v2model(w2v2_model_path, processor)


    stream = p.open(format = FORMAT,
        channels = CHANNELS,
        rate = RATE, #サンプリングレート
        input = True,
        frames_per_buffer = chunk
    )
    print('ready to recognize...')
    cnt = 0
    frames = []
    try:
        while True:
            data = stream.read(chunk)
            x = np.frombuffer(data, dtype="int16") / 32768.0
            frames.append(x)
            if np.max(np.abs(x)) < 0.003: #音量、無音とみなす
                cnt += 1
            else:
                cnt = 0
            if cnt > waitseconds / (chunk / RATE): #chunk / RATEは一度に処理する時間、ここでは0.021秒くらい
                y = np.concatenate(frames)
                
                if np.max(np.abs(y))>THRESHOLD:
                    y /= np.max(np.abs(y))
                    hira = hira_predict(y, processor, w2v2, RATE)
                    print(hira)

                frames = []
                cnt = 0
                
    except KeyboardInterrupt:
        print('stop recognizing...')
 
    stream.close()
    p.terminate()


if __name__=='__main__':
    main()



