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
    'charsiu/zh_w2v2_tiny_fc_10ms', 
    attention_dropout=0.2,
    hidden_dropout=0.2,
    feat_proj_dropout=0.2,
    mask_time_prob=0.075,
    layerdrop=0.2,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    )

    model.lm_head = nn.Linear(384, len(processor.tokenizer))
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
    max_input_length = 256
    max_target_length = 128

    waitseconds = 1.5 #待ち時間

    vocab_path='./vocab_hira.json'
    w2v2_model_path='./wav2vec2_tiny_ja_hira/checkpoint-7400/pytorch_model.bin'
    
    tokenizer_hira = load_tkn_hira(vocab_path)
    feature_extractor = load_fe(RATE)
    processor = load_processor(feature_extractor, tokenizer_hira)
    w2v2 = load_w2v2model(w2v2_model_path, processor)

    p = pyaudio.PyAudio()

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
                y *= 2.0
                y = np.clip(y, -1, 1)
                if np.max(np.abs(y))>0.1:
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



