import sys 
import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder import preprocess
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

model = sys.argv[1]
speaker1 = sys.argv[2]
speaker2= sys.argv[3]

if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")
        
print("Preparing the encoder")
ensure_default_models(Path("saved_models"))
encoder.load_model(args.enc_model_fpath)
    
 #preprocessing wavs


wav_1_fpaths = list(Path(path, speaker_1).glob("*.wav"))
wav_2_fpaths = list(Path(path, speaker_2).glob("*.wav"))


speaker_1_wavs = {speaker_1: list(map(preprocess_wav, wav_1_fpaths))}
speaker_2_wavs = {speaker_2: list(map(preprocess_wav, wav_2_fpaths))}
speaker_1_wavs.update(speaker_2_wavs) 
speaker_wavs = speaker_1_wavs


#embedding speaker
spk_embeds_a = np.array([encoder.embed_speaker(wavs[:len(wavs) // 2]) \
                         for wavs in speaker_wavs.values()])
spk_embeds_b = np.array([encoder.embed_speaker(wavs[len(wavs) // 2:]) \
                         for wavs in speaker_wavs.values()])
spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b)



   
