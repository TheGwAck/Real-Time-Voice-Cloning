import sys 
import librosa
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
import itertools

from sklearn.preprocessing import normalize
from encoder.audio import preprocess_wav
import encoder.inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from similarity import plot_similarity_matrix, plot_histograms

from utils.argutils import print_args
from itertools import groupby
import pickle

model = sys.argv[1]
path = sys.argv[2]
pkl_name = sys.argv[3]
threshold_min = float(sys.argv[4])
if len(sys.argv) > 5:        
        threshold_max = float(sys.argv[5])
        threshold_step = float(sys.argv[6])

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

_model = encoder.load_model(model)
#preprocessing wavs
pkl_fpath = f'/content/drive/MyDrive/Collabera_William/computation_wavs/speaker_wavs_{pkl_name}.pkl'
if Path(pkl_fpath).exists():
        print(f'Audio files being loaded from {pkl_fpath}')
        speaker_wavs = pd.read_pickle(pkl_fpath)
else:

        wav_fpaths = sorted(list(Path(path).glob("**/**/*.wav")))

        # Group the wavs per speaker and load them using the preprocessing function provided with 
        # resemblyzer to load wavs in memory. It normalizes the volume, trims long silences and resamples 
        # the wav to the correct sampling rate.
        speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                        groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"), 
                                lambda wav_fpath: wav_fpath.parent.stem)}

        # save dictionary to pickle file
        with open(pkl_fpath, 'wb') as file:
                pickle.dump(speaker_wavs, file, protocol=pickle.HIGHEST_PROTOCOL)

# embedding speaker
spk_embeds = np.array([encoder.embed_speaker(speaker_wavs[speaker]) for speaker in tqdm(speaker_wavs, "Embedding", len(speaker_wavs), unit='wavs')])
spk_embeds = torch.from_numpy(spk_embeds)
sim_matrix = _model.similarity_matrix(spk_embeds)
sim_matrix = torch.mean(sim_matrix, dim = 1).detach().numpy()
sim_matrix = normalize(sim_matrix, axis=1, norm='max')

'''Create a pandas dataframe with columns:
        -speaker-a-n
        -speaker-b-k
        -similarity_score
        -match: binary value determining whether the similarity score yields a match determined by threshold
        -same: binary value whether speaker-a and speaker-b are the same
        -correct: binary value if match and same are equal --> 1 else 0
        add metrics for speaker-a-n and speaker-b-k
   '''     
def create_panda_cols(x, sim_matrix, threshold):
 
        similarity = sim_matrix[x['speaker_a_indice'],x['speaker_b_indice']]
        if x['speaker_a'].split('_')[0] == x['speaker_b'].split('_')[0]:
                same = 1
        else:
                same = 0
        if similarity >= threshold and same == 1:
                correct = 'true positive'
        elif similarity < threshold and same == 0:
                correct = 'true negative'
        elif similarity >= threshold and same == 0:
                correct = 'false positive'
        else:
                correct = 'false negative'
        return (similarity, same, correct)

def get_pandas(sim_matrix, speaker_wavs, threshold):
        
        speakers = [i for i in speaker_wavs.keys()]
        combos = list(combinations(range(len(speakers)), 2))
        speaker_combo_indices = [((i,j),(j,i)) for (i,j) in combos]
        speaker_combo_indices_redundant = list(itertools.chain(*speaker_combo_indices))
        speakers_and_indices = [(i, speakers[i], j, speakers[j]) for (i,j) in speaker_combo_indices_redundant]
        df = pd.DataFrame(speakers_and_indices, columns = ['speaker_a_indice','speaker_a', 'speaker_b_indice', 'speaker_b'])
        df[['similarity','same', 'correct']] = df.apply(lambda x: create_panda_cols(x, sim_matrix, threshold), axis=1, result_type = 'expand')
       
        return df

# df = get_pandas(sim_matrix, speaker_wavs, threshold)
# df.to_pickle('/content/drive/MyDrive/Collabera_William/similarity' + pkl_name +'.pkl')
if threshold_max:
        thresholds = np.arange(threshold_min,threshold_max, threshold_step)
        print(f'Creating pandas similarity tables with thresholds from {threshold_min} to {threshold_max} with a step of {threshold_step}.')
else: 
        thresholds = threshold_min
        print(f'Creating pandas similarity table with threshold of {threshold_min}')


for thresh in thresholds:
        df = get_pandas(sim_matrix, speaker_wavs, thresh)
        df.to_pickle('/content/drive/MyDrive/Collabera_William/similarity/similarity' + str(thresh)+'.pkl')

# Draw the plots
fix, axs = plt.subplots(1, 2, figsize=(8, 5))

labels_a = ["%s-A" % i for i in speaker_wavs.keys()]
labels_b = ["%s-B" % i for i in speaker_wavs.keys()]
mask = np.eye(len(sim_matrix), dtype=np.bool_)
plot_similarity_matrix(sim_matrix, labels_a, labels_b, axs[0],
                       "Cross-similarity between speakers")
plot_histograms((sim_matrix[mask], sim_matrix[np.logical_not(mask)]), axs[1],
                "Normalized histogram of similarity\nvalues between speakers")
plt.show()