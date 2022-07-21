from pydub import AudioSegment
from pydub.utils import make_chunks
import os

myaudio = AudioSegment.from_file("/content/Real-Time-Voice-Cloning/output_0.wav" , "wav") 
chunk_length_ms = 12000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
outpath = '/content/drive/MyDrive/Collabera_William/testing_datasets/TEDLIUM_release-3/Tedlium_Conversion_Testing/'

#Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
    chunk_name = os.path.join(outpath+"chunk{0}.wav".format(i))
    print("exporting", chunk_name)
    chunk.export(chunk_name, format="wav")
