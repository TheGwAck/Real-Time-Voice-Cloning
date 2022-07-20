'''

Index of all similarity-related code in Resemblyzer and Real-Time-Voice-Cloning repos:

    - Real-Time-Voice-Cloning
        - encoder/model.py:
            def similarity_matrix(self, embeds):
                
    - Resemblyzer
        - demo_utils.py:
            def plot_similarity_matrix(matrix, labels_a=None, labels_b=None, ax: plt.Axes=None, title=""):
        - demo01_similarity.py and similarity_folder.py:
            # Compute the similarity matrix. The similarity of two embeddings is simply their dot 
            # product, because the similarity metric is the cosine similarity and the embeddings are 
            # already L2-normed.
            # Short version:
            utt_sim_matrix = np.inner(embeds_a, embeds_b)
            # Long, detailed version:
            utt_sim_matrix2 = np.zeros((len(embeds_a), len(embeds_b)))
            for i in range(len(embeds_a)):
                for j in range(len(embeds_b)):
                    # The @ notation is exactly equivalent to np.dot(embeds_a[i], embeds_b[i])
                    utt_sim_matrix2[i, j] = embeds_a[i] @ embeds_b[j]
            assert np.allclose(utt_sim_matrix, utt_sim_matrix2)


'''
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import torch

def similarity(embeds_a, embeds_b):
    # NEEDS EXTRA THINGS FROM SIMILARITY_FOLDER.py
# Compute the similarity matrix. The similarity of two embeddings is simply their dot 
            # product, because the similarity metric is the cosine similarity and the embeddings are 
            # already L2-normed.
            # Short version:
            utt_sim_matrix = np.inner(embeds_a, embeds_b)
            # Long, detailed version:
            utt_sim_matrix2 = np.zeros((len(embeds_a), len(embeds_b)))
            for i in range(len(embeds_a)):
                for j in range(len(embeds_b)):
                    # The @ notation is exactly equivalent to np.dot(embeds_a[i], embeds_b[i])
                    utt_sim_matrix2[i, j] = embeds_a[i] @ embeds_b[j]
            assert np.allclose(utt_sim_matrix, utt_sim_matrix2)
            ## Similarity between two speaker embeddings
            # Divide the utterances of each speaker in groups of identical size and embed each group as a
            # speaker embedding
            spk_embeds_a = np.array([encoder.embed_speaker(wavs[:len(wavs) // 2]) \
                                    for wavs in speaker_wavs.values()])
            spk_embeds_b = np.array([encoder.embed_speaker(wavs[len(wavs) // 2:]) \
                                    for wavs in speaker_wavs.values()])
            spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b)

            return spk_sim_matrix, utt_sim_matrix

# def similarity_matrix(self, embeds):
#     """
#     Computes the similarity matrix according the section 2.1 of GE2E.

#     :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
#     utterances_per_speaker, embedding_size)
#     :return: the similarity matrix as a tensor of shape (speakers_per_batch,
#     utterances_per_speaker, speakers_per_batch)
#     """
#     speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
    
#     # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
#     centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
#     centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

#     # Exclusive centroids (1 per utterance)
#     centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
#     centroids_excl /= (utterances_per_speaker - 1)
#     centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

#     # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
#     # product of these vectors (which is just an element-wise multiplication reduced by a sum).
#     # We vectorize the computation for efficiency.
#     sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
#                                 speakers_per_batch).to(self.loss_device)
#     mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
#     for j in range(speakers_per_batch):
#         mask = np.where(mask_matrix[j])[0]
#         print(mask)
#         sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
#         print('First sim_matrix: ', sim_matrix)
#         sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
#         print('Second sim_matrix: ', sim_matrix)

#     sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
#     return sim_matrix

def plot_similarity_matrix(matrix, labels_a=None, labels_b=None, ax: plt.Axes=None, title=""):
    if ax is None:
        _, ax = plt.subplots()
    fig = plt.gcf()
        
    img = ax.matshow(matrix, extent=(-0.5, matrix.shape[0] - 0.5, 
                                    -0.5, matrix.shape[1] - 0.5))

    ax.xaxis.set_ticks_position("bottom")
    if labels_a is not None:
        ax.set_xticks(range(len(labels_a)))
        ax.set_xticklabels(labels_a, rotation=90)
    if labels_b is not None:
        ax.set_yticks(range(len(labels_b)))
        ax.set_yticklabels(labels_b[::-1])  # Upper origin -> reverse y axis
    ax.set_title(title)

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    fig.colorbar(img, cax=cax, ticks=np.linspace(0.4, 1, 7))
    img.set_clim(0.4, 1)
    img.set_cmap("inferno")
    
    return ax

def plot_histograms(all_samples, ax=None, title=""):
    """
    Plots (possibly) overlapping histograms and their median 
    """
    _default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if ax is None:
        _, ax = plt.subplots()
    
    for samples, color in zip(all_samples, _default_colors):
        ax.hist(samples, density=True, color=color + "80")
    ax.legend()
    ax.set_xlim(0.35, 1)
    ax.set_yticks([])
    ax.set_title(title)
        
    ylim = ax.get_ylim()
    ax.set_ylim(*ylim)      # Yeah, I know
    for samples, color in zip(all_samples, _default_colors):
        median = np.median(samples)
        ax.vlines(median, *ylim, color, "dashed")
        ax.text(median, ylim[1] * 0.15, "median", rotation=270, color=color)
    
    return ax


# def plot_histograms(all_samples, ax=None, names=None, title=""):
#     """
#     Plots (possibly) overlapping histograms and their median 
#     """
#     _default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     if ax is None:
#         _, ax = plt.subplots()
    
#     for samples, color, name in zip(all_samples, _default_colors, names):
#         ax.hist(samples, density=True, color=color + "80", label=name)
#     ax.legend()
#     ax.set_xlim(0.35, 1)
#     ax.set_yticks([])
#     ax.set_title(title)
        
#     ylim = ax.get_ylim()
#     ax.set_ylim(*ylim)      # Yeah, I know
#     for samples, color in zip(all_samples, _default_colors):
#         median = np.median(samples)
#         ax.vlines(median, *ylim, color, "dashed")
#         ax.text(median, ylim[1] * 0.15, "median", rotation=270, color=color)
    
#     return ax
