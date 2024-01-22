# DL_wireless
This repo contains code for an end-to-end communication system over ADLAM-Pluto SDR.

The pluto_exp7.ipynb contains code for two types of MODCODS in an end-to-end communication system over ADLAM-Pluto SDR.
1. Hamming FEC with 2-point data constellation.
2. Feed forward neural network based MODCODS.

utils.py contains the code for the actual transmission of symbols into Pluto SDR. Refer [PySDR](https://pysdr.org/content/pluto.html) for installation of pyadi drivers. Change the path of pyadi in the code.

fec_blocks.py contains a classical hamming encoder and decoder. 
model.py contains feed-forward architectures that could be used as MODCOD. However, experiments over BER suggest hamming-based system is better than the considered feed-forward neural network based encoders and decoders.
