# Data

## Data generation
**Goal:** Generate X training examples.
1 training example = 1 photo containing a racing gate + 1 set of coordinates indicating the position of gate in photo.

<p align="center"><img src="https://github.com/42-AI/42ai_alphapilot_challenge/blob/master/gate_detection/.doc/IMG_0376.JPG" width="70%" height="70%"></p>

## Current generator

**Data generation module features:**
> - Backgrounds
> - Random perspective/position of gate
> - Random light color & position
> - Random blur
> - Random obstacles

## Example of result

<p align="center"><img src="https://github.com/42-AI/42ai_alphapilot_challenge/blob/master/gate_detection/.doc/IMG_CUSTOM.png" width="70%" height="70%"></p>

# Datasets

| NAME | PROVIDER | SIZE | BOUND | PATH |
| --- | --- | --- | --- | --- |
| Training (T) | HEROX | 10k | Inbound | /data/Data_Training |
| Generated (G) | US | 10k | Inbound | /data/Data_Generated |
| OutBound (O) | US | 50k | Outbound | /data/Data_OutBound |
| MultiGate (M) | US | 30k | Outbound | /data/Data_MultiGate |

# Models

## SSD

https://github.com/weiliu89/caffe/tree/ssd

## DSOD

https://github.com/szq0214/DSOD

### To read

https://medium.com/syncedreview/deep-direct-regression-for-multi-oriented-scene-text-detection-4bda214b6ff3

<p align="center"><img src="https://media.giphy.com/media/1eEv6Va9sUS0i7fyO4/giphy.gif" width="50%" height="50%"></p>
