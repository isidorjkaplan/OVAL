# OVAL
This is the repository for the project regarding auto encoders for APS360

# Sample Commands

This is for training an offline model. You can use -h for more arguments.   

```python offline.py --cuda --stop=10000 --save_model=data/models/offline.pt --max_frames=500```.     
This runs an offline model to see the video it generates.  

```python convert_video.py data/models/offline.pt data/videos/train/lwt_short.mp4 data/videos/out/lwt_short.mp4 --cuda --max_frames=200```.   
This takes an offline model and retrains it and evaluates it on an online live video feed.    
Watch the video it outputs to see it training in real-time. 

```python online.py --video=data/videos/train/lwt_short.mp4 --stop=300 --cuda --load_model=data/models/offline.pt --out=data/videos/out/online.mp4```.  


# Task Board

| Task Name  |  Notes | Assigned | Time (hr)  | Due  | Status (Completed, Pending, Active)  | 
|---|---|---|---|---|---|
| Project Idea  |   | Isidor  | 3  |  Jan 27 | Completed |
| Architecture Design  |   | Isidor  | 3  |  Jan 27 | Completed |
| Formal Preposal  |   | Adam  | 8 |  Feb 3 | Completed Feb 3 |
| Online Code |   | Isidor  | 8  |  Feb 12 | Completed Feb 1 |
| Offline Code |   | Isidor  | 6  |  Feb 12 | Completed Feb 9 |
| Model Video Evaluator |   | Isidor  | 2  |  Feb 12 | Completed Feb 9 |
| Model V1 |   | Isidor  | 3  |  Feb 12 | Completed Feb 9 |
| Model V2 |   | Ryan  | 3  |  Feb 12  | Active |
| Primitive Data Loader |   | Isidor  | 1  |  Feb 12  | Completed Feb 9 |
| Tuning | Tune model and all related settings to try and achieve a good result  | Khantil  | 3  |  -  | Active |
| Enhanced Data Loader | Must support multi-threaded pre-loading of frames to avoid stalling critical path  | Khantil  | 3  |  Feb 15  | Active |
| Noise | Add guassian noise for both offline and online training. DO NOT use noise in evaluation for offline or online.  |   | 1  |  -  | Active |
| Solve Online Memory Leak |  Causing online to crash after long training sessions | Adam  | 6  |  Feb 15  | Completed |
| Color Spaces | Tennative work to try fully switching which color space is used for offline | Isidor  | 3  |  Feb 17 | Complete |
| Project Management | Organizing and directing tasks. Peer-reviewing code. |   | 3  | - | Active |
| Color Spaces (Model Out) | What should model try to predict. Will help with the loss function and black/white problem. Will need to simply add a function that we call on the input image before passing into the loss with the model. Also will need to adjust the video generators to convert back to RGB before saving. This could potentially be internal to the model, it is a design desicion. | Khantil | 1  |  - | Completed |
| Color Spaces (Model In) |  We can try to perhaps show the model the image in all three color spaces. This is unrelated to the color space it predicts in. We could also make this internal to the model or the data loader, where we give the model an image in one but it converts into other color spaces internally. Either that or we could do it in the data loader where we return three tensors, one for each color space.   | Khantil  | 1  |  - | Completed |
| Transfer Learning | Try to do some tests using a variety of different pre-trained models. In some cases we may just try to explicitly use their model, whereas in others we may try to include their model in a larger one with some additonal layers.   | Ryan  | 8  |  - | Pending Model V2 |
| Conv LSTM |  It would be really helpful if we could apply a conv LSTM to both encoder and decoder. This will allow it to have memory. This could work in collaberatin with transfer learning where we apply the conv lstm at the end on it's extracted features.   | Ryan  | 2  |  -  | Pending Model V2 |
| Detailed Tensorboard Statistics |  We need better information about how our model is performing. Include a bunch more statistics into both offline and online (primarilly offline). We need things such as the compression ratios it achieves, the ratio of error to compression factor, etc etc. Lots of useful metrics and we need to see them. | Adam  | 3  |  Feb. 21 | Active |
| Hyperparameter Script | Write a script in bash or python that calls the offline and video conversion scripts. It should call them using the command line and pass them hyperparameters for things like batch, learning rate, network parameters, etc etc. Basically anything we have an option (including the new things like color space that have yet to be added) and it should easily support them. It should be rohbust to chanigng the existing hyperparameters. Any command line argument we add should be a quick tweek to this script. It will run a large number of models, save each of them, and run their video conversions. At the end a human can then look at the tensorboard and see all the statistics it generates and decide which models are worth taking to online learning. Can also examine the videos. This script will be essential for us to have, without it we are flying blind. Once we add in the ability to compare all the various tasks above this script is what allows us to evaluate which combination of them is the best. | Adam  | 4  |  Feb. 21 | Active |
| Variable Encoding | Gotta push this way down the line. If we can get a good conv model that produces a reasonable encoding and that encoding is on the larger side then we can use this to basically act as a variable sized autoencoder just on that last few layers which we will use to compress its already compressed result a variable amount. For a reward function I was thinking we try to maximize some ratio of compression factor and bit size. AKA, maximize quality per bits.   | -  | 5  |  - | Pending good results |
