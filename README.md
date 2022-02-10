# OVAL
This is the repository for the project regarding auto encoders for APS360

# Sample Commands

This is for training an offline model. You can use -h for more arguments. 
`python offline.py --cuda --stop=10000 --save_model=data/models/offline.pt --max_frames=500`
This runs an offline model to see the video it generates
`python convert_video.py data/models/offline.pt data/videos/train/lwt_short.mp4 data/videos/out/lwt_short.mp4 --cuda --max_frames=200`
This takes an offline model and retrains it and evaluates it on an online live video feed.   
Watch the video it outputs to see it training in real-time
`python online.py --video=data/videos/train/lwt_short.mp4 --stop=300 --cuda --load_model=data/models/offline.pt --out=data/videos/out/online.mp4`

