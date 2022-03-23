#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
FILES="data/videos/test/*"
for f in $FILES
do
    NAME="$(basename $f)"
    echo "Processing $NAME file..."
    python online.py --load_model=data/models/offline8.pt --video=$f --out=data/videos/out/online_noupdate_$NAME --cuda --update_err=1000
    python online.py --load_model=data/models/offline8.pt --video=$f --out=data/videos/out/online_withupdate_$NAME --cuda --update_err=0.01
done