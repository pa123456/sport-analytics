#!/bin/bash

python3 main.py --side uploads/side_shot.mp4 --front uploads/front_shot.mp4
ffmpeg -i ./processed/front.mp4 -an -vcodec libx264 -crf 23 ./processed/front-edit.mp4
rm ./processed/front.mp4
mv ./processed/front-edit.mp4 ./processed/front.mp4
ffmpeg -i ./processed/side.mp4 -an -vcodec libx264 -crf 23 ./processed/side-edit.mp4
rm ./processed/side.mp4
mv ./processed/side-edit.mp4 ./processed/side.mp4
