#!/usr/bin/env bash

rocker --persist-image --ipc=host --x11 --nvidia --name rt-sam2 \
  --port 5000:5000 --devices /dev/video0 --volume .:/home/user/REALTIME_SAM2 -- \
  ghcr.io/robinlabuji/rt-sam2:latest bash

