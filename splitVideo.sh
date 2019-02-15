#!/usr/bin/env bash

foldername="${1%%.*}"
foldername="${foldername##*/}"

mkdir "$2$foldername/"

ffmpeg -i $1 -vf scale=120:30 -r 10 "$2$foldername/%04d.jpeg"
