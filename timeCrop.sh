#!/usr/bin/env bash

filename="${1%.*}"
extension="${1##*.}"
tempfile="$filename-temp.$extension"

ffmpeg -i $1 -ss $2 -to $3 $tempfile
yes | cp -f $tempfile $1
rm -f $tempfile
