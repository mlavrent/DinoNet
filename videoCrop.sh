#!/usr/bin/env bash

filename="${1%.*}"
extension="${1##*.}"
tempfile="$filename-temp.$extension"

ffmpeg -i $1 -filter:v "crop=800:200:556:141" $tempfile
yes | cp -f $tempfile $1
rm -f $tempfile
