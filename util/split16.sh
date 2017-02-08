#!/bin/bash

if [ $# -ne 1 ]; then
  echo " "
  echo "./split16.sh requires the original wavfile name"
  echo " "
  echo "USAGE ./split16.sh <filename.wav> "
  echo " "
  echo "OUTPUT: "
  echo "       filename-1.wav"
  echo "       filename-2.wav"
  echo "          ..."
  echo "       filename-16.wav"
  echo " "
  exit
fi
wavfile=$1

for i in `seq 1 16`; do
  splitfile=${wavfile%.*}
  echo sox $wavfile $splitfile-$i.wav remix $i
  sox $wavfile $splitfile-$i.wav remix $i
done
