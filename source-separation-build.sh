#!/bin/bash

git checkout tags/SpeakerSeparation
docker build -t magnolia-demo-r1 -f Dockerfile_copy .
git checkout origin
