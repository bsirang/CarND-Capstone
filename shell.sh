#!/bin/bash

# Drop into a shell in the docker container...
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone

