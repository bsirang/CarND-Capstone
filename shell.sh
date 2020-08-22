#!/bin/bash

# Drop into a shell in the docker container...
docker run --network host -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone

