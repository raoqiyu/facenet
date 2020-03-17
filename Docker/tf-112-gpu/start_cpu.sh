#!/usr/bin/env bash

################################################################
# Author    Version     Date          Comments
# Xiaolei   v0.1       2017-10-30     Add nvidia and cuda libs
################################################################

# Helper functions
die() {
  echo $@
  exit 1
}

jupyter lab --allow-root "$@"