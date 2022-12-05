#!/bin/bash

MY_BUCKET=init_test_456789123
MOUNT_DIR=/home/jupyter/bucket

# mount dataset bucket to directory
gcsfuse --implicit-dirs --rename-dir-limit=100 --disable-http2 --max-conns-per-host=100 $MY_BUCKET $MOUNT_DIR
