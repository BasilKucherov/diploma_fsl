#!/bin/bash
set -e

echo "Testing SimCLR..."
python3 train_miniimagenet.py simclr --debug

echo "Testing VICReg..."
python3 train_miniimagenet.py vicreg --debug

echo "Testing BYOL..."
python3 train_miniimagenet.py byol --debug

echo "Testing SwAV..."
python3 train_miniimagenet.py swav --debug

echo "All tests passed!"

