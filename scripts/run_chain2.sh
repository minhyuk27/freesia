#!/bin/bash
cd ~/freesia
.venv/bin/python -u research/screen2.py "$3" "$2" "$1" > logs/scr_$1.log 2>&1
touch logs/scr_$1.done
