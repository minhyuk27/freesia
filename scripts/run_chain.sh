#!/bin/bash
# 사용: run_chain.sh <tag> <tf> <families>  (bars 준비 후 스크리닝)
cd ~/freesia
.venv/bin/python -u research/build_bars.py > logs/scr_bars_$1.log 2>&1
.venv/bin/python -u research/screen.py "$3" "$2" "$1" > logs/scr_$1.log 2>&1
touch logs/scr_$1.done
