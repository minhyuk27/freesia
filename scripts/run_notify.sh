#!/bin/bash
# 사용: scripts/run_notify.sh <이름> <명령...>  — 명령 종료 후 텔레그램으로 결과 알림(실패 사유는 logs/notify.log에 남김)
cd ~/freesia
name="$1"; shift
start=$(date +%s)
"$@" > "logs/${name}.out" 2>&1; code=$?
dur=$(( $(date +%s) - start ))
msg="freesia ${name} 종료(코드 ${code}, ${dur}초). $(tail -n 1 logs/${name}.out | cut -c1-120)"
.venv/bin/python -m freesia.notify "$msg" >> logs/notify.log 2>&1
touch "logs/${name}.done"
