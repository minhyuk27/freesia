#!/bin/bash
# 사용: scripts/run_notify.sh <이름> <명령...>  — 명령 종료 후 텔레그램으로 알림.
# logs/<이름>.summary 가 있으면 그 내용(사람이 읽는 결과 요약)을, 없으면 종료 상태만 보낸다(로그 마지막 줄은 보내지 않는다).
cd ~/freesia
name="$1"; shift
start=$(date +%s)
"$@" > "logs/${name}.out" 2>&1; code=$?
dur=$(( $(date +%s) - start ))
if [ -s "logs/${name}.summary" ]; then body="$(cat logs/${name}.summary)"; else body="세부 결과는 로그를 확인하세요."; fi
status="완료"; [ "$code" -ne 0 ] && status="오류 종료(코드 ${code})"
.venv/bin/python -m freesia.notify "[freesia] ${name} ${status} (${dur}초)
${body}" >> logs/notify.log 2>&1
touch "logs/${name}.done"
