#!/bin/sh
clear
python3 ~/NoFever/nofever/app_server.py &
sleep 10
python3 ~/NoFever/nofever/main.py &
sleep 60
python3 ~/NoFever/nofever/pid_check.py &
sleep 5
python3 ~/NoFever/nofever/git_scan_pusher.py &

## How to add this bash script to the system's autoboot:
## 1. sudo nano /etc/rc.local
## 2. Add this two lines before last line 'exit 0':
## sleep 30
## sudo -H -u jetson sudo /home/jetson/NoFever/nofever/boot.sh


## Save & close the file.
## This will wait 30 seconds after the system boots up, then will run boot.sh (this file.)