#!/bin/bash

f=driver
kernprof -l $f.py
python -m line_profiler $f.py.lprof > profile.txt
rm $f.py.lprof
