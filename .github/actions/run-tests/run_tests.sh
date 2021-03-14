#!/usr/bin/sh

scons
python3 src/run_tests.py --include-system
