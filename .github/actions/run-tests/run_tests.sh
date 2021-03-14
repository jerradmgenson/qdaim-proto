#!/usr/bin/sh

scons
python src/run_tests.py --include-system
