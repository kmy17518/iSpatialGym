#!/bin/bash
until grep -q "VALIDATE DONE" $HOME/vec_eval85/next/validate.out 2>/dev/null; do sleep 20; done
cd $HOME/vec_eval85/next && bash run_control.sh > control.out 2>&1
