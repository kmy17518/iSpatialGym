#!/bin/bash
until grep -q "CHAIN3 DONE" $HOME/vec_eval85/next/chain3.out 2>/dev/null; do sleep 20; done
cd $HOME/vec_eval85/next && bash run_validate.sh > validate.out 2>&1
