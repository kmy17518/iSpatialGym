#!/bin/bash
until grep -q "CHAIN2 DONE" $HOME/vec_eval85/next/chain2.out 2>/dev/null; do sleep 20; done
cd $HOME/vec_eval85/next && bash run_chain3.sh > chain3.out 2>&1
