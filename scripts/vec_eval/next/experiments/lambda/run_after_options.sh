#!/bin/bash
until grep -q "OPTIONS QUEUE DONE" $HOME/vec_eval85/next/options_queue.out 2>/dev/null; do sleep 30; done
cd $HOME/vec_eval85/next && ./run_restore_probe_var.sh inter_object_relation_satisfaction-inside-sunflower-vase-coffee_table \
  "n3idle:3" "n1_sap:1:--physics-override broadphase=SAP" "n4_sap:4:--same-instance-all-slots --physics-override broadphase=SAP" \
  "n1_thr1:1:--physics-override num_threads=1" "n4_thr1:4:--same-instance-all-slots --physics-override num_threads=1"
