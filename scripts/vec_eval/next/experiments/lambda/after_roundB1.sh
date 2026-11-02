#!/bin/bash
until grep -q "ROUNDB_LAMBDA DONE" $HOME/vec_eval85/next/roundB_lambda.out 2>/dev/null; do sleep 30; done
cd $HOME/vec_eval85/next && bash run_roundB_lambda.sh mental_rotation-carving_knife mental_rotation-classroom_mock_up mental_rotation-crab mental_rotation-hook mental_rotation-mug mental_rotation-soup_ladle mental_rotation-statue mental_rotation-teaspoon mental_rotation-toy_figure mental_rotation-walker > roundB_lambda2.out 2>&1
