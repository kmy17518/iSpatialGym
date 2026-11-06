#!/bin/bash
# trainfast follow-up F1 (owner item 4, 2026-11-01): CPU vs GPU PhysX for a single-scene physics-only collector, one instance per job,
# alone on a TITAN RTX; each instance twice per arm (run-to-run noise). Instances: the median-length successful demo of a finished job
# (disembedding_03 crop4, mental_rotation-mug crop4, ontop_small_01 refine_nearest). Output: out/gpudyn/<short>/<cpu|gpu>_r<k>/
F=/vision/u/minyeong/projects/isg_home/trainfast
W=/vision/u/minyeong/projects/ispatialgym-train-fast
O=$F/out
while read short job tro; do
  for arm in cpu gpu; do
    G=0; [ $arm = gpu ] && G=1
    for r in 1 2; do
      out=$O/gpudyn/$short/${arm}_r$r
      [ -e $out ] && { echo "exists $out"; continue; }
      mkdir -p $out
      sbatch --parsable -J isgd_${short} -A vision -p svl --gres=gpu:titanrtx:1 -c 12 --mem=32G --time=01:30:00 \
        $F/scripts/job_gpu_dyn.sh $O/$job $O/$job/${tro} $W $out $G | sed "s|^|SUBMITTED $short $arm r$r |"
    done
  done
done <<LIST
de03 crop4/disembedding_03/g100336-100359 inst/disembedding-video_game-video_game_2_board_game_1-desk/office_cubicles_right__support-desk_mdhelw_1-T1__destination-tray-hvlfig__source-tray-uekqey__target-video_game-ceaxsd__distractors-board_game_1-video_game_2/instances/disembedding-video_game-video_game_2_board_game_1-desk__office_cubicles_right__support-desk_mdhelw_1-T1__destination-tray-hvlfig__source-tray-uekqey__target-video_game-ceaxsd__distractors-board_game_1-video_game_2__100352-tro_state.json
mrmug crop4/mental_rotation-mug/g100240-100263 inst/mental_rotation-mug/Rs_int__room-living_room_0__target-mug-ppzttc/instances/mental_rotation-mug__Rs_int__room-living_room_0__target-mug-ppzttc__100250-tro_state.json
ontop01 refine_nearest/inter_object_relation_satisfaction-ontop_small_scale_01/g100417-100440 inst/inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table/house_single_floor__support-breakfast_table_rhjoby_0-T1__partner-platter-csanbr__target-pack_of_kielbasa-dgodnn/instances/inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table__house_single_floor__support-breakfast_table_rhjoby_0-T1__partner-platter-csanbr__target-pack_of_kielbasa-dgodnn__100437-tro_state.json
LIST
