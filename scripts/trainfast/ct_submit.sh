#!/bin/bash
# trainfast packing test E16 (2026-11-02): collector-on-collector contention on disembedding_03 (the F1 instance, crop4 job), TITAN RTX.
#   ct_submit.sh threads   N collectors x CPUs x PhysX threads (02:15)
#   ct_submit.sh env       2 collectors on 12 CPUs, one part of the job environment separated per process (job_ct.sh MODE)
#   ct_submit.sh pinsize   2-3 collectors on 12 CPUs, each pinned to 2 / 3 / 4 CPUs
#   ct_submit.sh copy      2 collectors on 12 CPUs, each loading its own copy of the template (+ its own job environment)
# Arms <name> <N collectors> <cpus> <PhysX threads> [MODE]; output out/ct/de03/<name>/
F=/vision/u/minyeong/projects/isg_home/trainfast
W=/vision/u/minyeong/projects/ispatialgym-train-fast
O=$F/out
JOB=crop4/disembedding_03/g100336-100359
TRO=inst/disembedding-video_game-video_game_2_board_game_1-desk/office_cubicles_right__support-desk_mdhelw_1-T1__destination-tray-hvlfig__source-tray-uekqey__target-video_game-ceaxsd__distractors-board_game_1-video_game_2/instances/disembedding-video_game-video_game_2_board_game_1-desk__office_cubicles_right__support-desk_mdhelw_1-T1__destination-tray-hvlfig__source-tray-uekqey__target-video_game-ceaxsd__distractors-board_game_1-video_game_2__100352-tro_state.json
case "$1" in
  threads) LIST="n1_c12_t0 1 12 0
n2_c12_t0 2 12 0
n2_c12_t2 2 12 2
n2_c12_t4 2 12 4
n2_c24_t0 2 24 0
n3_c16_t0 3 16 0
n3_c16_t2 3 16 2" ;;
  env) LIST="m_same 2 12 0 same
m_home 2 12 0 home
m_isgcache 2 12 0 isgcache
m_jobid 2 12 0 jobid
m_sepjob 2 12 0 sepjob
m_pin 2 12 0 pin" ;;
  pinsize) LIST="pin2_n2 2 12 0 pin2
pin3_n2 2 12 0 pin3
pin4_n2 2 12 0 pin4
pin3_n3 3 12 0 pin3" ;;
  copy) LIST="m_copy 2 12 0 copy
m_sepjob_copy 2 12 0 sepjob_copy" ;;
  *) echo "usage: ct_submit.sh threads|env|pinsize|copy"; exit 2 ;;
esac
while read name n cpus t mode; do
  out=$O/ct/de03/$name
  [ -e $out ] && { echo "exists $out"; continue; }
  mkdir -p $out
  sbatch --parsable -J isgc_$name -A vision -p svl --gres=gpu:titanrtx:1 -c $cpus --mem=48G --time=01:30:00 \
    $F/scripts/job_ct.sh $O/$JOB $O/$JOB/$TRO $W $out $n $t ${mode:-same} | sed "s|^|SUBMITTED $name n=$n cpus=$cpus T=$t mode=${mode:-same} |"
done <<< "$LIST"
