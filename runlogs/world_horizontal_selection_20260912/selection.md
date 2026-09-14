# World-horizontal object selection (rounds 1-3)

Scores: `2V + 2L + H` (max 18; contract `review_contract.md`): V visual, L level readability, H hangability. valid = V >= 3, L >= 2, H >= 1. `cat score` = mean of the two best valid models (single valid model: score - 2; weak verdict: - 1). Frames: `front` = axis out of the wall, `level` = axis along the wall when aligned, `up` = axis pointing up when hung, `thin` = thinnest bbox axis; `symmetric 180` = looks the same upside down (the sampler adds the half turn about the front axis to the goal's orientation equivalences).

## world_alignment-2 (wa2)

### round 1 (8 categories)

| category | cat score | valid / reviewed / ready | picks (score) | alternate | frames of the picks | used by other tasks | verdict | note |
|---|---|---|---|---|---|---|---|---|
| rolling_pin | 17.0 | 2 / 2 / 2 | mwrdae (18), smnklu (16) | - | mwrdae: front +x, level y, up +z, thin x, symmetric 180; smnklu: front +x, level y, up +z, thin x, symmetric 180 | - | good | Two clean wooden rolling pins, 0.40-0.42 m long, 5.5-6 cm barrels, handles both ends. Rod reads level at a glance, natural on kitchen wall hooks; front face arbitrary. |
| baseball_bat | 16.0 | 2 / 2 / 2 | yalgei (16), spcxrt (16) | - | yalgei: front -y, level x, up +z, thin y; spcxrt: front -y, level x, up +z, thin y | - | good | Two clean wooden bats 0.76-0.81 m, round 5-7 cm, lying along x; a bar that reads level instantly, either end left is fine; no logo, so displayed face is arbitrary. |
| shelf | 15.0 | 5 / 13 / 24 | xogqqm (15), qrmiur (15) | iruewm | xogqqm: front -z, level y, up +x, thin z; qrmiur: front +z, level y, up +x, thin z, symmetric 180 | - | good | Mixed category: thin planks 0.5-1.2 m read level well; four deep 0.29 m cubby boxes, two tiny L-shaped corner pieces, an oversize tray and a thick bracket shelf fail hangability. apiaid unrendered. |
| radiator | 14.0 | 5 / 11 / 15 | arfdpk (14), arsjbe (13) | dwmloz | arfdpk: front +x, level y, up +z, thin x; arsjbe: front +x, level y, up +z, thin x | - | good | Panel radiators (0.36-1.27 m wide, <=0.07 m thick) hang like landscape boards with a clear long top edge; six of eleven are cast-iron column floor units 0.18-0.28 m deep and cannot hang flat. Large panels are borderline wide. |
| drumstick | 14.0 | 2 / 2 / 2 | jauzlc (15), lkvzmj (15) | - | jauzlc: front +z, level x, up +y, thin z; lkvzmj: front +z, level x, up +y, thin z, symmetric 180 | - | weak | Thin wooden rods 0.38-0.41 m long, 1.2-1.5 cm thick. Level axis is unambiguous, but the sticks are hairline-thin, round and not typical wall items; hard to see and push. |
| wall_mounted_tv | 13.5 | 4 / 9 / 15 | ggfznh (14), dxyfxf (12) | ylvjhb | ggfznh: front +x, level y, up +z, thin x, symmetric 180; dxyfxf: front +x, level y, up +z, thin x, symmetric 180 | - | good | Flat 16:9 panels 0.56-1.32 m wide, all blank black/grey screens except bfehsl; long edge is the obvious level. Two duplicate pairs (leeqgt/ovzajm, dxyfxf/sodbbr); 1.3 m models and the deep pole-mounted bfehsl fail hangability. |
| umbrella | 13.0 | 5 / 5 / 5 | frpfnc (13), hdfkbc (13) | ziqzqo | frpfnc: front +z, level x, up +y, thin z; hdfkbc: front +z, level x, up +y, thin z | world_alignment-1 r2 | good | Closed umbrellas 0.5-1.0 m, round 7-10 cm bodies, hook or straight handles at one end. Long axis reads level clearly, but umbrellas naturally hang vertically and front face is arbitrary; all H=1. |
| whiteboard | 12.5 | 2 / 3 / 3 | bqqmtc (14), gvkgsi (11) | - | bqqmtc: front +x, level y, up +z, thin x; gvkgsi: front +x, level y, up +z, thin x | world_alignment-1 r3 | good | Landscape boards with a bottom marker tray giving a clear level and up cue. Only bqqmtc has drawn content; gvkgsi is a plain hollow-backed 1.23 m board, onhspt is a 3.2 m classroom board. |

### round 2 (5 categories)

| category | cat score | valid / reviewed / ready | picks (score) | alternate | frames of the picks | used by other tasks | verdict | note |
|---|---|---|---|---|---|---|---|---|
| license_plate | 18.0 | 5 / 5 / 5 | hjdaqn (18), nwfbxn (18) | yepdrj | hjdaqn: front +z, level x, up +y, thin z; nwfbxn: front +z, level x, up +y, thin z | configuration_matching-3 r1, world_alignment-1 r2 | good | Standard US plates, all 0.308 x 0.155 x 0.004 m with identical geometry and complete printed textures; long edge plus text baseline give an unmistakable level; common garage or bar wall decor; models differ only in print. |
| picture | 16.0 | 8 / 12 / 46 | chfljy (16), qavxsz (16) | wfdvzv | chfljy: front +z, level x, up +y, thin z; qavxsz: front +z, level x, up +y, thin z | world_alignment-1 r1 | good | Framed photos and canvases lying flat (thin z, image on +z). Mostly portrait 0.46x0.69 m where the short edge is level and must be read from content; three landscape 0.9-1.0 m; one blank, one chalkboard. |
| signpost | 16.0 | 6 / 7 / 7 | vwhtlm (16), vvjtkv (16) | ivvjfv | vwhtlm: front +x, level y, up +z, thin x; vvjtkv: front +x, level y, up +z, thin x | world_alignment-1 r1 | good | Indoor safety signs: green exit boards 0.41-0.63 m wide with strong text/pictogram level cues, plus 0.16-0.23 m fire-extinguisher and notice plaques that are borderline small and near-square; sdsfzw is a zero-thickness square sheet. |
| towel_rack | 16.0 | 6 / 6 / 6 | kqrmrh (16), crivoa (16) | pqhsry | kqrmrh: front +x, level y, up +z, thin x, symmetric 180; crivoa: front +x, level y, up +z, thin x | world_alignment-1 r1 | good | Mix of ~0.47 m wall bars/hook rails (best) and ladder-style heated rails 0.8-1.7 m tall; bars read level at a glance, ladders rely on rungs and are 0.14 m deep. |
| hanger | 15.0 | 6 / 6 / 6 | agrpio (15), nadesv (15) | lymqks | agrpio: front +z, level y, up -x, thin z; nadesv: front +z, level y, up -x, thin z | world_alignment-1 r1 | good | Six hangers 0.43-0.47 m wide lying in the x-y plane, hook toward -x; barred ones read level well, bar-less ones subtler; hanging flat on a wall is unusual (H=1). |

### round 3 (5 categories)

| category | cat score | valid / reviewed / ready | picks (score) | alternate | frames of the picks | used by other tasks | verdict | note |
|---|---|---|---|---|---|---|---|---|
| decorative_sign | 15.0 | 3 / 3 / 3 | rypeha (16), hgonwj (14) | xzsmxq | rypeha: front +x, level y, up +z, thin x; hgonwj: front +x, level y, up +z, thin x, symmetric 180 | world_alignment-1 r1 | good | Cut-out letter and word wall signs (black Z, teal LOVE, teal Welcome HOME), 0.33-0.93 m wide, 0.02-0.07 m thick; text baselines or letter bars give the level; single letter Z has a 90-deg Z/N ambiguity. |
| guitar | 14.0 | 3 / 3 / 3 | cybisc (15), hpqjrq (13) | gysapt | cybisc: front +z, level x, up +y, thin z; hpqjrq: front +z, level x, up +y, thin z | world_alignment-1 r1 | good | Three electric guitars 1.04 m long, 0.26-0.34 wide, 3.5-6.7 cm thick, lying along x; the long neck gives an instant level cue; all just over the 0.90 m band, so H borderline. |
| ruler | 14.0 | 3 / 3 / 5 | rutqab (15), ujizkx (15) | vmnaih | rutqab: front +z, level y, up +x, thin z; ujizkx: front +z, level y, up +x, thin z | inter_object_relation_satisfaction-3 r1 | weak | Three 0.33 m rulers, 2.6-2.8 cm wide: one triangular scale, two 1 mm steel rules. Strip reads level instantly, but strips are narrow, near-zero thickness and unusual as horizontal wall items. |
| address | 14.0 | 1 / 1 / 1 | pbfbxj (16) | - | pbfbxj: front +x, level y, up +z, thin x | - | weak | Single model: a landscape wooden name plaque 0.58 m wide, 3.6 cm thick, text upright with +z; ideal wall sign but only one candidate. |
| axe | 14.0 | 1 / 1 / 1 | plqyos (16) | - | plqyos: front +z, level x, up +y, thin z | - | weak | Single decorative axe, 0.54 m handle with head at one end; flat and workshop-wall natural, level handle reads clearly. Only one model, so weak by count. |

### not selected

| category | cat score | valid / reviewed / ready | verdict | used by other tasks | note |
|---|---|---|---|---|---|
| shovel | 14.0 | 1 / 1 / 1 | weak | - | Single compact D-handle spade, 0.74 m, blade 15 cm wide. Natural garage-wall item, shaft is an obvious level bar, symmetric about it; only one model, so weak by count. |
| tennis_racket | 14.0 | 1 / 1 / 1 | weak | - | Only one model but a strong one: 0.82 x 0.34 x 0.03 m, 2.4:1 handle-plus-oval silhouette reads level at a glance, thin, common on wall hooks; weak solely for lack of a second model. |
| bookcase_shelf | 13.0 | 1 / 1 / 1 | weak | - | One dark wooden plank 0.74 m long, 2.8:1, thin and clean; excellent level cue but a bare shelf board is an unusual wall item. Single model only, so weak. |
| bulletin_board | 13.0 | 1 / 2 / 2 | weak | - | Two cork boards about 1.08 x 0.75 m lying face-up in canonical frame (front +z). Aspect 1.4 is too square for a silhouette cue; only the pinned-content board reads level. Both exceed 0.90 m. |
| handle | 13.0 | 1 / 1 / 1 | weak | - | Single model: a 0.62 m black grab bar on end feet; bar reads level instantly but depth 0.13 m is borderline and only one candidate. |
| hook_rack | 13.0 | 1 / 1 / 1 | weak | - | Single model: a 0.96 m thin black hook rail with hanging hooks; reads level instantly but only one candidate and length just over the 0.9 m limit. |
| longboard | 13.0 | 1 / 1 / 1 | weak | - | Single complete longboard 1.04 x 0.33 x 0.15 m with wheels; 3:1 outline reads level instantly, wall display is natural, but length and depth sit at the borderline. |
| rake | 13.0 | 1 / 2 / 2 | weak | - | Long garden rakes, 1.0-1.4 m handles with heads at one end. Handle reads level, but sizes sit at or beyond the 1.3 m limit; only the flat fan rake is usable. |
| trumpet | 13.0 | 1 / 1 / 1 | weak | - | Single detailed trumpet 0.91 x 0.25 x 0.16 m; parallel tubing gives a strong level line and trumpets hang on wall hooks, but depth and length both slightly exceed the bands. |
| painting | 12.5 | 2 / 6 / 36 | weak | world_alignment-1 r1 | Framed/unframed landscape canvases 0.3-1.1 m wide, stood upright in canonical frame (thin axis x, front +x). Most have aspect ~1.4 with weak content, so the level edge is marginal; only two read clearly. |
| poster | 12.0 | 1 / 1 / 1 | weak | world_alignment-1 r3 | Single model only: a framed portrait movie poster, 1 cm thick, 0.62 m wide. Level comes from the short edge plus text and figures; good item but the category offers no second model. |
| wall_mounted_light | 12.0 | 3 / 6 / 6 | weak | - | Six grey/black fixtures, all untextured-looking; only the two near-identical 0.62 m track lights (0.2 m deep, over the flat limit) and a tube sconce read level; the square lamp, angled two-spot and swing-arm lamp do not. (override: track lights are 0.20 m deep (contract: H = 0 above 0.15 m); only the tube sconce remains valid, < 2 models; not selected) |
| board | 11.0 | 1 / 1 / 1 | weak | - | Single plywood-like sheet 0.74x1.12x0.014 m, clean and textured; aspect ~1.5 makes the level edge subtle and 1.12 m long edge is borderline for the camera. Only one model. |
| golf_club | 11.0 | 1 / 1 / 1 | weak | - | Single all-black iron 1.0 m long, 3 cm thick; straight shaft reads level well but is a thin dark line, head is small; length at the borderline, COM near head. |
| poker | 11.0 | 1 / 1 / 1 | weak | - | One thin black poker, 0.66 m, hook and grip give left/right cue. Rod reads level easily, yet pokers normally hang vertically and the 3 cm round shaft is a small target. |
| skateboard_deck | 11.0 | 1 / 1 / 1 | weak | configuration_matching-3 r1, configuration_matching-3 r3 | Single black deck with trucks, 1.04 x 0.26 x 0.11 m; 4:1 board reads level at a glance and boards hang on wall racks, but length and thickness are both borderline. |
| violin | 11.0 | 1 / 1 / 1 | weak | - | Single good-looking violin 0.60 x 0.21 x 0.10 m; the short neck and curvy body make the horizontal subtler than a bar, and violins normally hang scroll-up; thickness borderline. |
| air_conditioner | -1.0 | 0 / 1 / 1 | drop | - | Single model: a deep (0.30 m) wall-mounted split AC indoor unit. Strong horizontal from the long vent slot, but the body is boxy and cannot hang flat against a wall through a pivot; drop. |
| fishing_rod | -1.0 | 0 / 1 / 1 | drop | configuration_matching-3 r1, configuration_matching-3 r3 | One rod-and-reel model 1.33 m long exceeding the 1.30 m limit; the rod is a faint hairline so the level line is hard to read; pivot would sit near the reel end. |
| frame | -1.0 | 0 / 2 / 2 | drop | - | Two large landscape frames (1.46 and 1.76 m wide): one empty grey hollow frame, one black frame with black panel. Both exceed 1.30 m along the level axis, so nothing hangable. |
| range_hood | -1.0 | 0 / 6 / 7 | drop | - | Six models: two are flat 1-2 cm underside plates (broken), one a curved canopy shell, three are deep chimney or canopy boxes 0.26-0.48 m thick. Nothing can hang flat with a readable level edge; drop. |
| wall_mounted_loudspeaker | -1.0 | 0 / 1 / 1 | drop | - | Single deep black speaker box 0.31 m tall on a bracket; boxy, narrow horizontal edge, no readable level axis; does not fit a flat wall-pivot task. |

