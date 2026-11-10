**7 scenes**

- house single floor, house double floor upper, house double floor lower, Rs_int, hotel_suite_large, restaurant_diner, office_cubicles_right 

**7 task families** 

task family - defined by spatial concept 

- camera relocalization, object scaling, mental rotation, disembedding, alignment, configuration matching, inter object relation satisfaction

**23 tasks** 

task - defined by iddl goal. Ideally 1 bddl per task (probably then we should have  “placed” predicate that can mean either inside / ontop (only used for init). 

Also add to camera relocalization - obstructed at bddl goal that obstalce pose reached and its pose should be the same as init pose. (so robot should not move the obstacle). But do not put it in measure block. 

- camera relocalization - standard, camera relocalization - obstructed, object scaling, mental rotation, disembedding, alignment - world vertical, alignment - axial, configuration matching - block design, configuration matching - attachment, configuration matching - articulation close small scale, configuration matching - articulation open small scale, configuration matching - articulation close large scale, configuration matching - articulation open large scale, inter object relation satisfaction - ontop small scale, inter object relation satisfaction - under small scale, inter object relation satisfaction - inside small scale, inter object relation satisfaction - nextto small scale, inter object relation satisfaction - touching small scale, inter object relation satisfaction - ontop large scale, inter object relation satisfaction - under large scale, inter object relation satisfaction - inside large scale, inter object relation satisfaction - nextto large scale, inter object relation satisfaction - touching large scale

**Eval (standard): 100 task class, each has 1 template, each template has 10 instances** 

task class(category?) - Defined by target object & partner object category & distractor object category ( 1 task class can actually have multiple support categories ! we treat desk, countertop, bed, sofa, etc as 1 category “support”) 

template - Defined by scene (task room), target object model & partner object model & (max) distractor object model(s) & support object model 

instance - Exact init, goal poses of target object model & partner object model & distractor object models

(# task class (N_TC) per task in eval suite)

- Camera relocalization - standard (1)
- Camera relocalization - obstructed (9) e.g. camera relocalization - obstructed - hamper
- Object scaling (10)
- Mental rotation  (10)
- Disembedding (10) e.g. disembedding - bottle_of_milk - bottle_of_juice,carton_of_milk,carton_of_juice
- Alignment - world vertical (5)
- Alignment - axial
- configuration matching - block design (1)
- configuration matching - attachment (4)
- configuration matching - articulation close small scale (2)
- configuration matching - articulation open small scale (2)
- configuration matching - articulation close large scale (3)
- configuration matching - articulation open large scale (3)
- inter object relation satisfaction - ontop small scale (4) e.g. inter object relation satisfaction - ontop small scale - pack of kielbasa - platter
- inter object relation satisfaction - under small scale (4) e.g. inter object relation satisfaction - ontop small scale - saucer - teacup (in bddl order (saucer under teacup)
- inter object relation satisfaction - inside small scale (4)
- inter object relation satisfaction - nextto small scale (4)
- inter object relation satisfaction - touching small scale (4)
- inter object relation satisfaction - ontop large scale (3)
- nter object relation satisfaction - under large scale (3)
- inter object relation satisfaction - inside large scale (3)
- inter object relation satisfaction - nextto large scale (3)
- inter object relation satisfaction - touching large scale (3)

**Eval - (held out)** 

1) object model level - same 100 (actually 99, because camera relocalization standard has no object model) task class as standard, but each with 1 different template with unseen object models. Each template has N instances (see table below) 

- all object models involved (target, partner, support, distractor) should not have been used globally as any (target, partner, support, distractor) globally (across all task families)
- the held out support should be an unseen support that has similar aspect ratio. It doesn’t have to be same category as the held in eval class, but its utility should be similar (e.g. desk ↔ breakfast table are similar but bookcase desk is not similar.)

2) object category & scene level - 23 task class (1 per task), 1 template per task class with unseen object category & scene (actually one room). Each template has N instances (see table below) 

- all object categories & scene rooms involved (target, partner, support, distractor) should not have been used globally as any (target, partner, support, distractor) globally (across all task families)
- wdyt - the object category her doesn’t need to have multiple models actually it’s better if it has single ready model. Also make the task classes share held out rooms & supports (& scene object partner) so that we don’t have to hold out too many. E.g. desk used as held out support for tabletop tasks can also be used as partner for iors large scale. If it is possible to add large scale articulated object and/or a object that can be used for iors large scale inside to the held out room, do so.

**Train: 100 + task class (must include the 100 eval task classes), each has 1+ template, each template has N instances. Ideally, there should be 1000+ templates and 100k + instances.** 

- We only apply perturbations here - NEVER apply perturbation for ANY eval instances
    - scale perturbation: for each target model, select 2 random scales within range [0.8, 1.2] of the default scale (default scale might not be 1 for some object models)
    - height perturbation: for each support model, select 2 random heights (only scale height) within range [0.8, 1.2]
    - reach bands & grasps should be computed per scale & keyed by scale. scale (and other perturbations) should also be reloadable later when loading instances (E.g. open loop replay)
        - wdyt: for grasps, test if it saves time if we start with the grasps of the default scale - and adjust so that the gripper end tip touch approximately the same region.

|  | Room anchoring  | Selecting object model for templates |
| --- | --- | --- |
| camera relocalization - standard | pick 10 rooms |  |
| camera relocalization - obstructed | pick 10 rooms. Shared by all task classes. 

At sampling time:
For each room loaded, sample N instances of M templates  | Try other object models of the task’s eval class first. (10 max per obstacle category - 11 if including held out). Then fill in 11*N_TC object models. These can be object models used in other held in tasks, or categories with only 1 object model. 

Object models must be commonsense “obstacles” in a house.  |
| Object scaling  | pick 10 rooms. Shared by all task classes. 

At sampling time:
For each room loaded, sample N instances of M templates  | Try other object models of the task’s eval class first. (10 max per obstacle category - 11 if including held out). Then fill in 11*N_TC object models. These can be object models used in other held in tasks, or categories with only 1 object model. 

Object models must be graspable. It’s also a plus if they are symmetric.  |
| Mental rotation | pick 10 rooms. Shared by all task classes. 

At sampling time:
For each room loaded, sample N instances of M templates  | TODO: make 10 more mental rotation blocks (the eval class “classroom mock up”) 

Try other object models of the task’s eval class first. (10 max per obstacle category - 11 if including held out). Then fill in 11*N_TC object models. These can be object models used in other held in tasks, or categories with only 1 object model. 

Object models must be graspable. They must NOT be symmetric - so that there are no ambiguities. Use agent to review.  |
| Disembedding  | pick 10 supports. supports anchor the task rooms. Shared by all task classes. 

supports must fit the two trays + distance between them. Filter first by surface area. 

At sampling time:
For each support loaded, sample N instances of M templates  | TODO: make 10 more puzzles (the eval class “puzzle?) 

tray models:
choose N candidates. N depends on # ready tray models with flat surface. Scale them to the tray sizes used for the eval demos. Hold 2 tray models out (one for source one for dst) as held out model. 
For held out category select from platter category (unless platter is used right now in eval classes), scale them to about same size as the tray models.  

target object: 
Try other object models of the task’s eval class first. (10 max per object category - 11 if including held out). Then fill in 11*N_TC object models. These can be object models used in other held in tasks, or categories with only 1 object model. 

distractors: 
each task class should have a list of distractor category (and thus list of distractor models). Hold out models used for eval-held out models). Others can be randomly chosen from the list of distractor models  |
| Configuration matching -block design

 | pick 10 supports. supports anchor the task rooms. 

supports must fit the board + 4 blocks (max scale) - wdyt: see the average?max? layout space needed in the sampled instances

At sampling time:
For each support loaded, sample N instances of M templates  | target object: 
TODO: make 10 more blocks with
same set of patterns as the existing kohs block, but different permutation per side. 
TODO: predefine N top-side patterns. Each instance can choose different pattern (so the 10 eval instances can have different goal patterns) 

the four blocks in one instance should be identical models.

held out object model: 
use Wechsler Block Design block, also 2x2  |
| Configuration matching -articulation (small scale) | pick 10 supports per task class (but there can be large overlaps between them). supports anchor the task rooms. 

supports must fit max scaled small scale articulation object model of that task class

At sampling time:
For each support loaded, sample N instances of M templates  | target object:
Try other object models of the task’s eval class first. (10 max per object category - 11 if including held out). Then fill in 11*N_TC object models. These can be object models used in other held in tasks, or categories with only 1 object model. The object model must have a articulated joint. 
If an object model has >1 articulated joints that can be moved, treat(and count) each joint as 1 model  |
| Configuration matching -articulation (large scale) | target object(origin scene object) anchors the task rooms. 

At sampling time:
For each target object(origin scene object)/task room loaded, sample N instances of M templates  | target object:
  • 10 max per object category - 11 if including held out. 
  • Max 11*N_TC object models
1) If, out of the 7 scenes there is another object model of the same category in eval class use that (priority: held-out > held-in)
2)  Add all available articulated scene objects as task classes (reserve 1 for held out category)
3) If there isn’t enough, replace the target object model in eval template with another object model of the same category, scaled to be no larger than the original object model, put at the same place as the original one.  |
| Configuration matching -attachment | pick 10 supports per task class (but there can be large overlaps between them). supports anchor the task rooms. 

supports must fit max scaled biggest attachment layout of that task class -wdyt: see the average?max? layout space needed in the sampled instances

At sampling time:
For each support loaded, sample N instances of M templates  | target object & partner object must not be scene objects. They should be able to be placed on a support. 

target object & partner object pair
  • 10 max per object category pair - 11 if including held out. 
  • Max 11*N_TC pairs

For attachment - the held out “object models” may not be exactly the same category for both target & partner. Often it will be the case that the target is same category but partner is not (See representatives.csv) 

1) Collect all the eval class’s target object category variants & their attachment partners 
2) Collect all available attachment pairs  |
| Alignment - world vertical | pick 10 supports. supports anchor the task rooms. 

The support (wall) should not be transparent glass wall. 

At sampling time:
For each support loaded, sample N instances of M templates  | Try other object models of the task’s eval class first. (10 max per obstacle category - 11 if including held out). Then fill in 11*N_TC object models. These can be object models used in other held in tasks, or categories with only 1 object model. 

Object models must have commonsense & distinguishable “world vertical” aligned direction. For example, some wall clocks were blacklisted because they had ambiguous aligned direction (e.g. no numbers so we can’t tell where should be the top)  |
| Alignment - axial | pick 10 supports. supports anchor the task rooms. 

The support should be rectangular. 

At sampling time:
For each support loaded, sample N instances of M templates  | Try other object models of the task’s eval class first. (10 max per obstacle category - 11 if including held out). Then fill in 11*N_TC object models. These can be object models used in other held in tasks, or categories with only 1 object model. 

Object models must have commonsense & distinguishable “axial” aligned direction. Avoid using objects that have ambiguity (e.g. a textureless rectangular cube would be two-way ambiguous. But with texture, like board game it’s not ambiguous  |
| Inter-object spatial relationship
(small scale) | pick 10 supports per task class (but there can be large overlaps between them). supports anchor the task rooms. 

supports must fit max scaled biggest layout of that task class -wdyt: see the average?max? layout space needed in the sampled instances

At sampling time:
For each support loaded, sample N instances of M templates  | target object & partner object must not be scene objects. They should be able to be placed on a support. 

target object & partner object pair
  • 10 max per object category pair - 11 if including held out. 
  • Max 11*N_TC pairs

1) Collect all the eval class’s target object category variants & partner object category variants. Try to make each combination unique [(A,B) (A,C) → not unique, (A, B) (C,D) → unique]
2) Fill in 11*N_TC pairs with new classes. These can be object models used in other held in tasks, or categories with only 1 object model. |
| Inter-object spatial relationship
(large scale) | partner object(origin scene object) anchors the task rooms. 

At sampling time:
For each partner object(origin scene object)/task room loaded, sample N instances of M templates  | target object & partner object pair
  • 10 max per object category pair - 11 if including held out. 
  • Max 11*N_TC pairs

Collect all the eval class’s target object category variants
Try to make each combination unique [(A,B) (A,C) → not unique, (A, B) (C,D) → unique]
1) If, out of the 7 scenes there is another object model of the same category partner in eval class use that (priority: held-out > held-in)
2) If there isn’t enough, replace the partner object model in eval template with another object model of the same category, scaled to be no larger than the original object model, put at the same place as the original one. 
3) if still isn’t enough, allow non-unique combinations.  |

|  | train 

room pick: 
  • for camera relocalization, consider the path buckets - we need some rooms to allow 5-10m paths
  • want some amount of diversity in total. Aliases with scene objects have rooms fixed so diversity comes mostly from camera relocalization, mental rotation and object scaling 

perturbations: 
perturbations should be saved in tro and be able to be loaded later e.g. for replay or eval 
for same goal group instances, the same sampling time perturbation should apply. (because one instance’s goal image should be able to guide another same group’s instance) 
  1. scene perturbation
for each instance, with probability 0.5 remove(wdyt: actually, park to non visible area because it needs to be restored later for next instance) non-structural objects in the scene (wdyt: record the removed object in the tro file, and when loading instance later, we need to make sure object is removed). The removed objects should never be task relevant objects. 
2. lighting perturbation 
for each instance, with probability 0.5 alter the lighting fixtures’ intensity & skybox color. the lighting fixture intensity should be sampled from normal distribution with default lighting at middle. Find min and max lighting intensity using agents so scene isn’t too dark and bright such that it is unnatural.  
3. scale perturbation 
we only apply scaling perturbation to non scene objects / and scene objects that rest on the floor and which scaling and settling them at same root pose (with the bottom touching floor, not penetrating it) does not make it collide with other scene objects. often this will only allow smaller scaling and not bigger scaling. 

Q. can lighting perturbation be done at replay time? or should demo collection time & replay time have same lighting settings (so perturbation saved at collection time → reloaded at replay time)? 
 

etc: 
  • save start & goal poses that failed to find a oracle solution for later use - split b/w sampling failure and demo collection failure & note the reasons (e.g. whether it’s pose settled failure, mid collision failure, etc) 
  • If tabletop task & support is a table-like object, remove all chairs near them. 
  • sampling time perturbations should be the same for the goal pairs  | eval - held in | eval - held out  |
| --- | --- | --- | --- |
| Camera relocalization - standard

path bucket: 
  • (0, 3] m
  • (3, 5] m
  • (5, 10] m | tot 1000

pick 10 rooms 
  • assign # instances per room by traversable area in room
  • want to balance # instances per object model and # instances per path bucket 

Sampling time perturbations 
  •  scene perturbations (wdyt: bc this needs to happen before visibility checks) 

Demo collection time perturbation: 
  • lighting perturbation  | 10 instances
  •  3, 4, 3 per path bucket  | 10 instances in held out room 
  • 3, 4, 3 per path bucket  |
| Camera relocalization - obstructed | tot 9000

pick 10 rooms 
  • assign # instances per room by traversable area in room
  • want to balance # instances per object model and # instances per path bucket 

Sampling: 
for each room, load all obstacle model candidate (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the obstacle model differs) but template & tro should only carry the object model used for that instance. 

Sampling time perturbations
  •  scene perturbations (wdyt: bc this needs to happen before visibility checks) 
  • scale perturbation. for each instance, with probability 0.5, scale the obstacle model randomly within min-max range (choose an absolute min max size per obstacle category using commonsense)

Demo collection time perturbation: 
  • lighting perturbation  | 10 instances per 9 category (1 model per category)  
  • 3, 4, 3 per path bucket  | obstacle (model, category)

9  instances per 9 held out model (1 per held-in category)
9 instance per 1 held out category (1 model) 

  • 3, 3, 3 per path bucket  |
| Object scaling

path bucket: 
  • 0.05–0.10 m
  • 0.10–0.15 m
  • 0.15–0.30 m | tot 10k

pick 10 rooms 
  • assign # instances per room by traversable area in room
  • want to balance # instances per object model and # instances per path bucket 

Sampling:
  •  Important: the robot base pose should not be fixed per room. (e.g. fixed to point or fixed to region)
  • For each room, load all target model candidates (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  •  Choose target object’s goal pose first. Then, choose robot base pose. then choose grasp & init pose. so the same target object’s goal pose(&robot base pose) gets max 3 instances (with different grasp&init pose pair). (wdyt: this is so that later, we can test whether giving goal image from another rollout can also work if main goal is the same. we also need to record these groups for this later use)

Sampling time perturbations
  • scale perturbation. random select scale of target object from the scale candidates

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 10 category (1 model per category)  
  • 3, 4, 3 per path bucket 

 | target (model, category)

9  instances per 10 held out model (1 per held-in category) 
  • 3, 3, 3, per path bucket 
10 instance per 1 held out category (1 model)
  • 3, 4, 3 per path bucket  |
| Mental rotation

path bucket: 
  • 30–45°
  • 45–75°
  • 75–120° | tot 10k

pick 10 rooms 
  • assign # instances per room by traversable area in room
  • want to balance # instances per object model and # instances per path bucket 

Sampling:
  • Important: the robot base pose should not be fixed per room. (e.g. fixed to point or fixed to region)
  • for each room, load all target model candidates (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  • Choose target object’s goal pose first. Then, choose robot base pose. then choose grasp & init pose. so the same target object’s goal pose(&robot base pose) gets max 5 instances (with different grasp&init pose pair). (wdyt: this is so that later, we can test whether giving goal image for another rollout can also work. we also need to record these groups for this later use)

Sampling time perturbations
  • scale perturbation. random select scale of target object from the scale candidates

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 10 category (1 model per category)  
  • 3, 4, 3 per path bucket  | target (model, category)

9  instances per 10 held out model (1 per held-in category) 
  • 3, 3, 3, per path bucket 
10 instance per 1 held out category (1 model)
  • 3, 4, 3 per path bucket  |
| Disembedding

crowdedness(coverage of the source tray) 
  • <0.3 
  • 0.30–0.45
  • ≥ 0.45

 | tot 10k

N rooms (comes naturally from support candidates) 
  • assign # instances per support by # of usable edges (wdyt: usable edges means that there’s a free area for robot to be at standoff after the chairs near the support are removed. arms can be at any position when judging this. it doesn’t need to be at untucked pose.) 
  • want to balance # instances per object model and # instances per crowdedness

Sampling:
  • STRICT: the start & goal orientation of target object should be the same. trays xy should align with table xy. trays should be at same height from table edge. 
  • for each support, load all target model candidates and distractor model candidates(wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. Each instance can get different number of distractors even if they are same template)
  • Choose target object’s goal pose & goal tray position first. Unlike before the target object’s goal position should not be fixed to center of the tray. Then the start tray position naturally follows(only need to choose left/right). Then choose the target object’s start pose & grasp. its orientation should be the same as the goal pose, so you only need to choose a position on the start tray. Then choose distractors & their poses (randomly from the distractor pool of that category/alias & place them. exception: the puzzle. it has fixed init layout & distractor set.). So same [goal tray pose & target object goal pose] pair should have  max 5 instances with different [target object start pose, target object grasp, distractor models& poses (only one of these needs to be different to count as different pair)] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object from the scale candidates.(for the puzzle the target object & distractor should be scaled equally so they have same layout when scaled) random select height of support object from the height candidates. 

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 10 category (1 model per category)  
  • 3, 4, 3 per crowdedness | target (model, category) 
distractors (model, category) 
support (model)

9  instances per 10 held out model (1 per held-in category) 
  • 3, 3, 3 per crowdedness
10 instance per 1 held out category (1 model) 
  • 3, 4, 3 per crowdedness |
| Configuration matching -block design

object model variation: 

same set of patterns as the existing kohs block, but different permutation per side. different colors. 

the four blocks in one “model” should be identical. 

held out object model: 
use Wechsler Block Design block, also 2x2  | tot 1k 

N rooms (comes naturally from support candidates) 
  • assign # instances per support by # of usable edges (wdyt: usable edges means that there’s a free area for robot to be at standoff after the chairs near the support are removed. arms can be at any position when judging this. it doesn’t need to be at untucked pose.) 
  • want to balance # instances per object model and # instances per # flips necessary

Sampling: 
  • STRICT: the four blocks in one “model” should be identical. the board xy should align with table xy.
  • for each support, load all target model candidates(4 per model) (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  • Choose target object group’s goal pose from the list of goal patterns first. Then choose the blocks’ start poses and grasps (the board pose should be same for start and goal). So same object group’s goal pose pair (goal pattern) should have  max 5 instances with different [blocks start poses, block grasps] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object group from the scale candidates. (the blocks and boards should be scaled equally so that they fit) random select height of support object from the height candidates. 

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 1 category (1 model per category)  
  • 2, 3, 3, 2 per # flips necessary | target (model, category) 
support (model) 

10  instances per 1 held out model (1 per held-in category) 
  • 2, 3, 3, 2 per # flips necessary |
| Configuration matching -articulation (small scale) | tot 4k (2k per open, close)

N rooms (comes naturally from support candidates) 
  • assign # instances per support by # of usable edges (wdyt: usable edges means that there’s a free area for robot to be at standoff after the chairs near the support are removed. arms can be at any position when judging this. it doesn’t need to be at untucked pose.) 
  • want to balance # instances per object model and # instances per % joint limit open (>50%, ≤ 50%)

Sampling: 
  • STRICT: If there are multiple articulated joints per Only one articulated joint should change 
  • for each support, load all target model candidates (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  • Choose target object’s goal pose & goal angle first(in close, the target angle is fixed to 0). Then choose the start angle(in open, the start is fixed to 0) & grasp. the object start pose will be same as its position should be the same as the goal pose, so you only need to choose a orientation(joint angle) . So same goal pose, goal angle pair should have max 5 instances with different [start angle, grasp] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object from the scale candidates. random select height of support object from the height candidates. 

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 20 (10 open, 10 close) instances per 2 category (1 model per category)  
  • 5, 5 per % joint limit | target (model, category) 
support (model)

18(9 open, 9 close) instances per 2 held out model (1 per held-in category) 
  • 5, 4 per % joint limit open(>50% or not, at start or end)
4(2 open, 2 close) instance per 1 held out category (1 model) 
  • 2, 2 per % joint limit  |
| Configuration matching -articulation (large scale)

object models: 
scene object + replacements  | tot 6k (3k per open, close)

N rooms  (comes naturally from target object models that are scene objects )
  •  assign # instances per target object model by # of movable articulated joints (handle needs to be reachable for an articulated joint to be movable
  • want to balance # instances per % joint limit open (>50%, ≤ 50%)

Sampling: 
  • STRICT: If there are multiple articulated joints per Only one articulated joint should change 
  •  for each target object model that is a scene object in the room, load all target model candidates -which are scene object + replacements for this task (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. If a replacement is chosen, the scene object should be removed & replaced with the replacement (with pre-configured scale)- and this be recorded in template/tro so it can be restored at replay/eval. 
  • Choose target object’s goal pose & goal angle first(in close, the target angle is fixed to 0). Then choose the start angle(in open, the start is fixed to 0) & grasp. the object start pose will be same as its position should be the same as the goal pose, so you only need to choose a orientation(joint angle) . So same goal pose, goal angle pair should have max 5 instances with different [start angle, grasp] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object from the scale candidates. (only when applicable. for example, door shouldn’t be scaled - it would require the whole wall to be scaled with it). Ask agent whether scaling them sounds okay. 

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 20 (10 open, 10 close) instances per 3 category (1 model per category)  
  • 5, 5 per % joint limit | target (model, category)

18(9 open, 9 close) instances per 3 held out model (1 per held-in category) 
  • 5, 4 per % joint limit open(>50% or not, at start or end)
6(3 open, 3 close) instance per 1 held out category (1 model) 
  • 2, 2 per % joint limit |
| Configuration matching -attachment | tot 4k 

N rooms (comes naturally from support candidates) 
  • assign # instances per support by # of usable edges (wdyt: usable edges means that there’s a free area for robot to be at standoff after the chairs near the support are removed. arms can be at any position when judging this. it doesn’t need to be at untucked pose.) 
  • want to balance # instances per object model pair

Sampling: 
  • for each support, load all target and partner model candidates (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  • Choose target object and partner object’s goal pose first. the partner’s start pose naturally follows. Then choose the target object’s start pose & grasp. So same [target object, partner object] goal pose pair should have max 5 instances with different [target object start pose, grasp] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object & partner object from the scale candidates. (the target & partner has to be scaled together such that they fit/can attach each other after scaling). random select height of support object from the height candidates. 

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 4 category (1 model per category)  
 | target (model, category) 
support (model)

9  instances per 4 held out model (1 per held-in category) 
4 instance per 1 held out category (1 model)  |
| Alignment - world vertical | tot 5k 

N rooms (comes naturally from support candidates) 
  • assign # instances per support by # of usable space on the wall
  • want to balance # instances per object model and per range bucket

Sampling: 
  • STRICT: There must be one goal orientation per object model & support across instances. 
for each support, load all target and partner model candidates (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  • Choose target object’s goal position first. goal orientation is fixed for alignment family. Then choose the target object’s start orientation & push(optional brace). So same target object goal pose pair should have max 5 instances with different [target object start pose (orientation), push(optional brace)] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object from the scale candidates. 

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 5 category (1 model per category)  
• 3, 4, 3, per range bucket

 | target (model, category) 

9  instances per 5 held out model (1 per held-in category) 
• 3, 3, 3, per range bucket
5 instance per 1 held out category (1 model) 
• 2, 2, 1, per range bucket |
| Alignment - axial | tot 5k 

N rooms (comes naturally from support candidates) 
  • assign # instances per support by # of usable space on the wall
  • want to balance # instances per object model and per range bucket 

Sampling: 
  • STRICT: There must be one goal orientation per object model & support across instances. 
for each support, load all target and partner model candidates (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  • Choose target object’s goal position first. goal orientation is fixed for alignment family. Then choose the target object’s start orientation & push(optional brace). So same target object goal pose pair should have max 5 instances with different [target object start pose (orientation), push(optional brace)] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object from the scale candidates. 

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 5 category (1 model per category)  
• 3, 4, 3, per range bucket

 | target (model, category) 

9  instances per 5 held out model (1 per held-in category) 
• 3, 3, 3, per % range bucket
5 instance per 1 held out category (1 model) 
• 2, 2, 1, per % range bucket |
| Inter-object spatial relationship
(small scale)

ontop 
  • pack of kielbasa, platter
  • steak, frying pan
  • bar soap, soap dish
  • carving knife, cutting board 
under
  • teacup, saucer 
  • beer glass, coaster
  • plate, place mat 
  • spatula, paper sheet 
inside
  • brussel sprouts, bowl
  • hard candy, hingeless jar
  • sunflower, vase 
  • garlic clove, saucepot 
nextto
  • marker, colored pencil 
  • salt shaker, pepper shaker 
  • tablespoon, chopstick 
  • gym shoe, gym shoe 
touching
  • box of sanitary napkins, box of tissues 
  • box of cereal, box of cereal
  • hardback, hardback
  • spray bottle, bottle of detergent | tot 20k

N rooms (comes naturally from support candidates) 
  • assign # instances per support by # of usable edges (wdyt: usable edges means that there’s a free area for robot to be at standoff after the chairs near the support are removed. arms can be at any position when judging this. it doesn’t need to be at untucked pose.) 
  • want to balance # instances per object model pair (target, partner)

Pre-Sampling / object model selection
  • STRICT: for object models & scales selected for target and partner, we need to check if it can satisfy the symbolic goal. Also, for nextto/touching we should have constraints / desired relative poses for the goal beforehand. 
For ontop/under/inside if the partner object is placed on a rectangular support and there is a clear aligned direction (e.g. platter on a rectangular table) it should be placed that way.  
      ◦ ontop/under: target should be able to placed at resting orientation fully on top of the partner’s flat surface (e.g. think of steak and frying pan) 
      ◦ inside: target should be able to fit inside the partner’s convex hall 
      ◦ nextto/touching: use agent to review the natural/desired relative goal orientation/position between the objects. (e.g. marker and colored pencil would ideally be with the tips facing same side, and their vertical positions would be within the longer one’s range. for salt shaker and pepper shaker it wouldn’t matter. for gym shoe gym shoe, the left shoe and right shoe should be at left and right, and the tips would be facing same side, and its verticals would be aligned. table spoon and chopstick would have tips facing same side, and probably its bottom aligned. box of sanitary napkins, box of tissues or box of cereal, box of cereal should have a large facing touching each other, hardback should be upright & spine facing robot, spray bottle, bottle of detergent - bottle of detergent should have wider face as the touching side, spray bottle should not have the sprayer side as the touching side) 

Sampling 
  • STRICT: the target model’s start pose should not be restricted - for example, even if goal orientation of tablespoon and chopstick should be tips facing same direction, at init the chopstick can be in different orientation. This also means that unlike now, the start and goal orientation of the target model can change. 
  • STRICT: outside the constraints above, there should be some amount of variance. For example, if there is a range of distances that satisfy nextto, select distance randomly from mid 80% of that range per instance. 
  • For inside, target object’s orientation doesn’t matter at eval. (it can roll inside) 
  • for each support, load all target and partner model candidates (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  • Choose target object and partner object’s goal pose first. Then choose the target object’s start pose & grasp. So same [target object, partner object] goal pose pair should have max 5 instances with different [target object start pose, grasp] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object & partner object from the scale candidates. (the target & partner has to be scaled so that they meet the STRICT requirements). random select height of support object from the height candidates. 

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 20 (4 per ontop, under, nextto, touching, inside) category (1 model per category)   | target (model, category) 
partner (model, category)
support (model)

9 instances per 20 (4 per ontop, under, nextto, touching, inside) held out model (1 per held-in category) 
4 instances per 5(1 per ontop, under, nextto, touching, inside)  held out category (1 model per category)  |
| Inter-object spatial relationship
(large scale)

ontop 
  • soap dispenser, countertop
  • mouse, desk
  • pillow, bed 
under
  • sandal, bed
  • briefcase, desk
  • trash can, desk
inside
  • hardback, bookcase
  • jar of spaghetti sauce, fridge
  • plate, sink
nextto
  • swivel chair, desk 
  • serving cart, coffee table
  • coffee table, sofa 
touching 
  • ottoman, sofa
  • umbrella, hall tree
  • storage box, bookcase

partner model candidates are replacements  | tot 15k

N rooms (comes naturally from support candidates) 
  • assign # instances per support by# of usable edges or usable compartments (e.g. inside hardback bookcase, inside jar of spaghetti sauce fridge, touching umbrella hall tree - because it is actually kind of “inside” a compartment of a hall tree) (wdyt: usable edges means that there’s a free area for robot to be at standoff after the chairs near the support are removed. arms can be at any position when judging this. it doesn’t need to be at untucked pose.) 
  • want to balance # instances per object model pair (target, partner)

Pre-Sampling / object model selection
  • STRICT: for object models & scales selected for target and partner, we need to check if it can satisfy the symbolic goal. 
      ◦ ontop/under/inside: does the target model fit ontop/under partner
      ◦ nextto/touching: does target model fit nextto/touching free area beside partner. 
  • STRICT: Also, objects that will be solved by pick & place then push - agent needs to review and pre-pick what is best push orientation (and thus the goal orientation because orientations ideally do not change during push) - this applies to the three under aliases and storage box, bookcase. 

Sampling 
  • STRICT: special cases - hardback bookcase hardback should be straight up & leaned against bookcase wall. umbrella, hard tree - umbrella should be lying on top of one of the compartments of halltree (umbrella goal pose shouldn’t be on the floor) 
  • STRICT: aliases requiring pushes of large objects should only push in a straight line 
  • STRICT: the target model’s start pose should not be restricted to be same with goal orientation This means that unlike now, the start and goal orientation of the target model can change. Exception: big objects only solved through pushing (e.g. swivel chair, serving cart, coffee table, ottoman) will have same start and goal orientation. 
  • STRICT: outside the constraints above, there should be some amount of variance. For example, if there is a range of distances that satisfy nextto, select distance randomly from mid 80% of that range per instance 
  • for each support, load all target and partner model candidates (wdyt: and park them in non visible area) at world build (wdyt: this way we don’t need to re-load or load in between sampled instances even if the object model differs) but template & tro should only carry the object model used for that instance. 
  • Choose target object and partner object’s goal pose first. Then choose the target object’s start pose & grasp. So same [target object, partner object] goal pose pair should have max 5 instances with different [target object start pose, grasp] pairs. 

Sampling time perturbations
  • scale perturbation. random select scale of target object & partner object from the scale candidates. (the target & partner has to be scaled so that they meet the STRICT requirements).

Demo collection time perturbation: 
  • scene perturbation 
  • lighting perturbation | 10 instances per 15 (3 per ontop, under, nextto, touching, inside) category (1 model per category)   | target (model, category) 
partner (model, category)

9 instances per 15 (3 per ontop, under, nextto, touching, inside) held out model (1 per held-in category) 
3 instances per 5(1 per ontop, under, nextto, touching, inside)  held out category (1 model per category)  |

Object models to generate: 

- texture/color/shape should be same as the models used in eval template.
- See /home/minyeongk/Desktop/projects/ispatialgym/object_models_to_generate.zip (it also contains goal patterns for block design)
- See /home/minyeongk/Desktop/projects/ispatialgym/cursor-session-dc7b6a94/chat.md to see how the existing models were created.
1. Disembedding puzzle models 
    
    The context shapes of the Leuven Embedded Figures Test.
    
    The context shapes are coded with a unique ID that indicates with which target shape they should be combined.
    
    The target shapes are indicated by an ID ranging from: 000 – 063. The context shapes are indicated by an ID. The first three digits of this ID will correspond to a unique target shape. For instance, target shape “000” is made to be combined with the context shapes with ID code “000a”, “000b” and “000c”. The letter (ranging from a to c) indicates whether the shape contains the target or is a distractor: “a” always contains the target and “b” and “c” are always contexts that do not contain the target shape (wrong alternatives).
    
    So use xxx (Target) + xxxa (Distractor) 
    
2. Mental rotation blocks
    
    This project contains data and a set of 384 three-dimensional stimuli to be used to study mental rotation,
    
3. Block design 
    
    (held in object models) - use different permutation of sides. 
    
    (held out category) use Wechsler Block Design block, also 2x2 
    

Overall: 

- Alway try to use the cluster gpus (reachable via minyeong@svl11 or minyeong@sc and shiyuc@svl11 or shiyuc@sc, each has partition svl account vision and partition viscam account viscam, and has a cap) to the max cap (probably around 96 gpus total). Also rent vast gpus for speedup & massive parallelization. SPEED is most important here. You don’t need to limit the type of vast gpu to 4090 - as long as it speeds up it can be any rtx gpu that satisfies driver requirements. Use many subagents to parallelize jobs (e.g. writing oracle programs for different templates)
- parallelize jobs that can be done in parallel (e.g. while we choose the candidates, we can implement & test sanity & speed of changes made to the standard eval template production codes). And don’t wait in groups (e.g. if grasps & reach bands & exist for a template move directly to sampling instead of waiting for entire grasp & reach bands computation to be done). So it may be beneficial to first create a document about the specs, and plan & design the entire timeline first.
- Priority: Eval templates should work (& eval instances produced) → then train-only templates (& train instances produced). validate each template with 1 successful demo + 1 successful open-loop replay before producing instances. But they can work in parallel for speed.
    - If there are other templates for a standard eval class, the held out object model template for that eval class doesn’t necessarily have to be the pinned one. It can be any one of other templates that succeed quickly.
    - DO NOT make changes to oracle code that would affect the working oracle programs for the 100 standard eval templates
    - When validating a oracle program for a template works we only need 1 successful demo. don’t wait for multiple ones. (but maybe we need a couple to figure out good timeout for the goal pair group).
    - Start with templates of the standard eval classes - they probably can use the same oracle programs as the standard eval templates
    - Give templates that need new oracle programs a hard time limit - if we can’t get a oracle program to work within that hard time limit throw away the template. So we might want to start with
- all the held-in eval instances should have 1 goal pair group in the training instances (so a goal pair in held in eval instance should be one with min 2 ~ pairs). (except for tasks like camera relocalization that doesn’t have goal pair group)
    - effectively, we would try to produce goal pair group - if it doesn’t produce min 2 within time limit it would become training instance, not eval instance. if it produces min 2, one of them becomes eval instance, others become training instance.
- There should be a time limit per goal pair group (like we have time limit for producing instance) so we don’t wait for 5 every time. You can reduce max 5 to e.g max 3 if you think it limits diversity too much.
- Always montior the progress - if some are stalling and there’s a clear reason fix, if some are just slow because they are low yield/slow templates - split their instance production across many gpus & stop when the goal # instances in total are reached. So plan should be changed dynamically, not executed statically.
- We are freshly making all instances (eval & train) - don’t copy ones we have already made.
- registered variants are free train templates. use them.
- Hard limit for finishing whole demo collection is Nov 12 12PM PDT (just before the batched replay. But the batched replay should be prepared) But don’t automatically shut it down at that time, I will tell you.

Read these to get the most recent version of the 100 eval templates. These should be adapted to the requirements above, but they should be the starting point. Do NOT overwrite / mess up the code & resources used in the previous runs. Work in a sandboxed worktree. But you can copy useful stuff like grasp library & reach bands from them.

The oracle/template might change for touching_03, under_large_02, under_large_01, under_large_03 and nextto_04. I’ll tell you when the changes finish. 

Files are in `tmp/outputs`:

- GUIDE_run_100_aliases.md
- GUIDE_speedup_bugfix_strategies.md

Here’s a guide on how to use vast gpus /home/minyeongk/Desktop/projects/ispatialgym/VAST_AI_GUIDE.md

I’ll be with you for about an hour, so you should ask me questions / points to review (self conflicting, ambiguous) during that hour. Afterwards, you should be able to work on your own. 
Also after planning & timeline - maintain an artifact I can access to check live progress & state. live state should include # gpus being used.