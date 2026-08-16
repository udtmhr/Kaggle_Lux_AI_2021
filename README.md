# Kaggle_Lux_AI_2021

## Survival-strategic fork

This fork adds a scratch-distilled student while preserving the original fast IMPALA environment and the complete
19/17/4 worker/cart/city action schema. The first-place checkpoint is used only to generate target logits; its weights
are never loaded into the student. New run checkpoints record both `scratch_initialization: true` and the teacher SHA.
Offline teacher targets and online IMPALA teacher KL use original-plus-Rot180 averaged logits. Distillation also applies
Rot180 training augmentation with probability 0.5, and the rollout actor samples from the same two-view ensemble.

The new student combines masked local ConvNeXt-style blocks with axial attention at 8x8 and explicit city fuel,
unit return, resource-control, and temporal observations. The original `conv_model` remains unchanged, and
`conf/conv16_matched.yaml` is the parameter-matched architecture ablation (3.76M vs 3.80M parameters).

Distillation cache schema v2 stores replay-level fp16/uint8 observations and only actionable entity logits using
ragged positions and offsets. Schema v1 `.pt` shards are intentionally rejected; generate v2 into a new directory.

Install and run a small data smoke test:

```bash
uv sync
# The bundled teacher is tracked with Git LFS, so fetch its actual bytes first.
git lfs pull
uv run luxsr-prepare-data \
  --replay-dir /path/to/replays \
  --output-dir data/distillation_v1 \
  --teacher-checkpoint internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/lux_ai/rl_agent/062179520_weights.pt \
  --teacher-config internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/lux_ai/rl_agent/config.yaml \
  --legacy-prepared-cache-dir /home/ueda/workspace/LuxPythonEnvGym/models/teachers/lux_2021_first_place/prepared \
  --max-replays 1 --max-turns 2
uv run luxsr-train-distill --dataset-dir data/distillation_v1 --output-dir outputs/distill_v1 --epochs 1
uv run luxsr-validate-run outputs/distill_v1
```

Continue with IMPALA from the evaluated distilled `best.pt` (policy weights only), first with
`survival_strategic_shaping`, then with `survival_strategic`. Enable the frozen online teacher only after filling
`teacher_load_dir` and `teacher_checkpoint_file`. Its KL coefficient decays using the `teacher_kl_*` fields.
The default strategic configs also enable an episode-level opponent league: current-policy self-play, the frozen
first-place model, two historical checkpoints, and a deterministic economy rule bot. The learner side is randomized;
external-opponent actions and values are excluded from the loss. Adjust `league_opponents[*].weight`, or set
`league_enabled=false` to recover pure self-play.
The v2 mixture is 40% self-play, 20% first-place, 20% mid-history (`28576448`), 10% late-history (`59822400`),
and 10% rule-based. `league_config_version` upgrades older saved mixtures on full-state resume while retaining the
checkpoint optimizer, scheduler, step, and game counters; explicit CLI overrides remain authoritative.
Survival shaping decays against a process-shared global completed-game counter initialized from
`total_games_played`; it therefore remains continuous across actors, vectorized environments, and full-state resumes.
The shaping config uses 14,000 global games, calibrated to reach zero around the end of the current 5M-step stage.

```bash
uv run python run_monobeast.py --config-name survival_strategic_shaping \
  load_dir=/absolute/path/to/outputs/distill_v1 checkpoint_file=best.pt weights_only=true
```

To train with temporal changes in relative city-tile and unit counts instead,
start a separate run with the dedicated reward configuration:

```bash
uv run python run_monobeast.py --config-name survival_strategic_relative_counts \
  load_dir=/absolute/path/to/outputs/distill_v1 checkpoint_file=best.pt weights_only=true
```

This uses `delta[(own - enemy) weighted counts]` for shaping, clips each step,
decays shaping over 14,000 globally counted games, and retains the terminal
win/draw/loss reward as the primary objective.

Use `latest.pt` only to resume the same distillation run; it also contains optimizer state.

Promotion evaluation must use both orientations for every seed. Store one JSON object per game with `opponent`,
`seed`, `candidate_player`, and `winner`, then aggregate it with:

```bash
uv run luxsr-evaluate-checkpoint \
  outputs/survival_strategic/2026-08-11/15-57-59/1140224_weights.pt
```

`luxsr-evaluate-checkpoint` automatically builds `/tmp/lux_candidate_<step>/main.py`, runs the first-place opponent
in both orientations on one 12x12 seed, and writes `games.jsonl` plus `report.json` under
`outputs/evaluation/step_<step>_vs_first_place`. A candidate that stops responding after turn 0 fails the evaluation.
For a full 80-game evaluation, add `--seeds 10 --map-sizes 12 16 24 32`. Use `--opponent` and `--opponent-name` to
select another opponent. `--backend auto` first compares a small official-CLI sample with the internal engine and,
when winners agree, evaluates eight games per GPU inference batch. Use `--batch-games 4` if GPU memory is constrained,
or `--backend official` to force isolated CLI matches. Official matches run with `--workers 2` by default.
`luxsr-prepare-eval-agent` remains available when only the isolated agent is needed.

Do not promote from aggregate score alone: inspect opponent-specific paired delta, bootstrap lower bound, teacher
non-regression, city survival, and stranded-fuel diagnostics.

### Gated strength-v3 training

`survival_strategic_strength_v3` keeps first-place at an exact 25% rollout probability while PFSP distributes the
remaining 75%, and floors online teacher KL at `0.005`. Use the coordinator to stop at fixed milestones, save a full
checkpoint, run matched first-place evaluation while training is not using the GPU, and resume optimizer/scheduler
state only after the quality gate passes:

```bash
uv run --locked luxsr-train-eval \
  --base-checkpoint /absolute/path/to/old-5m/5000128_weights.pt \
  --run-root outputs/strength_v3_gated_001 \
  --milestones 250000 500000 750000 1000000 \
  --seeds 5 --map-sizes 12 16 24 32
```

To start a new run without repeating an unchanged baseline evaluation, point
`--reuse-baseline-evaluation` at the previous run's `baseline_evaluation` directory. The command validates the
opponent, complete seed/map/orientation schedule, evaluation backend, and report against `games.jsonl` before it
creates the new run directory:

```bash
uv run --locked luxsr-train-eval \
  --base-checkpoint /absolute/path/to/base_weights.pt \
  --reuse-baseline-evaluation /absolute/path/to/previous_run/baseline_evaluation \
  --run-root /absolute/path/to/new_run \
  --seeds 10 --map-sizes 12 16 24 32 \
  --eval-backend internal
```

The baseline and every milestone use the same seeds, map sizes, and both player orientations. By default, training
stops when score rate is more than 5 percentage points below the old 5M baseline or city-extinction rate is more
than 5 points above it. Results and decisions are written atomically to `evaluation_progress.json`. Increase
`--seeds` for the final selection; auto evaluation uses eight-game GPU batches after its parity check. Lower
`--eval-batch-games` if GPU memory is constrained. `--continue-on-fail` is available only for deliberate ablation
runs.

### Pure Evolution Strategies fine-tuning

`luxsr-train-es` evolves only action-affecting policy parameters from an RL checkpoint. It uses layer-scaled
antithetic noise, common match schedules, centered ranks, ClipUp, and a recent-update/checkpoint-difference active
subspace. Value and intent heads remain fixed. The default configuration first runs the sigma usefulness gate, then
eight gated generations, and finally matched official-CLI promotion evaluation. Each candidate is evaluated against
only `first_place` and `initial_model`; the internal backend batches all games for one opponent together. The pilot
uses two games per opponent and the main search uses three games per opponent (four/six total respectively). On CUDA,
antithetic `+/-` candidates and gate `old/new` candidates are evaluated two at a time by default; use
`--candidate-workers 1` to disable this. Official-CLI fallback remains single-candidate because it rewrites one
candidate bundle checkpoint between evaluations.

Inspect all paths and the evolved parameter set without creating a run:

```bash
uv run --locked luxsr-train-es \
  --init-checkpoint outputs/strength_winloss_001/step_1000000/0657088_weights.pt \
  --config outputs/strength_winloss_001/step_1000000/config.yaml \
  --run-dir outputs/es_strength_001 --dry-run
```

Run the requested CPU resume/checkpoint smoke with a deliberately forced sigma (this skips only the pilot):

```bash
uv run --locked luxsr-train-es \
  --init-checkpoint outputs/strength_winloss_001/step_1000000/0657088_weights.pt \
  --config outputs/strength_winloss_001/step_1000000/config.yaml \
  --run-dir outputs/es_smoke_001 --backend internal --skip-parity \
  --force-sigma 0.005 --generations 1 --directions 2 \
  --games-per-candidate 1 --gate-pairs 1 --action-probe-states 0 \
  --skip-formal-eval --device cpu --cpu-threads 4
```

For the overnight run, omit smoke overrides:

```bash
uv run --locked luxsr-train-es \
  --init-checkpoint outputs/strength_winloss_001/step_1000000/0657088_weights.pt \
  --config outputs/strength_winloss_001/step_1000000/config.yaml \
  --run-dir outputs/es_strength_001 --workers 2
```

Resume the same directory with the same checkpoint and model config by appending `--resume`. The run stores
`manifest.json`, `fitness.jsonl`, `generations.jsonl`, `latest_es.pt`, and `best_weights.pt`; it reuses completed
candidate IDs and the backend selected by the initial parity check. A rejected proposal restores the center and halves
both sigma and ClipUp speed. Internal-backend fitness rows also record forward/environment timing and CUDA peak memory;
parallel rows additionally record `candidate_group_size`, aggregate group throughput, and whether candidate concurrency
was active. Only revisit EGGROLL if `backend_profile.forward_fraction` shows neural forward as the bottleneck. Keep ES
artifacts even if the final promotion gate reports `not_promoted`.

For the actor-head-only search with an independent 4-pair screen and 20-pair confirmation gate, use
`conf/survival_strategic_es_actor_head_twostage.yaml`. The sigma pilot is mandatory for this configuration;
do not pass `--force-sigma`. A tied screen may advance to confirmation, but confirmation accepts only a strictly
positive score delta with no excessive city-extinction regression. The screen and confirmation metrics are both
stored in `generations.jsonl`.

```bash
UV_CACHE_DIR=/tmp/lux-fork-uv-cache uv run --locked luxsr-train-es \
  --init-checkpoint outputs/strength_winloss_001/step_1000000/0657088_weights.pt \
  --config outputs/strength_winloss_001/step_1000000/config.yaml \
  --es-config conf/survival_strategic_es_actor_head_twostage.yaml \
  --run-dir outputs/es_actor_head_twostage_001 \
  --device cuda --candidate-workers 2 --workers 2
```

### Reusable official-CLI agent bundle

Export the inference runtime once with an initial evaluated checkpoint. The archive contains observation construction,
action masks, collision handling, model code, and the checkpoint/config pair:

```bash
uv run python tools/export_agent_bundle.py \
  --checkpoint outputs/survival_strategic/2026-08-11/15-57-59/0659264_weights.pt \
  --config outputs/survival_strategic/2026-08-11/15-57-59/config.yaml \
  --model-name step-0659264 \
  --output survival_strategic_agent.tar.gz
```

On the machine containing `Lux-Design-S1`, extract the archive and create the isolated agent environment once:

```bash
tar -xzf survival_strategic_agent.tar.gz -C /path/to/Lux-Design-S1/agents
cd /path/to/Lux-Design-S1
sh agents/survival_strategic_agent/setup_venv.sh
```

Later checkpoints do not require another export or environment setup. Register each checkpoint together with the
config from the same run, then select it by name at match time:

```bash
python agents/survival_strategic_agent/install_model.py \
  step-1040128 /path/to/1040128_weights.pt /path/to/that-run/config.yaml

python agents/survival_strategic_agent/run_match.py \
  step-1040128 kits/python/simple/main.py \
  --candidate-player 0 --seed 2021 --out replays/step-1040128-p0.json
```

Run the second orientation with `--candidate-player 1`. The official CLI and opponent retain their existing Python;
only `launcher.py` switches this agent process to the bundle-local `.venv`.

This repository contains the code for the Lux AI Season 1 competition, hosted on Kaggle. The full write-up can be found on [Kaggle's forums](https://www.kaggle.com/c/lux-ai-2021/discussion/294993) and is copied below.

# Toad Brigade’s Approach - Deep Reinforcement Learning

## Introduction

Before beginning, we would like to thank the Lux AI and Kaggle teams for putting together a wonderful competition. Everyone was responsive, receptive to feedback, open about the development process, and, perhaps most importantly, the game was incredibly deep and fun to watch and strategize about. This is an impressive feat for Lux AI’s first competition, and we look forward to future seasons already! Furthermore, the Kaggle community contributed to excellent discussions of strategies, techniques, tutorials, and ranking algorithms on the forums and discord alike, and for that we are so grateful – we always learn so much from these competitions!

We initially took a multi-part approach to designing our agents: Liam worked on a rules-based agent, I, Isaiah, tackled reinforcement learning (RL), and Rob worked on meta-analyses of our games and top agent games to find weak spots and generally improve our understanding of the game and what was important in it. Our initial assumption, motivated by the top results in last year’s Halite competition, was that the rules-based approach would be more successful and would be where most of our efforts would lie. However, to our surprise, within the first month the RL approach began to beat the rules-based one, and seemed to be improving monotonically without any signs of plateauing, so the rules-based agent was abandoned from the August sprint onwards.

For those just looking for a high-level overview of what I did, I will present that information in the first part of this write-up. In the following sections, I’ll go over things more deeply and include the technical details about the pieces that I felt to be the most innovative and important. For those looking for the code, it is all open source and can be found on GitHub: [https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021), alongside our [submissions over time](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/tree/main/internal_testing/hall_of_fame).

## High Level Overview

Lux AI is a game where two players issue commands for their respective teams composed of workers, carts, and city tiles on a square grid, ranging in size from 12x12 to 32x32. The goal is to take control of and mine the available resources in order to build more cities which can, in turn, research new resources and build new workers and carts. Every 30 turns, there are 10 turns of night time during which units and cities must consume the resources they’ve amassed in order to burn fuel to survive, as anything which does not have enough fuel will disappear into the night. After 360 turns, the player who has the most surviving city tiles wins the game.

In reinforcement learning, an agent interacts with an environment repeatedly by taking actions at each turn and receiving rewards and new observations, and in so doing tries to learn the best sequence of actions given observations to maximize the expected sum of rewards. After many games of experience, the agent will hopefully learn which actions are good and lead to a positive reward, and which ones are bad and lead to a negative one. For this 2-player game, a reward was given at the final timestep of -1 for losing and +1 for winning the game, with a reward of 0 at all other times. A deep convolutional neural network parameterized an action policy (in other words, the strategy), which was trained using backpropagation to maximize the probability (specifically, the log-likelihood) of winning the game over losing it (since the game result was the only source of non-zero reward), and in the process of playing against itself many many times, learned to take good actions over bad ones at each step.

One important challenge for applying RL to this game is that there are a variable number of workers, carts, and cities on the board. One way to handle this problem would be to have a network control each unit separately, but I felt this would be inadequate as it would be challenging for the separate units to learn to work together in a harmonious fashion, and it is not so clear how to assign reward to the independent units to help them do so. I opted instead to have a single network which issued commands for each worker, cart, and city tile on all squares of the board simultaneously. I then used only the actions from the squares with units and cities that needed orders, thereby allowing the network to learn to coordinate an entire arbitrarily-sized fleet given only one reward signal.

Through this procedure, and starting from a random initialization, the neural network convincingly and consistently made improvements to its gameplay, learning novel strategic and tactical behavior independently and without human intervention. This process continued over the course of the competition with training overnight most nights, and the agent improved near-continuously, only plateauing somewhat in the final few weeks. [At first](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-22592874), the agent learned to simply harvest wood and build cities near the forests. [Next](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-22612130), it learned the importance of denying the opponent access to resources, and used a scorched earth strategy to consume all the available resources without heed for its own survival - a strategy which was enough to win the August sprint prize. However, after the rules change, (where wood regrowth was added and fuel costs were reduced to weaken swarming scorched earth strategies) [it began playing more conservatively](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-23032370) and protecting the renewable forests to maximize the resources available to it over the course of the game. Finally, by the end of the competition, the agent has become a formidable player, ruthlessly surrounding and defending the available resources – especially the valuable forests, infiltrating and stealing the opponent’s resources when given the opportunity, and waiting until the final day-night cycles to build large cities right before the game’s end. It has been very cool to watch the agent develop complex behaviours that we'd struggle to imagine implementing in a rules-based agent, especially its keen long-term strategic sense, ability to exploit small advantages, and fluid cooperation between units. Even after months of seeing it learn and play, it is still difficult to describe my sense of wonder that the agent has learned to do so much from so little signal. Such is the magic of RL.

## Model visualisations

Below is a visualisation of [episode #33947998](https://www.kaggle.com/c/lux-ai-2021/leaderboard?dialog=episodes-episode-33947998), which shows a few common patterns in the agent’s strategy. Our agent is the blue player, versus RLIAYN as red. Full credit to Liam for creating these excellent visualisations. (If the GIFs don't load, try the MP4 links)

![](https://raw.githubusercontent.com/IsaiahPressman/Kaggle_Lux_AI_2021/main/media/luxai_33947998_compressed.gif)

[Link to MP4 (for better video control)](https://drive.google.com/file/d/1X-DvDgWHNvUHkTxzmSAwSMYBWYtTKI-R/view?usp=sharing)

  

The top left of the visualisation shows a simplified version of the game state. Cities controlled by our agent (blue in this case) are highlighted according to the probability of the ‘build worker’ action: lighter means the probability is closer to one. The top right figure shows how important each cell is for the value function of each player. It was constructed by deleting the contents of each cell (resource/unit/city) and observing how much that changed the network output. The bottom figure plots the agent’s value function over time, ranging between one and minus one for an expected win or loss.

A few different observations from this match:

-   The agent invades the opponent’s area and blocks counter-attacks, and begins building an efficient long-term city structure from the mid game.
    
-   Between steps 200-300, the agent is aware that the blue cities surrounding the forest in the bottom right are actually harmful because they make it difficult to move wood away from the forest for city building.
    
-   The agent has very good forest management, keeping trees close to max capacity while skimming off wood for city building (the cell importances consistently show the two forest tiles on the map edges to be the most critical).
    
-   Apart from the forest tiles, the agent also places a large value on workers and cities deep in the opponent’s territory.
    
-   The sharp increase in the value function on step 69 appears to be due to the blue agent researching coal, although the control of coal (or uranium) doesn’t seem to be a deciding factor in the match.
    

  

The next match below ([episode #34222068](https://www.kaggle.com/c/lux-ai-2021/leaderboard?dialog=episodes-episode-34222068)) is a loss which shows some of the weaknesses of the agent (we play red in this match, RLIAYN is blue).

![](https://raw.githubusercontent.com/IsaiahPressman/Kaggle_Lux_AI_2021/main/media/luxai_34222068_compressed.gif)

[Link to MP4](https://drive.google.com/file/d/1yC6ccY3xvWWNJhNzxJLJiheD4iDG3fIx/view?usp=sharing)

-   The agent thinks it is winning initially, and it does seem to have some space advantages in the center of the board.
    
-   However, it has trouble budgeting enough fuel to keep its cities alive, and doesn’t effectively block blue from invading its side of the board.
    
-   From step 200 onwards, it feels like the agent gives up, with most workers standing around and not taking even basic productive actions (it’s possible that keeping some light reward shaping or training against a league of more diverse opponents could help the agent to be more resilient in situations like this).
    

  

The final match below ([episode #34014720](https://www.kaggle.com/c/lux-ai-2021/leaderboard?dialog=episodes-episode-34014720)) is a clear demonstration of the agent’s focus on positional advantage (our agent is red, RLIAYN is blue). The GIF just shows the first 100 steps of the match.

![](https://raw.githubusercontent.com/IsaiahPressman/Kaggle_Lux_AI_2021/main/media/luxai_34014720_compressed.gif)

[Link to MP4 of full match](https://drive.google.com/file/d/1bkiniJAfpZ8Pdi_LI1B8nhVSmrl7dn0T/view?usp=sharing)

-   The cell importance shows that, in the very early game, blue was expected to control the entire upper half of the map. But, when a row of blue’s cities disappeared during the first night, this allowed our agent to compete for this area (I think blue’s cities served two purposes: partly to directly block red’s units, and also to enable worker production to do more blocking).
    
-   Throughout the rest of the match, the agent remains very confident about victory even when it is far behind in city count.
    

# Implementation Details

## Input encoding

In order to handle the varying board sizes, I padded all boards with 0s to be of size 32x32, and masked the outputs after every convolutional layer to ensure that information did not leak from layer to layer along the manually padded edges. Additionally, for some global information, such as research points, I broadcasted the values to all cells of the board before processing. When encoding the board for the neural network, I used learnable 32-dimensional embedding layers to encode each of the discrete observation channels separately, followed by a concatenation, 1x1 convolutional layer to project to 128x32x32, and LeakyReLU activation. 32-dimensional embeddings were excessive for some of the features, since many had only 2 or 3 options, and were I to do things over, I would reduce the dimensionality of the embeddings for many of the discrete features. For the continuous observations, I first applied per-feature normalization, followed by a concatenation, 1x1 convolutional layer to project to 128x32x32, and LeakyReLU activation. Finally, the projected continuous and discrete embeddings were concatenated once again, followed by a final 1x1 convolution before passing the 128x32x32 tensor to the main residual network. The full list of features can be found [in the code](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/lux_ai/lux_gym/obs_spaces.py#L253), but I’ll go over a few here:

-   Worker: A 3-dimensional discrete variable, indicating either the absence of a worker, the presence of an allied worker, or the presence of an opposing worker. (With similar features for carts and city tiles)
    
-   Worker cooldown: A continuous variable, indicating the remaining cooldown for the worker on that square.
    
-   Worker cargo wood: A continuous variable, indicating the amount of wood in the worker’s cargo on that square. There are separate features for coal and uranium cargo.
    
-   Worker cargo full: A 2-dimensional discrete variable which is True if a worker with a full cargo is present, and False otherwise. This allowed the network to more easily detect when a worker could stop mining.
    
-   City tile fuel: A continuous variable, indicating this city tile’s share of the fuel available to the city, calculated by taking the amount of fuel in the whole city divided by the number of tiles in that city. City tile cost was treated similarly.
    
-   Wood: A continuous variable, indicating the amount of wood on the tile. (With similar features for other resources)
    
-   Distance from center X: A continuous variable, indicating the distance from the center of the board along the X axis. This should allow a given worker to better orient itself on the varying board sizes.
    
-   Research points: A continuous variable with one channel for each team, indicating the number of research points that each team has.
    
-   Researched coal: A 2-dimensional discrete variable with one channel for each team, which is True if a team has researched coal and False otherwise.
    
-   Day night cycle: 40-dimensional discrete variable, indicating the current turn number mod 40. This allowed the network to explicitly encode every timestep within the day-night cycle using a different embedding, something which may be particularly important when a worker is trying to mine as much as possible, but needs to return to the city just before nighttime.
    
-   Game phase: 9-dimensional discrete variable, indicating the turn number divided by 40 and rounded down. This allowed the network to more easily condition it’s strategy on the different phases of the game. Once I added this, the network quickly developed dramatically different behaviors during the beginning, middle, and end game, and I think this feature was a crucial part of its success.
    
-   Board size: 4-dimensional discrete variable, indicating the board size - either 12x12, 16x16, 24x24, or 32x32. This helped the network to condition its strategy on the current board size.
    

## Neural network architecture

The neural network body consisted of a fully convolutional ResNet architecture with squeeze-excitation layers. All residual network blocks used 128-channel 5x5 convolutions, and notably did not include any type of normalization. The network had four outputs consisting of three actor outputs - a 32x32xN-actions tensor for workers, carts and city tiles - and a critic output, which outputted a single value in [-1, 1]. The final network consisted of 24 residual blocks, plus the input encoder and output layers, for a grand total of ~20 million parameters.

![](https://raw.githubusercontent.com/IsaiahPressman/Kaggle_Lux_AI_2021/main/media/network_architecture.png)

## Reinforcement learning algorithm

For reinforcement learning, I used FAIR’s implementation of the IMPALA algorithm, with additional UPGO and TD-lambda loss terms. I also had a frozen teacher model perform inference on all states, and added a KL loss term for the current model’s policy from that of the teacher. This helped to stabilize behavior and prevent strategic cycles – both of which are problems that plague a pure self-play setup. Policy losses were computed by summing over the log probabilities of the selected actions for all units that acted in a given timestep, effectively computing the log of the joint probability of all the selected actions.

In order to speed up training and aid the agent in developing rudimentary behaviors despite the sparse win/loss reward signal, I performed reward shaping for the first 20 million steps, by awarding/penalizing points for building/losing cities and units, researching, and fueling cities, alongside winning/losing the game. After training a smaller 8-block network with the shaped reward, I then trained a 16-block and eventually 24-block on the sparse reward, with the smaller previous networks as teachers each time. All training was done on my personal PC - an 8-core/16-thread dual-GPU system.

The full action space was available to the agent with one exception: transfers. In order to discretize the action space, I decided to only allow a unit to transfer all of a given resource at once. This meant that workers had 19 available actions (no-op, 4 moves, 4 transfers for each of 3 resources, build city, pillage), carts had 17 (same as workers minus build city and pillage), and city tiles had 4 (no-op, research, build worker, build cart). Additionally, the agent was only allowed to take viable actions, with illegal actions masked by setting the logits to negative infinity. This did not reduce the available action space – the agent could always elect to take the no-op action – but did reduce the complexity of the learning task.

For overlapping units of the same type, I sampled the actions without replacement until a no-op, at which point all remaining units took the no-op action. This prevented multiple units from trying to exit a city to the same square, while not interfering with the agent’s ability to keep units inside at night. In order to compute the log-probabilities for stacked units, I computed the probability that the sampled actions until the no-op had occurred in that order.

## Test-time modifications

I made a few modifications to improve test-time performance. I performed a single data augmentation by rotating the observation 180 degrees, getting the action probabilities for both the actual and rotated state, and then taking the average. I would have performed additional data augmentation, but there was not enough time as the model took 2-2.5 seconds for inference on the Kaggle servers with a batch size of 2. I did not sample actions randomly at test-time, but instead always selected the most likely action.

In addition to data augmentation, I added a few handwritten rules to aid the agent’s test-time behavior. City tiles selected build and research actions in order of the probability that the model gave to take the selected action with the selected city tile. Once there were enough build or research actions queued to reach the unit or research cap respectively, the rest of the city tiles had to select a different action. Similarly, units moved in order of the probability that the model gave to taking their selected actions. Once a unit’s move action had been queued for a given square, other units were forbidden from moving to (or staying in) the target square unless it was a city tile. Further, I added a rule to some agents that after the first night, city tiles were forbidden to take the no-op action until research was complete. This rule helped stabilize behavior against weaker agents and climb the leaderboard faster, but did not seem to have much of an effect on the agent’s final performance against other top agents. For the last few agents that we submitted, we added a tie-break assistance rule that all cities must build a cart on the final turn of the game. This rule is almost always irrelevant, but it’s [amusing to see in action](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-24152545) regardless.

# Conclusion

This has been an amazing competition, and we are so grateful to the organizers and competitors alike for making it all happen. We are looking forward to reading many more write-ups over the coming weeks, and learning a ton as always. Good luck to all and we hope to see you in season 2!
