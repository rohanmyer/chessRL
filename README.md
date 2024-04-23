# chessRL

## Setup

### 1. Setup Filestructure

#### Install Dependencies

`pip install -r requirements.txt`

#### Create data and weights folders

From the root of this project, `mkdir data` and `mkdir weights`.

### 1. Download Stockfish

Download the appropriate Stockfish binaries:

`cd data`

#### Linux: 
`wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar`

`tar -xvf stockfish-ubuntu-x86-64-avx2.tar`

#### MacOS: 
`wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-macos-m1-apple-silicon.tar`

### 2. Download Puzzles

Download the puzzles from https://database.lichess.org/lichess_db_puzzle.csv.zst

`cd data`

`wget https://database.lichess.org/lichess_db_puzzle.csv.zst`

### 3. Build Puzzles

Specify the path to the puzzle file in `build_puzzles.py` and then run `python build_puzzles.py`

## Train With Puzzles

To train an RL agent to play from puzzle scenarios, run `python puzzle_train.py`

## Train Against Stockfish

To train an RL agent to play from full games against the Stockfish engine, run `python train_agent.py`
