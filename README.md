# chessRL

## Setup

### 1. Setup Filestructure

#### Install Dependencies

`pip install -r requirements.txt`

#### Create data and weights folders

Create a `data` folder from the root of this project. Create two folders, one for `logs` and one for `weights` inside `rl_chess`.

### 1. Download Stockfish

Download the appropriate Stockfish binaries inside the `data` folder:

#### Linux: 
`wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar`

`tar -xvf stockfish-ubuntu-x86-64-avx2.tar`

#### MacOS: 
`wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-macos-m1-apple-silicon.tar`

### 2. Download Puzzles

Download the puzzles from https://database.lichess.org/lichess_db_puzzle.csv.zst

From the `data` folder:

`wget https://database.lichess.org/lichess_db_puzzle.csv.zst`

### 3. Build Puzzles

Specify the path to the puzzle file in `build_puzzles.py` and then run `python build_puzzles.py`

## Train From Self-Play

To train an RL agent to play by playing against itself, run `python main.py self`

## Train With Puzzles

To train an RL agent to play from puzzle scenarios, run `python main.py puzzle`

## Train Against Stockfish

To train an RL agent to play from full games against the Stockfish engine, run `python main.py engine`

## Models

1. Self-Play
2. Engine
3. Puzzle
4. Self-Play + Puzzle
5. Engine + Puzzle
6. Puzzle + Self-Play
7. Puzzle + Engine