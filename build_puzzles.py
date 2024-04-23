import pandas as pd
import zstandard as zstd
import io
import chess
import pickle
from tqdm import tqdm

tqdm.pandas()

PUZZLE_PATH = "data/lichess_db_puzzle.csv.zst"


def build_puzzles(row):
    fen = row["FEN"]
    moves = row["Moves"].split(" ")
    board = chess.Board(fen)
    board.set_fen(fen)
    board.push_uci(moves[0])

    puzzles = []
    for move in moves[1:]:
        puzz = {"fen": board.fen(), "best_move": move}
        puzzles.append(puzz)
        board.push_uci(move)
    return puzzles


# Open the file in read binary mode
print("Reading the compressed file...")
with open(PUZZLE_PATH, "rb") as file:
    # Create a Zstandard decompressor object
    dctx = zstd.ZstdDecompressor()
    # Decompress the file into a bytes object
    with io.BytesIO() as out_buf:
        dctx.copy_stream(file, out_buf)
        # Read the decompressed bytes into a pandas DataFrame
        df = pd.read_csv(io.BytesIO(out_buf.getvalue()))

# Apply the build_puzzles function to each row of the DataFrame
print("Building the puzzles...")
puzzle_list = df.progress_apply(build_puzzles, axis=1)

# Concatenate the results into a single list
print("Concatenating the puzzles...")
puzzles = [p for sublist in puzzle_list for p in sublist]

# Save the puzzles to a file
print("Saving the puzzles...")
with open("data/lichess_puzzles.pkl", "wb") as file:
    pickle.dump(puzzles, file)

print("Done!")
