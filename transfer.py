from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="export tensorboard data")
parser.add_argument("--in-path", type=str, required=True, help="event files")
parser.add_argument("--out-path", type=str, required=True, help="out csv files")

args = parser.parse_args()
event_data = event_accumulator.EventAccumulator(args.in_path)
event_data.Reload()
keys = event_data.scalars.Keys()
df = pd.DataFrame(columns=keys[:])
for key in tqdm(keys):
    df[key] = pd.DataFrame(event_data.Scalars(key)).value_counts()
df.to_csv(args.out_path)
print("Tensorboard data exported successfully")
