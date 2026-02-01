import random
import pandas as pd

rows = []
for _ in range(200):
    scores = [random.randint(0,10) for _ in range(10)]
    avg = sum(scores)/10
    if avg >= 8:
        risk = 0
    elif avg >= 5:
        risk = 1
    else:
        risk = 2
    rows.append(scores + [risk])
cols = [
    "picture","story","fluency","visual","logic",
    "recall","sentence","thinking","delayed","sequence","risk"
]

df = pd.DataFrame(rows, columns=cols)
df.to_csv("training_data.csv", index=False)

print("Dataset generated:", df.shape)
