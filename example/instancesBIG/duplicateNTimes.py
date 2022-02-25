import pandas as pd
import sys

db1o = pd.read_csv('base_inst1_original.csv')
db2o = pd.read_csv('base_inst2_original.csv')

n = int(sys.argv[1])

db1f = pd.concat([db1o] * n)
db2f = pd.concat([db2o] * n)

print(f"db1f size {db1f.shape[0]}, db2f size {db2f.shape[0]}")

db1f.to_csv("base_inst1.csv", index=False)
db2f.to_csv("base_inst2.csv", index=False)