import numpy as np
import csv
with open(r"/home/cyyuan/Data/Income/responses/baseincome_p31.csv", mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        for i,row in enumerate(reader):
                if row["responses"] == row["gts"]:
                        cnt= cnt +1

        print(cnt/(i))