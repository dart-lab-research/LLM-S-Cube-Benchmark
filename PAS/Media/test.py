
import csv
response1,response2,responsefew1,responsefew2,gt=1,1,1,1,1

with open('/home/cyyuan/ACL2025/Data/Anes2020/duolungpt/biden_round2_inrow.csv', mode='w', newline='') as file:
    # 写入数据行
    writer = csv.writer(file)
    writer.writerow(['response1', 'response2', 'responsefew2', 'responsefew1', 'gt'])
    print(f"write already, ")

with open('/home/cyyuan/ACL2025/Data/Anes2020/duolungpt/biden_round2_inrow.csv', mode='a', newline='') as file:
    # 写入数据行
    writer = csv.writer(file)
    writer.writerow([response1, response2, responsefew2, responsefew1, gt])
    print(f"write already, ")