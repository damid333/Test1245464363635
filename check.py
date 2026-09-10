import pandas as pd

by_task = pd.read_csv('audit/by_task.csv')
by_task['states'] = by_task['states'].fillna('')

want = by_task[by_task['states'].str.contains('completed|in progress', regex=True)]
print(len(want), 'тасок,', int(want['boxes'].sum()), 'боксов')
print(want[['task_id','task_name','frames','boxes','states']].to_string(index=False))

want[['task_id','task_name']].to_csv('to_restore.csv', index=False)


import os, pandas as pd
from cvat_sdk import make_client

ids = pd.read_csv('to_restore.csv')['task_id'].astype(int).tolist()
alive, gone = [], []

with make_client(host=os.environ['CVAT_HOST'],
                 credentials=(os.environ['CVAT_USER'], os.environ['CVAT_PASS'])) as c:
    for tid in ids:
        try:
            c.tasks.retrieve(tid)
            alive.append(tid)
        except Exception:
            gone.append(tid)

print(f'живых: {len(alive)}, удалено: {len(gone)}')
if alive:
    print('эти трогать не надо:', alive)

pd.read_csv('to_restore.csv').query('task_id in @gone').to_csv('to_restore.csv', index=False)
