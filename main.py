import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.visualization.petri_net import visualizer as pn_vis
import pandas as pd

# ساخت دیتافریم لاگ
log_data = [
    {"case:concept:name": "A", "concept:name": "start", "time:timestamp": "2025-04-11 09:00:00"},
    {"case:concept:name": "A", "concept:name": "process", "time:timestamp": "2025-04-11 09:05:00"},
    {"case:concept:name": "A", "concept:name": "end", "time:timestamp": "2025-04-11 09:10:00"},
    {"case:concept:name": "B", "concept:name": "start", "time:timestamp": "2025-04-11 09:02:00"},
    {"case:concept:name": "B", "concept:name": "process", "time:timestamp": "2025-04-11 09:07:00"},
    {"case:concept:name": "B", "concept:name": "end", "time:timestamp": "2025-04-11 09:12:00"},
    {"case:concept:name": "C", "concept:name": "start", "time:timestamp": "2025-04-11 09:03:00"},
    {"case:concept:name": "C", "concept:name": "process", "time:timestamp": "2025-04-11 09:08:00"},
    {"case:concept:name": "C", "concept:name": "end", "time:timestamp": "2025-04-11 09:13:00"},
]

df = pd.DataFrame(log_data)

# فرمت بندی دیتا
log = pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
event_log = pm4py.convert_to_event_log(log)

# کشف مدل فرآیند با Heuristics Miner
net, im, fm = heuristics_miner.apply(event_log)

# اجرای Token-Based Replay
parameters = {}
replayed_traces = token_replay.apply(event_log, net, im, fm, parameters=parameters)
print(f"Token Replay Results: {replayed_traces}")

# رسم مدل
gviz = pn_vis.apply(net, im, fm)
pn_vis.view(gviz)
