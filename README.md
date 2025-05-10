# 📊 پروژه تحلیل فرآیند با PM4Py

این پروژه یک نمونه ساده از تحلیل فرآیند با استفاده از کتابخانه [PM4Py](https://pm4py.fit.fraunhofer.de/) در زبان Python است. در این پروژه:

- یک لاگ فرآیند ساده به‌صورت دستی تعریف می‌شود.
- با استفاده از الگوریتم **Heuristics Miner**، مدل فرآیندی استخراج می‌شود.
- از **Token Replay** برای بررسی تطابق لاگ با مدل فرآیند استفاده می‌شود.
- مدل نهایی به صورت دیداری نمایش داده می‌شود.

---

## 📁 پیش‌نیازها

قبل از اجرای کد، مطمئن شوید که پایتون 3.8 یا بالاتر نصب است و بسته‌های زیر را نصب کرده‌اید:

```bash
pip install pm4py pandas
```

---

## ▶️ اجرای کد نمونه

```python
import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.visualization.petri_net import visualizer as pn_vis
import pandas as pd

# تعریف لاگ فرآیند به صورت دستی
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

# تبدیل به DataFrame و فرمت لاگ
df = pd.DataFrame(log_data)
log = pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
event_log = pm4py.convert_to_event_log(log)

# استخراج مدل فرآیند با Heuristics Miner
net, im, fm = heuristics_miner.apply(event_log)

# بررسی تطابق (Token Replay)
parameters = {}
replayed_traces = token_replay.apply(event_log, net, im, fm, parameters=parameters)
print(f"Token Replay Results: {replayed_traces}")

# نمایش مدل شبکه پتری
gviz = pn_vis.apply(net, im, fm)
pn_vis.view(gviz)
```

---

## 📌 توضیحات

- **Heuristics Miner**: برای استخراج مدل فرآیند از لاگ‌ها استفاده می‌شود و توانایی کار با داده‌های نویزی را دارد.
- **Token Replay**: ابزاری برای بررسی میزان تطابق لاگ واقعی با مدل استخراج‌شده.
- **Petri Net Visualization**: مدل استخراج‌شده به صورت دیداری نمایش داده می‌شود تا تحلیل فرآیند راحت‌تر شود.

---

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است.

---

## 🤝 مشارکت

در صورت علاقه‌مندی برای گسترش این پروژه، خوشحال می‌شوم مشارکت کنید. Pull Request ارسال کنید یا Issue جدید باز کنید.