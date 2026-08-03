# packages
import json
import pandas as pd
from tqdm import tqdm 
from pathlib import Path

# json paths
chat_dir = Path("Chat_Random")
json_files = sorted(chat_dir.rglob("*.json"))

# extract data
demo_rows = []
turn_rows = {tid: [] for tid in range(12)}  

for p in tqdm(json_files):
    with p.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    sid = obj.get("ID")
    cond = obj.get("Condition")
    role = obj.get("Role")
    turns = obj.get("turns", [])

    demo_rows.append({"ID": sid, "Condition": cond, "Role": role})

    by_tid = {t.get("turn_id"): t for t in turns if isinstance(t, dict)}

    for tid in range(12):
        bot_ans = None
        if tid in by_tid:
            bot_ans = by_tid[tid].get("bot", None)
        turn_rows[tid].append({"ID": sid, "bot": bot_ans.strip()})

demo_df = pd.DataFrame(demo_rows)
turn_dfs = {tid: pd.DataFrame(rows) for tid, rows in turn_rows.items()}
assert not demo_df["ID"].duplicated().any()

# save 
demo_df.to_excel('Chat_Random/summary/ids.xlsx', index=False)

turn_dfs[0].to_excel('Chat_Random/summary/role_play.xlsx', index=False)

turn_dfs[1].to_excel('Chat_Random/summary/immediate_recall.xlsx', index=False)
turn_dfs[11].to_excel('Chat_Random/summary/delayed_recall.xlsx', index=False)

turn_dfs[2].bot = turn_dfs[2].bot.apply(lambda x: x.lower())
turn_dfs[2].to_excel('Chat_Random/summary/cat_fluency.xlsx', index=False)
turn_dfs[3].bot = turn_dfs[3].bot.apply(lambda x: x.lower())
turn_dfs[3].to_excel('Chat_Random/summary/let_fluency.xlsx', index=False)

turn_dfs[4].to_excel('Chat_Random/summary/dg_forward.xlsx', index=False)
turn_dfs[5].to_excel('Chat_Random/summary/dg_backward.xlsx', index=False)
turn_dfs[6].to_excel('Chat_Random/summary/dglt_forward.xlsx', index=False)
turn_dfs[7].to_excel('Chat_Random/summary/dglt_backward.xlsx', index=False)

turn_dfs[8].to_excel('Chat_Random/summary/procedure.xlsx', index=False)

turn_dfs[9].to_excel('Chat_Random/summary/scene.xlsx', index=False)

turn_dfs[10].to_excel('Chat_Random/summary/coreference.xlsx', index=False)



























