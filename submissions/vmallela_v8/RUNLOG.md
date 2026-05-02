# v8 run log

Append-only operational log for the v8 build. Every phase decision, gate
result, auto-tuning attempt, timing, and halt reason gets logged here with
timestamps. Read this top-to-bottom for the morning summary.

Format:
```
[YYYY-MM-DDTHH:MM:SSZ] <component> <event> <details>
```

---

