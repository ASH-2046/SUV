# Odyssey closed-loop client

SUV runs as the Odyssey planner. The simulator, Fixer, and scene assets stay in
the Odyssey checkout. This adapter reads the front camera at 2 Hz and returns
eight ego-frame poses at 0.5 s.

On the H20 host, after `source ~/odyssey_env.sh`:

```bash
cd "$ODYSSEY_ROOT"
python -m odyssey_runtime check --agent "$SUV_ROOT/experiments/odyssey/suv.yaml" --gpu 4
python -m odyssey_runtime run --agent "$SUV_ROOT/experiments/odyssey/suv.yaml" \
  --scene odyssey_scene001 --react nr --gpu 4 --max-steps 40
```

`SUV_NUM_INFERENCE_STEPS` defaults to 2 for a smoke test. The NAVSIM evaluation
uses 10. SUV does not read Odyssey's SD route; the driving command is the
NAVSIM one-hot carried in the ego status.
