python ./src/stableshot/select_conf.py \
  --trace-metrics ./results/stable_shots_grid/trace_metrics.csv \
  --target-tvd 0.05 \
  --min-success-rate 1.0 \
  --risk-metric max \
  --objective min_median_shots \
