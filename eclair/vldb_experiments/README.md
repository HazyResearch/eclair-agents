# Experiments for VLDB Paper

## Demonstrate

### SOP Generation

See `eclair/vldb_experiments/demonstrate_sop_generation/`

## Execute

### SOP

See `eclair/vldb_experiments/execute_experiments/`
`
### Grounding

See `eclair/vldb_experiments/execute_grounding/`

## Validate

See `eclair/vldb_experiments/validate_experiments/`

### Trace Cleanup

Run in order:

```bash
cd trace_cleanup/
trace_truncate_non_chrome_apps.py
trace_merge_scrolls.py
trace_fix_coords.py
screenshots_resample_keystrokes.py
trace_add_webarena_task_json.py
trace_add_gt_labels.py
trace_force_add_gt_element.py
```
