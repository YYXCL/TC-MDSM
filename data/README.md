# Foundation data

The runner reads `compute_load/compute_node_load_timeseries.csv`, `agent_capabilities/agent_capabilities.json`, and `tasks/tasks.json`. `model/data.py` constructs decision snapshots, workflows, and logged observations from these inputs. Training, validation, and test partitions remain separate.

Dataset field names, scenario labels, skill IDs, permissions, and task text retain their source representation because they participate in lookup and text encoding.

`raw/Alibaba-cluster-trace-v2026-GenAI/source/` contains Alibaba GenAI trace CSV inputs. `capability_sources/` contains source capability cards. `dataset_manifest.json` records foundation-dataset provenance. The capability-generation utility updates capability records separately; this manifest describes the original foundation generation, not that later update.

```sh
python data/scripts/build_alibaba_dataset0.py
python data/scripts/generate_simulation_datasets.py --help
python tools/tc_mdsm_v3_capabilities.py --help
```

The dataset generator writes foundation tables and task templates. It is not required to run the experiment with the supplied inputs. Regenerating foundation tables overwrites them, including current capability records; preserve a copy before doing so.

The trace does not identify physical servers: virtual compute nodes use a deterministic mapping from container identifiers. Link states, capability assignments, workflows, and execution behavior contain synthetic simulation assumptions. They should not be interpreted as measurements of deployed agents. `compute_load/dataset0.csv` is the aggregated trace table. Source archives, the workbook export, and redundant table copies are omitted.
