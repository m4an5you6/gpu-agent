# Backend Training Contract

The ComputingPlatform deployment backend exposes training jobs through structured fields. GPUCLOUD main/worker integration should preserve these fields instead of flattening them into shell strings.

## Important Schemas

`TrainingCreate`:

- `gpu_id`
- `model_name`
- `dataset_name`
- `required_gpu_type`
- `auto_start`
- `config_json`
- `user_id`

`TrainingOut`:

- `id`
- `gpu_id`
- `status`
- `model_name`
- `dataset_name`
- `required_gpu_type`
- `created_at`, `started_at`, `ended_at`
- `config_json`
- `training_gpus`
- `training_gpu_display`
- `weight_path`

`TrainingGpuInfo`:

- `gpu_id`
- `node_id`
- `node_name`
- `host`
- `internal_ip`
- `node_gpu_index`
- `gpu_name`
- `display`
- `role`
- `weight_path`
- `weight_path_verified`

`GpuOut`:

- `id`
- `node_id`
- `node_gpu_index`
- `workload_kind`
- `workload_ref_id`
- `allocate_user`

`NodeOut`:

- `id`
- `name`
- `provider`
- `host`
- `internal_ip`
- `port`
- `username`
- `auth_method`
- `status`
- `gpu_count`
- `gpu_types`
- `last_error`

## Endpoint Lifecycle

The backend contains routes for creating training jobs, starting jobs, syncing logs, exporting logs, deleting finished jobs, and exporting artifacts. GPUCLOUD main should treat backend responses as allocation/job metadata and should not double-start a training job if worker WSS dispatch is already responsible for local execution.

The backend may wrap results as `{status, statusText, data}` or as paginated `{records, total, current, page_size}`. Tool code should unwrap these shapes before constructing worker tasks.

## Config JSON Mapping

Common `config_json` keys to preserve:

- `auto_install`
- `training_type`
- `batch_size`
- `learning_rate`
- `max_steps`
- `distributed`
- `dataset_config`
- `docker.enabled`
- `hrl3d.enabled`
- `hrl3d.master_node_id`
- `hrl3d.master_addr`
- `hrl3d.master_port`
- `megatron.enabled`
- `megatron.backend`
- `megatron.auto_setup`
- `megatron.swift`
- mirror fields such as `pip_index_url`, `pip_extra_index_url`, `pip_trusted_host`, and `hf_endpoint`

Unknown fields should be carried through under structured config or reported as validation warnings; do not silently turn them into shell fragments.
