# Deployment Backend Runtime Lessons

The ComputingPlatform deployment backend contains a remote SSH launcher and a large one-click training runner. GPUCLOUD worker mode should not import that launcher. Extract only the runtime rules that help a child agent prepare its local environment.

## Fields to Preserve

Map backend/runtime settings into structured worker task fields:

- `auto_install` -> `environment.auto_install`
- `pip_index_url`, `pip_extra_index_url`, `pip_trusted_host` -> `environment` and `runtime.env`
- `hf_endpoint` -> `environment.hf_endpoint` and `HF_ENDPOINT`
- `docker.enabled` -> `docker.enabled` or a validation warning if worker-local Docker is unsupported
- `workdir` or `cwd` -> worker-local working directory under the data disk
- `dataset_config`, uploaded dataset file, and dataset format -> `training.dataset_config` or `training.dataset`
- `setup_command` and `activate_command` -> preflight guidance, not hidden training command generation

## Runtime Repair Rules

The one-click backend repaired missing conda environments, resolved Python paths from activation commands, selected torch wheels by driver CUDA compatibility, and installed HF stack versions compatible with torch. GPUCLOUD should keep these as preflight/setup behaviors, not as LLM-generated command templates.

Recommended worker checks:

- Resolve the actual Python executable after activation.
- Probe torch version, CUDA version, device count, and NCCL availability.
- Use mirror defaults when Chinese network paths are configured.
- Keep HF caches and dataset caches on the data disk.
- Log the effective runtime paths, package versions, and cache roots.

## Process and Log Behavior

The backend detached long training jobs, wrote logs, tracked process liveness, tailed incrementally, parsed loss lines, and converted exit status into job status. GPUCLOUD worker-local mode should follow the same reliability shape with local subprocess wrappers:

- stdout/stderr go to job log files.
- exit codes go to a sidecar `.exitcode` file.
- status reads pid plus exit code.
- log APIs return tails, not entire logs.
- stop only signals local pid or process group.

## What Not to Migrate

Do not copy backend `.env`, DB/WAL files, storage artifacts, export tarballs, cached datasets, model weights, or SSH credentials into GPUCLOUD skills. The zip is a source of field and workflow knowledge only.
