# ComputingPlatform Deployment-master Inference Status

The `ComputingPlatform-Deployment-master.zip` inference runner is currently disabled.

The source defines statuses as `disabled` and raises:

```text
inference disabled: app.models.inference_artifact and app.models.inference_deployment are not installed
```

Disabled functions include:

- `prepare_inference_artifact_for_job`
- `create_or_start_deployment`
- `run_inference_deployment`
- `spawn_inference_deployment_worker`
- `stop_inference_deployment`
- `refresh_inference_deployment`
- vLLM runtime helpers

## Consequence for GPUCLOUD

Do not claim the one-click deployment backend already provides a working inference deployment API. For now:

- Worker-local vLLM deployment is the reliable path.
- Backend artifact export can be treated as a source of trained output metadata.
- UI/backend status writeback needs a separate implemented endpoint or must read GPUCLOUD main worker endpoints.
- If a user asks for deployment through the backend, first verify that inference models/routes are installed in the active backend.
