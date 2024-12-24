from notorch.types import TargetTransformConfig, TaskTransformConfig


def build_transform_config(
    transform_key_map: dict[str, dict[str, str]], task_transforms: dict[str, TaskTransformConfig]
) -> dict[str, TargetTransformConfig]:
    return {
        target_group: {
            mode: {
                "module": task_transforms[target_group][mode],
                "key": transform_key_map[target_group][mode],
            }
            for mode in ["preds", "targets"]
            if mode in transform_key_map[target_group]
        }
        for target_group in (transform_key_map or dict()).keys()
    }
