from notorch.types import GroupTransformConfig, TaskTransformConfig


def build_group_transform_configs(
    transform_key_map: dict[str, dict[str, str]], task_transforms: dict[str, TaskTransformConfig]
) -> dict[str, GroupTransformConfig]:
    return {
        target_name: {
            mode: {
                "module": task_transforms[target_name][mode],
                "key": transform_key_map[target_name][mode],
            }
            for mode in ["preds", "targets"]
            if mode in transform_key_map[target_name]
        }
        for target_name in (transform_key_map or dict()).keys()
    }
