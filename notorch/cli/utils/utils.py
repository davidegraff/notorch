from notorch.types import GroupTransformConfig, TaskTransformConfig


def build_group_transform_configs(
    transform_key_map: dict[str, dict[str, str]], task_transforms: dict[str, TaskTransformConfig]
) -> dict[str, GroupTransformConfig]:
    """Build the group transform configs by merging their corresponding key map and task transforms.

    Parameters
    ----------
    transform_key_map : dict[str, dict[str, str]]
        A mapping from a task name to a dictionary with two keys, `"preds"` and `"targets"`, that
        maps to the corresponding key to modify in the tensordict.
    task_transforms : dict[str, TaskTransformConfig]
        a mapping from a task name to a dictionary with two keys, `"preds"` and `"targets"`, and the
        corresponding transform.

    Returns
    -------
    dict[str, GroupTransformConfig]
        the transform config, a mapping from task name to a nested dictionary. The outer dict
        contains two keys, `"preds"` and `"targets"`, and each inner dict contains the following
        key-value pairs:

        - ``"module"``: any ``Callable``, typically a :class:`~torch.nn.Module`, that has **no
        learnable parameters** that will be applied to the corresponding `key` in the input
        - ``"key"``: the key in the tensordict whose value will be _modified in place_
    """

    return {
        task_name: {
            mode: {
                "module": task_transforms[task_name][mode],
                "key": transform_key_map[task_name][mode],
            }
            for mode in ["preds", "targets"]
            if mode in transform_key_map[task_name]
        }
        for task_name in (transform_key_map or dict()).keys()
    }
