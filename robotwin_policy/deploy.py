"""Env-side loop used by XPolicyLab. RoboTwin's own client is the smoke path."""


def eval_one_episode(task_env, model_client):
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        obs = task_env.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        for action_idx, action in enumerate(actions):
            task_env.take_action(action)
            if task_env.is_episode_end() or action_idx + 1 == len(actions):
                break


def eval_one_episode_batch(task_env, model_client):
    raise NotImplementedError("scripted_vla smoke eval is single-env only")
