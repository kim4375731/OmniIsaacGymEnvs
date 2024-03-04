from skrl.envs.torch import wrap_env
from skrl.utils.omniverse_isaacgym_utils import get_env_instance

import hydra
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    # get environment instance
    env = get_env_instance(headless=True)
    
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig    
    
    sim_config = SimConfig(cfg)
    
    # import and setup custom task
    from omniisaacgymenvs.tasks.anymal import AnymalTask
    cfg._task_cfg.get("renderingInterval", 1)    
    task = AnymalTask(name="Anymal", sim_config=cfg._task_cfg, env=env)

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

    # wrap the environment
    env = wrap_env(env, "omniverse-isaacgym")        


if __name__ == "__main__":
    parse_hydra_configs()
