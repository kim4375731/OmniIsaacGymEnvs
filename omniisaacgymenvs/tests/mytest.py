import omni
from omni.isaac.kit import SimulationApp

SimulationApp({"renderer": "RayTracedLighting", "headless": False})
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})
# simulation_app.update()

# from omni.isaac.core.utils.extensions import enable_extension
# from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core import World
# from omni.isaac.core.objects import DynamicCuboid
# from omni.isaac.wheeled_robots.robots import WheeledRobot
# from omni.isaac.occupancy_map import _occupancy_map
# from omni.isaac.occupancy_map.scripts.utils import update_location, compute_coordinates, generate_image
# import numpy as np

# from omniisaacgymenvs.robots.articulations.warthog import Warthog

def main():     
    # timeline = omni.timeline.get_timeline_interface()
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()    
    # primpath = "/World/Warthog"
    # warthog = world.scene.add(Warthog(primpath))
    world.reset()
    
    # timeline.play()
    # while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()

    # timeline.stop()
    # simulation_app.close()
    

if __name__ == "__main__":
    main()