
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

""" usage: 
    cd IsaacLab
    isaaclab.bat -p scripts/demos/def_body.py 
"""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)


    shape_cfgs = [
        sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        sim_utils.MeshSphereCfg(
            radius=0.15, 
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=5e3),
        ),
        sim_utils.MeshCylinderCfg(
            radius=0.15,
            height=0.3,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=5e3),
        ),
    ]

    objects = []

    for i, shape in enumerate(shape_cfgs):

        cfg = DeformableObjectCfg(
            prim_path=f"/World/Origin{i}/Object",
            spawn=shape,
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0,0,1))
        )

        objects.append(DeformableObject(cfg))

    # returning to the objects we created

    scene_entities = {"objects": objects}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    objects = entities["objects"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            sim_time = 0.0
            count = 0

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")

            for i, obj in enumerate(objects):

                nodal_state = obj.data.default_nodal_state_w.clone()

                pos_w = torch.rand(1, 3, device=sim.device) * 0.1 + origins[i]
                quat_w = math_utils.random_orientation(1, device=sim.device)

                nodal_state[..., :3] = obj.transform_nodal_pos(
                    nodal_state[..., :3], pos_w, quat_w
                )

                obj.write_nodal_state_to_sim(nodal_state)
                obj.reset()
            

    # since i do not have one instance we should go over every items in the object list
        for obj in objects:
            obj.write_data_to_sim()

        sim.step()

        for obj in objects:
            obj.update(sim_dt)

        count += 1
        


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()