import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Two cubes colliding with squash effect.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""
usage:
    cd IsaacLab
    isaaclab.bat -p scripts/collision_squash.py
"""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """İki küp sahnesi: biri solda, biri sağda, ortada çarpışacaklar."""

    # Zemin — yüksek restitution squash için iyi
    cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.3)
    )
    cfg.func("/World/defaultGroundPlane", cfg)

    # Işık
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # İki küpün spawn origin'leri
    # Sol küp: x=-1.0, Sağ küp: x=+1.0, ikisi de y=0, z=0
    origins = [
        [-0.5, 0.0, 0.0],   # Sol küp
        [ 0.5, 0.0, 0.0],   # Sağ küp
    ]

    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Her iki küp için aynı deformable cuboid config
    # Düşük youngs_modulus --> daha yumuşak/squashy davranış
    cube_cfg = sim_utils.MeshCuboidCfg(
        size=(0.3, 0.3, 0.3),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0,
            contact_offset=0.001,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            poissons_ratio=0.48,   # 0.5'e yakın → incompressible → squash etkisi
            youngs_modulus=1e4,  
        ),
    )

    objects = []
    for i in range(2):
        cfg = DeformableObjectCfg(
            prim_path=f"/World/Origin{i}/Cube",
            spawn=cube_cfg,
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 0.15)),  # z=0.15 → zemin üstünde
        )
        objects.append(DeformableObject(cfg))

    return {"objects": objects}, origins


def run_simulator(sim: SimulationContext, entities: dict, origins: torch.Tensor):
    objects = entities["objects"]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Küplerin birbirine doğru hızları: sol küp +x, sağ küp -x
    initial_velocities = [
        torch.tensor([6.0, 0.0, 0.0]),    
        torch.tensor([-6.0, 0.0, 0.0]), 
    ]

    while simulation_app.is_running():

        if count % 400 == 0:
            sim_time = 0.0
            count = 0
            print("----------------------------------------")
            print("[INFO]: Resetting — cubes moving toward each other...")

            for i, obj in enumerate(objects):
                nodal_state = obj.data.default_nodal_state_w.clone()

                # Başlangıç pozisyonu: origin'den spawn
                pos_w = origins[i].unsqueeze(0).clone()
                pos_w[..., 2] += 0.15   # Yerden kaldır
                quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)

                nodal_state[..., :3] = obj.transform_nodal_pos(
                    nodal_state[..., :3], pos_w, quat_w
                )

                vel = initial_velocities[i].to(sim.device)
                nodal_state[..., 3:6] = vel.unsqueeze(0).unsqueeze(0).expand_as(
                    nodal_state[..., 3:6]
                )

                obj.write_nodal_state_to_sim(nodal_state)
                obj.reset()

        for obj in objects:
            obj.write_data_to_sim()

        sim.step()
        sim_time += sim_dt

        for obj in objects:
            obj.update(sim_dt)

        # Her 50 adımda bir kısa durum logu
        if count % 50 == 0:
            for i, obj in enumerate(objects):
                mean_pos = obj.data.nodal_pos_w.mean(dim=1)  # (num_envs, 3)
                mean_vel = obj.data.nodal_vel_w.mean(dim=1)
                print(
                    f"[t={sim_time:.2f}s] Cube {i} | "
                    f"pos: {mean_pos[0].cpu().numpy().round(3)} | "
                    f"vel: {mean_vel[0].cpu().numpy().round(3)}"
                )

        count += 1


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[0.0, -4.0, 1.5], target=[0.0, 0.0, 0.15])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()
    print("[INFO]: Setup complete — watch the squash collision!")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
