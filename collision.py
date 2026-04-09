"""
Needle Penetration Simulation - Isaac Lab (4.x+)
================================================
3 cisim:
  - Arm  : rigid body küp  → soft tissue'ya çarpar, squash etkisi
  - Needle: rigid silindir → soft tissue içinden GEÇer (collision yok)
  - Soft  : deformable cube (soft tissue)

Hareket: Arm + Needle birlikte X ekseninde ±1 m arasında ileri-geri.

Gereksinimler: Isaac Lab 4.x  (omni.isaac.core KULLANILMAZ)
Çalıştırma:
  python collision.py --headless          # GPU render yok
  python collision.py                     # GUI
"""

import argparse
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1. AppLauncher — her zaman en başta
# ──────────────────────────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Needle Penetration Sim")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ──────────────────────────────────────────────────────────────────────────────
# 2. Isaac Lab / USD imports  (AppLauncher başladıktan SONRA)
# ──────────────────────────────────────────────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg

from pxr import (
    Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
)
import omni.usd
import carb

# ──────────────────────────────────────────────────────────────────────────────
# 3. Simulation Context
# ──────────────────────────────────────────────────────────────────────────────
sim_cfg = SimulationCfg(
    dt=1 / 120,
    render_interval=2,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
    ),
)
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=[0.0, -3.0, 1.5], target=[0.0, 0.0, 0.0])

stage = omni.usd.get_context().get_stage()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Zemin
# ──────────────────────────────────────────────────────────────────────────────
ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/Ground", ground_cfg)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Işık
# ──────────────────────────────────────────────────────────────────────────────
light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 1.0))
light_cfg.func("/World/Light", light_cfg)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _set_pos(prim_path: str, xyz):
    """XformCommonAPI ile pozisyon ata (translate op)."""
    prim = stage.GetPrimAtPath(prim_path)
    xf = UsdGeom.XformCommonAPI(prim)
    xf.SetTranslate(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _get_pos(prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def _add_material(prim_path: str, color: tuple):
    """Basit OmniPBR / Display Color."""
    mat_path = prim_path + "/Mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


# ──────────────────────────────────────────────────────────────────────────────
# 7. SOFT TISSUE (Deformable Cube)
# ──────────────────────────────────────────────────────────────────────────────
SOFT_PATH = "/World/Soft"

soft_cube = UsdGeom.Cube.Define(stage, SOFT_PATH)
soft_cube.CreateSizeAttr(1.0)          # birim küp
soft_prim = stage.GetPrimAtPath(SOFT_PATH)

# Pozisyon: orta (0,0,0) — tissue sabit durur
xf_soft = UsdGeom.XformCommonAPI(soft_prim)
xf_soft.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.4))
xf_soft.SetScale(Gf.Vec3f(0.4, 0.4, 0.4))  # 0.4 m küp

# Deformable API
deform_api = PhysxSchema.PhysxDeformableBodyAPI.Apply(soft_prim)
deform_api.GetSelfCollisionAttr().Set(True)
deform_api.CreateSolverPositionIterationCountAttr().Set(20)
# Yumuşaklık parametreleri
deform_api.CreateYoungsModulusAttr().Set(1e3)    # çok yumuşak tissue
deform_api.CreatePoissonsRatioAttr().Set(0.45)
deform_api.CreateDampingScaleAttr().Set(0.05)

# PhysxCollision
phys_col = PhysxSchema.PhysxCollisionAPI.Apply(soft_prim)
phys_col.CreateContactOffsetAttr().Set(0.01)
phys_col.CreateRestOffsetAttr().Set(0.002)

# Görünüm
_add_material(SOFT_PATH, (0.9, 0.4, 0.4))   # kırmızımsı tissue

# ──────────────────────────────────────────────────────────────────────────────
# 8. ARM (Rigid Cube)
# ──────────────────────────────────────────────────────────────────────────────
ARM_PATH = "/World/Arm"

arm_cube = UsdGeom.Cube.Define(stage, ARM_PATH)
arm_cube.CreateSizeAttr(1.0)
arm_prim = stage.GetPrimAtPath(ARM_PATH)

xf_arm = UsdGeom.XformCommonAPI(arm_prim)
xf_arm.SetTranslate(Gf.Vec3d(-0.8, 0.0, 0.4))
xf_arm.SetScale(Gf.Vec3f(0.18, 0.18, 0.18))

# Rigid body — kinematic (script tarafından hareket ettirileceği için)
rb_arm = UsdPhysics.RigidBodyAPI.Apply(arm_prim)
rb_arm.CreateKinematicEnabledAttr().Set(True)
UsdPhysics.CollisionAPI.Apply(arm_prim)
PhysxSchema.PhysxRigidBodyAPI.Apply(arm_prim)

_add_material(ARM_PATH, (0.2, 0.3, 0.9))   # mavi

# ──────────────────────────────────────────────────────────────────────────────
# 9. NEEDLE (Kinematic Cylinder)
# ──────────────────────────────────────────────────────────────────────────────
NEEDLE_PATH = "/World/Needle"

needle_cyl = UsdGeom.Cylinder.Define(stage, NEEDLE_PATH)
needle_cyl.CreateRadiusAttr(0.015)
needle_cyl.CreateHeightAttr(0.5)
needle_prim = stage.GetPrimAtPath(NEEDLE_PATH)

# İğneyi X yönüne döndür (varsayılan eksen Z)
xf_needle = UsdGeom.XformCommonAPI(needle_prim)
xf_needle.SetTranslate(Gf.Vec3d(-0.8, 0.0, 0.4))
xf_needle.SetRotate(Gf.Vec3f(0.0, 90.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

# Rigid — kinematic, collision KAPALI (tissue içinden geçer)
rb_needle = UsdPhysics.RigidBodyAPI.Apply(needle_prim)
rb_needle.CreateKinematicEnabledAttr().Set(True)
# Collision API EKLENMİYOR → needle soft tissue'ya çarpmaz, içinden geçer

_add_material(NEEDLE_PATH, (0.2, 0.9, 0.3))  # yeşil

# ──────────────────────────────────────────────────────────────────────────────
# 10. COLLISION FILTERING
#     Arm ↔ Soft   : çarpışma VAR  (squash)
#     Needle ↔ Soft: çarpışma YOK  (iğne geçer) — collision API yok zaten
# ──────────────────────────────────────────────────────────────────────────────
# Arm ile needle arasındaki çarpışmayı filtrele (arm needle'ı bloklayamasın)
filter_arm_needle = UsdPhysics.FilteredPairsAPI.Apply(arm_prim)
filter_arm_needle.CreateFilteredPairsRel().AddTarget(NEEDLE_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 11. Physics Scene ayarları
# ──────────────────────────────────────────────────────────────────────────────
scene_path = "/physicsScene"
if not stage.GetPrimAtPath(scene_path).IsValid():
    scene_prim = UsdPhysics.Scene.Define(stage, scene_path)
else:
    scene_prim = stage.GetPrimAtPath(scene_path)

physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
physx_scene.CreateEnableCCDAttr().Set(True)          # Continuous Collision Detection
physx_scene.CreateSolverTypeAttr().Set("TGS")        # daha kararlı solver
physx_scene.CreateTimeStepsPerSecondAttr().Set(120)

# ──────────────────────────────────────────────────────────────────────────────
# 12. Simülasyonu başlat
# ──────────────────────────────────────────────────────────────────────────────
sim.reset()
print("[INFO] Simülasyon başladı. Çıkmak için pencereyi kapatın.")

# ──────────────────────────────────────────────────────────────────────────────
# 13. Simülasyon döngüsü
# ──────────────────────────────────────────────────────────────────────────────
SPEED       = 0.005   # adım başına metre
X_MIN       = -1.0
X_MAX       =  1.0
direction   =  1.0    # başlangıçta sağa (+X)

# İlk pozisyonlar
arm_x    = -0.8
needle_x = -0.8   # arm'ın ucundan biraz ileride başlasın

step = 0
while simulation_app.is_running():

    # --- Pozisyon güncelle ---
    arm_x    += SPEED * direction
    needle_x  = arm_x + 0.35   # iğne, kolun 35 cm önünde

    # Yön değiştir
    if arm_x >= X_MAX:
        direction = -1.0
    elif arm_x <= X_MIN:
        direction =  1.0

    # USD üzerinden kinematik hareket
    _set_pos(ARM_PATH,    [arm_x,    0.0, 0.4])
    _set_pos(NEEDLE_PATH, [needle_x, 0.0, 0.4])

    # Sim adımı
    sim.step()
    step += 1

    if step % 200 == 0:
        np_arm = _get_pos(ARM_PATH)
        np_ndl = _get_pos(NEEDLE_PATH)
        print(f"[Step {step:6d}]  Arm x={np_arm[0]:.3f}  Needle x={np_ndl[0]:.3f}  dir={direction:+.0f}")

simulation_app.close()
