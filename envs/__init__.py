from gym.envs import classic_control, box2d, mujoco
# from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
# from pybullet_envs.deep_mimic.gym_env import HumanoidDeepMimicBackflipBulletEnv
# from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
# from rex_gym.envs.gym.galloping_env import RexReactiveEnv
from .continuous_acrobot import ContinuousAcrobotEnv
from .continuous_pendubot import ContinuousPendubotEnv
from .gym_monitor_nodone import Monitor

env_list = {
    'HalfCheetahEnv' : mujoco.HalfCheetahEnv,
    'HopperEnv' : mujoco.HopperEnv,
    # 'HalfCheetahBulletEnv' : gym_locomotion_envs.HalfCheetahBulletEnv,
    # 'HopperBulletEnv' : gym_locomotion_envs.HopperBulletEnv,
    # 'HopperBulletEnv' : HumanoidDeepMimicBackflipBulletEnv,
    # 'ReacherBulletEnv' : gym_manipulator_envs.ReacherBulletEnv,
    # 'Walker2DEnv' : gym_locomotion_envs.Walker2DBulletEnv
    # 'RexEnv' : RexReactiveEnv
    # 'ReacherEnv' : mujoco.ReacherEnv,
    # 'PusherEnv' : mujoco.PusherEnv,
    # 'ThrowerEnv' : mujoco.ThrowerEnv,
    # 'StrikerEnv' : mujoco.StrikerEnv,
    'PendulumEnv' : classic_control.PendulumEnv,
    'AcrobotEnv' : ContinuousAcrobotEnv,
    'PendubotEnv' : ContinuousPendubotEnv,
}

def getlist():
    out_str = ''
    for env_name in env_list.keys():
        out_str += env_name + '\n'
    return out_str
