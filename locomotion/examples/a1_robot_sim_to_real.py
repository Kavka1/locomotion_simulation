"""Apply the same action to the simulated and real A1 robot.


As a basic debug tool, this script allows you to execute the same action
(which you choose from the pybullet GUI) on the simulation and real robot
simultaneouly. Make sure to put the real robbot on rack before testing.
"""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pybullet as p  # pytype: disable=import-error
import time
from tqdm import tqdm

from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import a1_robot
from locomotion.robots import robot_config


def main(_):
  print("WARNING: this code executes low-level controller on the robot.")
  print("Make sure the robot is hang on rack before proceeding.")
  input("Press enter to continue...")
  # Construct sim env and real robot
  env = env_builder.build_regular_env(
      robot_class=a1.A1,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      on_rack=True,
      enable_rendering=True,
      wrap_trajectory_generator=False)
  robot = a1_robot.A1(pybullet_client=None)
  while not robot.GetMotorAngles():
    print("Robot sensors not ready, sleep for 1s...")
    time.sleep(1)

  # Add debug sliders
  action_low, action_high = env.action_space.low, env.action_space.high
  dim_action = action_low.shape[0]
  action_selector_ids = []
  robot_motor_angles = robot.GetMotorAngles()

  for dim in range(dim_action):
    action_selector_id = p.addUserDebugParameter(
        paramName='dim{}'.format(dim),
        rangeMin=action_low[dim],
        rangeMax=action_high[dim],
        startValue=robot_motor_angles[dim])
    action_selector_ids.append(action_selector_id)

  # Visualize debug slider in sim
  for _ in range(10000):
    # Get user action input
    action = np.zeros(dim_action)
    for dim in range(dim_action):
      action[dim] = env.pybullet_client.readUserDebugParameter(
          action_selector_ids[dim])

    robot.ApplyAction(action, robot_config.MotorControlMode.POSITION)
    env.step(action)

  robot.Terminate()


if __name__ == '__main__':
  app.run(main)