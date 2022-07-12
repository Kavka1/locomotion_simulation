# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A formal locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import pos

import numpy as np


class FormalForwardTask(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self.current_base_pos   = np.zeros(3)
    self.current_rpy        = np.zeros[3]
    self.current_motor_ang  = None

    self.last_base_pos      = np.zeros(3)
    self.last_rpy           = np.zeros(3)
    self.last_motor_ang     = None


    self.episode_step = 0
    self._max_episode_len = 1000

  def __call__(self, env):
    self.episode_step += 1
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    
    self.last_base_pos      = env.robot.GetBasePosition()
    self.last_rpy           = env.robot.GetBaseRollPitchYaw()
    self.last_motor_ang     = env.robot.GetMotorAngles()

    self.current_base_pos   = self.last_base_pos
    self.current_rpy        = self.last_rpy
    self.current_motor_ang  = self.last_motor_ang

    self.episode_step = 0

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos      = self.current_base_pos
    self.last_rpy           = self.current_rpy
    self.last_motor_ang     = self.current_motor_ang

    self.current_base_pos       = env.robot.GetBasePosition()
    self.current_rpy            = env.robot.GetBaseRollPitchYaw()
    self.current_motor_ang      = env.robot.GetMotorAngles()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    rot_quat = env.robot.GetBaseOrientation()
    pos      = env.robot.GetBasePosition()
    rpy      = env.robot.GetBaseRollPitchYaw()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)

    # 0.28 < height < 0.6 
    # |roll| < pi/2 * 0.4 
    # |pitch| < pi/2 * 0.2
    notdone = pos[-1] > 0.18 and \
              pos[-1] < 0.6 and \
              abs(rpy[0]) < 0.628 and \
              abs(rpy[1]) < 0.314 and \
              rot_mat[-1] >= 0.85 and \
              self.episode_step <= self._max_episode_len
    return not notdone

  def reward(self, env):
    """Get the reward without side effects."""
    last_action = env.last_action
    del env

    v_x = self.current_base_pos[0] - self.last_base_pos[0]
    v_y = self.current_base_pos[1] - self.last_base_pos[1]
    v_z = self.current_base_pos[2] - self.last_base_pos[2]
    
    w_r = self.current_rpy[0] - self.last_rpy[0]
    w_y = self.current_rpy[1] - self.last_rpy[1]
    w_p = self.current_rpy[2] - self.last_rpy[2]

    motor_angle_speed = self.current_motor_ang - self.last_motor_ang

    # x-coordinate velocity
    forward_r                           = np.min(self.current_base_pos[0] - self.last_base_pos[0], 0.35)
    # lateral movement penalty
    lateral_movement_and_rotation_r     = - np.linalg.norm([v_y], 2) - np.linalg.norm([w_y], 2)
    # z-coordinate movement penalty
    z_axis_vel_r                        = - np.linalg.norm([v_z], 2)
    # joint angles velocity penalty
    motor_angle_speed_r                 = - np.linalg.norm(motor_angle_speed, 2)
    # roll pitch penalty
    roll_pitch_r                        = - np.linalg.norm(self.current_rpy[:2], 2)
    # action magnitude
    action_magnitude_r                  = - np.linalg.norm(last_action, 2)

    reward =  20 * forward_r \
            + 21 * lateral_movement_and_rotation_r \
            + 0.07 * action_magnitude_r \
            + 0.002 * motor_angle_speed_r \
            + 1.5 * roll_pitch_r \
            + 2.0 * z_axis_vel_r

    return reward
