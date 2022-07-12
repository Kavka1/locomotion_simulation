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
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import pos

import numpy as np


class SimpleForwardTask(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.episode_step = 0
    self._max_episode_len = 1000

  def __call__(self, env):
    self.episode_step += 1
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos
    self.episode_step = 0

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

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
    del env
    return self.current_base_pos[0] - self.last_base_pos[0]
