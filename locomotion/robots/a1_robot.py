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

from io import BytesIO
import os
import sys
import inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import ctypes
import math
import os
import re
import numpy as np
import pybullet as pyb  # pytype: disable=import-error
import lcm
import threading

from locomotion.robots import laikago_pose_utils
from locomotion.robots import laikago_constants
from locomotion.robots import laikago_motor
from locomotion.robots import minitaur
from locomotion.robots import robot_config
from locomotion.robots.unitree_legged_sdk import comm
from locomotion.envs import locomotion_gym_config

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
_DEFAULT_HIP_POSITIONS = (
    (0.21, -0.1157, 0),
    (0.21, 0.1157, 0),
    (-0.21, -0.1157, 0),
    (-0.21, 0.1157, 0),
)

ABDUCTION_P_GAIN = 5.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 5.0
HIP_D_GAIN = 1.0
KNEE_P_GAIN = 5.0
KNEE_D_GAIN = 1.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([
    laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = "a1/a1.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3


class A1(minitaur.Minitaur):
  """A simulation for the Laikago robot."""

  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name="FR_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="FR_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="FR_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="FL_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="FL_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="FL_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="RR_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="RR_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="RR_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="RL_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="RL_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="RL_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
  ]

  def __init__(self,
               pybullet_client,
               urdf_filename=URDF_FILENAME,
               enable_clip_motor_commands=True,
               time_step=0.001,
               action_repeat=33,
               sensors=None,
               control_latency=0.002,
               on_rack=False,
               enable_action_interpolation=True,
               enable_action_filter=True,
               motor_control_mode=None,
               reset_time=-1,
               allow_knee_contact=False,
               command_channel_name='LCM_Low_Cmd',
               state_channel_name='LCM_Low_State'):
    # Initialize pd gain matrix
    self.motor_kps = np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * 4)
    self.motor_kds = np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * 4)

    # Initiate LCM channel for robot state and actions
    self.lc = lcm.LCM()
    self._command_channel_name = command_channel_name
    self._state_channel_name = state_channel_name
    self._state_channel = self.lc.subscribe(state_channel_name,
                                            self.ReceiveObservation)

    # Robot state variables
    self._motor_angles = None
    self._base_orientation = None
    self._raw_state = None

    self._is_alive = True
    self._SendZeroAction()
    self.subscribe_thread = threading.Thread(target=self._LCMSubscribeLoop,
                                             args=())
    self.subscribe_thread.start()

  def _LCMSubscribeLoop(self):
    while self._is_alive:
      self.lc.handle_timeout(100)

  def _SendZeroAction(self):
    """Sends zero action to the robot.

    This function is required to get initial sensor reading from the robot.
    Otherwise, the lcm server will return all-zero.
    """
    command = comm.LowCmd()
    command.levelFlag = 0xff
    for motor_id in range(NUM_MOTORS):
      command.motorCmd[motor_id].mode = 0x00
      command.motorCmd[motor_id].q = 0
      command.motorCmd[motor_id].Kp = 0
      command.motorCmd[motor_id].dq = 0
      command.motorCmd[motor_id].Kd = 0
      command.motorCmd[motor_id].tau = 0
    self.lc.publish(self._command_channel_name, command)

  def ReceiveObservation(self, channel, data):
    """Receive the observation from sensors.

    This function is called once per step. The observations are only updated
    when this function is called.
    """
    stream = BytesIO(data)
    state = comm.LowState()
    stream.readinto(state)  # pytype: disable=wrong-arg-types
    self._base_position = (0, 0, 0)
    self._base_orientation = list(state.imu.quaternion)
    self._motor_angles = [motor.q for motor in state.motorState[:12]]
    self._motor_velocities = [motor.dq for motor in state.motorState[:12]]
    self._raw_state = state

  def GetMotorAngles(self):
    return self._motor_angles

  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).
      motor_control_mode: A MotorControlMode enum.
    """
    command = comm.LowCmd()
    command.levelFlag = 0xff

    if motor_control_mode == robot_config.MotorControlMode.POSITION:
      for motor_id in range(NUM_MOTORS):
        command.motorCmd[motor_id].mode = 0x0A
        command.motorCmd[motor_id].q = motor_commands[motor_id]
        command.motorCmd[motor_id].Kp = self.motor_kps[motor_id]
        command.motorCmd[motor_id].dq = 0
        command.motorCmd[motor_id].Kd = self.motor_kds[motor_id]
        command.motorCmd[motor_id].tau = 0

      # Gravity compensation
      command.motorCmd[0].tau = -0.65
      command.motorCmd[3].tau = 0.65
      command.motorCmd[6].tau = -0.65
      command.motorCmd[9].tau = 0.65
    elif motor_control_mode == robot_config.MotorControlMode.TORQUE:
      for motor_id in range(NUM_MOTORS):
        command.motorCmd[motor_id].mode = 0x0A
        command.motorCmd[motor_id].q = 0
        command.motorCmd[motor_id].Kp = 0
        command.motorCmd[motor_id].dq = 0
        command.motorCmd[motor_id].Kd = 0
        command.motorCmd[motor_id].tau = motor_commands[motor_id]
    elif motor_control_mode == robot_config.MotorControlMode.HYBRID:
      raise NotImplementedError()
    else:
      raise ValueError('Unknown motor control mode for A1 robot.')

    self.lc.publish(self._command_channel_name, command)

  def Terminate(self):
    self._is_alive = False