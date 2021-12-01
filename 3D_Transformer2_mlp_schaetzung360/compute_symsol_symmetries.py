"""Loads the SYMSOL dataset."""


import os

import numpy as np
from scipy.spatial.transform import Rotation
import scipy.misc

import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
def compute_symsol_symmetries_tless(num_steps_around_continuous=360):

  # First class
  one_syms_qua = []
  for sym_val in np.linspace(0, 2*np.pi, num_steps_around_continuous):
    sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                      sym_val]))
    s = sym_rot.as_matrix()
    rot = R.from_matrix(s)
    qua = rot.as_quat()
    # qua = sym_rot.as_quat()
    qua = np.array(qua)
    one_syms_qua.append(qua)
    # cyl_syms.append(sym_rot)
  one_syms_qua = np.stack(one_syms_qua, 0)

  # Second class
  two_syms_qua = []
  for sym_val in np.linspace(0, 2 * np.pi, num_steps_around_continuous):
      sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                       sym_val]))
      s = sym_rot.as_matrix()
      rot = R.from_matrix(s)
      qua = rot.as_quat()
      # qua = sym_rot.as_quat()
      qua = np.array(qua)
      two_syms_qua.append(qua)
      # cyl_syms.append(sym_rot)
  two_syms_qua = np.stack(two_syms_qua, 0)

  # third class
  three_syms_qua = []
  for sym_val in np.linspace(0, 2 * np.pi, num_steps_around_continuous):
      sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                       sym_val]))
      s = sym_rot.as_matrix()
      rot = R.from_matrix(s)
      qua = rot.as_quat()
      # qua = sym_rot.as_quat()
      qua = np.array(qua)
      three_syms_qua.append(qua)
      # cyl_syms.append(sym_rot)
  three_syms_qua = np.stack(three_syms_qua, 0)

  # fourth class
  four_syms_qua = []
  for sym_val in np.linspace(0, 2 * np.pi, num_steps_around_continuous):
      sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                       sym_val]))
      s = sym_rot.as_matrix()
      rot = R.from_matrix(s)
      qua = rot.as_quat()
      # qua = sym_rot.as_quat()
      qua = np.array(qua)
      four_syms_qua.append(qua)
      # cyl_syms.append(sym_rot)
  four_syms_qua = np.stack(four_syms_qua, 0)

  return one_syms_qua, two_syms_qua, three_syms_qua, four_syms_qua

def compute_symsol_symmetries_tless_08(num_steps_around_continuous=360):

  # First class
  one_syms_qua = []
  for sym_val in np.linspace(0, 2*np.pi, num_steps_around_continuous):
    sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                      sym_val]))
    s = sym_rot.as_matrix()
    rot = R.from_matrix(s)
    qua = rot.as_quat()
    # qua = sym_rot.as_quat()
    qua = np.array(qua)
    one_syms_qua.append(qua)
    # cyl_syms.append(sym_rot)
  one_syms_qua = np.stack(one_syms_qua, 0)
  return one_syms_qua

def compute_symsol_symmetries_tless_5class(num_steps_around_continuous=360):

  # First class
  one_syms_qua = []
  # num_steps_around_continuous = 6
  for sym_val in np.linspace(0, 2*np.pi, num_steps_around_continuous):
    sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                      sym_val]))
    s = sym_rot.as_matrix()
    rot = R.from_matrix(s)
    qua = rot.as_quat()
    # qua = sym_rot.as_quat()
    qua = np.array(qua)
    one_syms_qua.append(qua)
    # cyl_syms.append(sym_rot)
  one_syms_qua = np.stack(one_syms_qua, 0)

  # Second class
  two_syms_qua = []
  # num_steps_around_continuous = 6
  for sym_val in np.linspace(0, 2 * np.pi, num_steps_around_continuous):
      sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                       sym_val]))
      s = sym_rot.as_matrix()
      rot = R.from_matrix(s)
      qua = rot.as_quat()
      # qua = sym_rot.as_quat()
      qua = np.array(qua)
      two_syms_qua.append(qua)
      # cyl_syms.append(sym_rot)
  two_syms_qua = np.stack(two_syms_qua, 0)

  # third class
  three_syms_qua = []
  # num_steps_around_continuous = 4
  for sym_val in np.linspace(0, 2 * np.pi, num_steps_around_continuous):
      sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                       sym_val]))
      s = sym_rot.as_matrix()
      rot = R.from_matrix(s)
      qua = rot.as_quat()
      # qua = sym_rot.as_quat()
      qua = np.array(qua)
      three_syms_qua.append(qua)
      # cyl_syms.append(sym_rot)
  three_syms_qua = np.stack(three_syms_qua, 0)

  # fourth class
  four_syms_qua = []
  # num_steps_around_continuous = 2
  for sym_val in np.linspace(0, 2 * np.pi, num_steps_around_continuous):
      sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                       sym_val]))
      s = sym_rot.as_matrix()
      rot = R.from_matrix(s)
      qua = rot.as_quat()
      # qua = sym_rot.as_quat()
      qua = np.array(qua)
      four_syms_qua.append(qua)
      # cyl_syms.append(sym_rot)
  four_syms_qua = np.stack(four_syms_qua, 0)

  # five class
  five_syms_qua = []
  # num_steps_around_continuous = 360
  for sym_val in np.linspace(0, 2 * np.pi, num_steps_around_continuous):
      sym_rot = Rotation.from_euler('xyz', np.float32([0, 0,
                                                       sym_val]))
      s = sym_rot.as_matrix()
      rot = R.from_matrix(s)
      qua = rot.as_quat()
      # qua = sym_rot.as_quat()
      qua = np.array(qua)
      five_syms_qua.append(qua)
  five_syms_qua = np.stack(five_syms_qua, 0)

  return one_syms_qua, two_syms_qua, three_syms_qua, four_syms_qua, five_syms_qua

def main():
    # one_syms_qua, two_syms_qua, three_syms_qua, four_syms_qua = compute_symsol_symmetries_tless()
    # one_syms_qua = compute_symsol_symmetries_tless_08()
    one_syms_qua, two_syms_qua, three_syms_qua, four_syms_qua, five_syms_qua = compute_symsol_symmetries_tless_5class()
    path = '/home/lei/Desktop/project-isas/code/Transformer_3D_Object_Tless/3D_Transformer2_mlp_schaetzung360/'
    name = 'symsol_symmetries_tless360class.npz'
    save_path = os.path.join(path, name)
    # np.savez(save_path, one_syms_qua=one_syms_qua, second_syms_qua=two_syms_qua,third_syms_qua=three_syms_qua,fourth_syms_qua=four_syms_qua)
    np.savez(save_path, one_syms_qua=one_syms_qua, second_syms_qua=two_syms_qua,third_syms_qua=three_syms_qua,fourth_syms_qua=four_syms_qua, fifth_syms_qua=five_syms_qua)
    return

if __name__ == "__main__":
    main()