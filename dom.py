#!/usr/bin/python
# -*- coding: utf-8 -*-


import scipy.sparse as sps


class DOM:

    def __init__(self, Nx=1, Ny=1, Nz=1, dx=1., dy=1., dz=1.):
        """
        DOM, Differential Operator Matrices. Generate big matrices for differential operators

        Parameters
        ----------
        Nx, Ny, Nz  :   integer
                        number of Yee cell sites in x, y, and z direction
        dx, dy, dz  :   float
                        Yee cell sizes

        """

        eye_Nx = sps.eye(Nx)
        eye_Ny = sps.eye(Ny)
        eye_Nz = sps.eye(Nz)

        if Nx > 1:
            # pxfr, pxbr, forward/backward partial_x for one row (a row is one line [0, Lx]in the physical solving domain)
            pxfr = sps.diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
            pxbr = sps.diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
        else:
            # if Nx = 1, means reduced dimension. Uniform in x, so paritial_x should be 0.
            pxfr = sps.diags([0.], 0, shape=(1, 1))
            pxbr = sps.diags([0.], 0, shape=(1, 1))
        self.pxf = sps.kron(eye_Nz, sps.kron(eye_Ny, pxfr)) / dx  # partial_x forward
        self.pxb = sps.kron(eye_Nz, sps.kron(eye_Ny, pxbr)) / dx

        if Ny > 1:
            self.pyf = sps.kron(eye_Nz,  (sps.kron(sps.diags([1, -1, 1], [-Ny + 1, 0, 1], shape=(Ny, Ny)), eye_Nx))  ) / dy
            self.pyb = sps.kron(eye_Nz,  (sps.kron(sps.diags([-1, 1, -1], [-1, 0, Ny - 1], shape=(Ny, Ny)), eye_Nx))  ) / dy
        else:
            self.pyf = sps.kron(eye_Nz,  (sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_Nx))  ) / dy
            self.pyb = sps.kron(eye_Nz,  (sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_Nx))  ) / dy

        eye_NyNx = sps.kron(eye_Ny, eye_Nx)
        if Nz > 1:
            self.pzf = sps.kron(sps.diags([1, -1, 1], [-Nz+1, 0, 1], shape=(Nz, Nz)), eye_NyNx) / dz
            self.pzb = sps.kron(sps.diags([-1, 1, -1], [-1, 0, Nz-1], shape=(Nz, Nz)), eye_NyNx) / dz
        else:
            self.pzf = sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_NyNx) / dz
            self.pzb = sps.kron(sps.diags([0.], 0, shape=(1, 1)), eye_NyNx) / dz


