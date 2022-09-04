# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from solidspy.preprocesor import rect_grid
import solidspy.postprocesor as pos
import solidspy.assemutil as ass
import solidspy.solutil as sol

plt.style.use("ggplot")
plt.rcParams["grid.linestyle"] = "dashed"

#%% Auxiliar functions
def fem_sol(nodes, elements, mats, loads):
    """
    Compute the FEM solution for a given problem.

    Parameters
    ----------
    nodes : array
        Array with nodes
    elements : array
        Array with element information.
    mats : array
        Array with material elements. We need a material profile
        for each element for the optimization process.
    loads : array
        Array with loads.

    Returns
    -------
    disp_comp : array
        Displacement for each node.
    """
    DME, IBC , neq = ass.DME(nodes, elements)
    stiff = ass.assembler(elements, mats, nodes, neq, DME)
    load_vec = ass.loadasem(loads, IBC, neq)
    disp = sol.static_sol(stiff, load_vec)
    disp_comp = pos.complete_disp(IBC, nodes, disp)
    return disp_comp


def weight(areas, nodes, elements):
    """Compute the weigth of the truss"""
    ini = elements[:, 3]
    end = elements[:, 4]
    lengths = np.linalg.norm(nodes[end, 1:3] - nodes[ini, 1:3], axis=1)
    return np.dot(areas, lengths)
    # return np.sum(areas * lengths)


def compliance(areas, nodes, elements, loads, mats):
    """Compute the compliance of the truss"""
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    forces = np.zeros_like(disp)
    forces[loads[:, 0].astype(int), 0] = loads[:, 1]
    forces[loads[:, 0].astype(int), 1] = loads[:, 2]
    return np.sum(forces*disp)


def stress_cons(areas, nodes, elements, mats, loads, stresses, comp):
    """Return the stress constraints"""
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    cons = np.asarray(stresses) -\
        pos.stress_truss(nodes, elements, mats2, disp)
    return cons[comp]


def stress_bnd(areas, nodes, elements, mats, loads, stresses):
    """Bounds on the stress for each member"""
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    return np.asarray(stresses) -\
        pos.stress_truss(nodes, elements, mats2, disp)


def grid_truss(length, height, nx, ny):
    """
    Generate a grid made of vertical, horizontal and diagonal
    members
    """
    nels = (nx - 1)*ny +  (ny - 1)*nx + 2*(nx - 1)*(ny - 1)
    x, y, _ = rect_grid(length, height, nx - 1, ny - 1)
    nodes = np.zeros((nx*ny, 5))
    nodes[:, 0] = range(nx*ny)
    nodes[:, 1] = x
    nodes[:, 2] = y
    elements = np.zeros((nels, 5), dtype=np.int)
    elements[:, 0] = range(nels)
    elements[:, 1] = 6
    elements[:, 2] = range(nels)
    hor_bars =  [[cont, cont + 1] for cont in range(nx*ny - 1)
                 if (cont + 1)%nx != 0]
    vert_bars =  [[cont, cont + nx] for cont in range(nx*(ny - 1))]
    diag1_bars =  [[cont, cont + nx + 1] for cont in range(nx*(ny - 1))
                   if  (cont + 1)%nx != 0]
    diag2_bars =  [[cont, cont + nx - 1] for cont in range(nx*(ny - 1))
                   if  cont%nx != 0]
    bars = hor_bars + vert_bars + diag1_bars + diag2_bars
    elements[:len(bars), 3:] = bars
    return nodes, elements


def plot_truss(nodes, elements, mats, loads, tol=1e-5):
    """
    Plot a truss and encodes the stresses in a colormap
    """
    disp = fem_sol(nodes, elements, mats, loads)
    stresses = pos.stress_truss(nodes, elements, mats, disp)
    max_stress = max(-stresses.min(), stresses.max())
    scaled_stress = 0.5*(stresses + max_stress)/max_stress
    min_area = mats[:, 1].min()
    max_area = mats[:, 1].max()
    areas = mats[:, 1].copy()
    max_val = 4
    min_val = 0.5
    if max_area - min_area > 1e-6:
        widths = (max_val - min_val)*(areas - min_area)/(max_area - min_area)\
            + min_val
    else:
        widths = 3*np.ones_like(areas)
    for el in elements:
        if areas[el[2]] > tol:
            ini, end = el[3:]
            color = plt.cm.seismic(scaled_stress[el[0]])
            plt.plot([nodes[ini, 1], nodes[end, 1]],
                     [nodes[ini, 2], nodes[end, 2]],
                     color=color, lw=widths[el[2]])
    plt.axis("image")


if __name__ == "__main__":
#%% Example from An Introduction to Structural Optimization

    ## 2.1 Weight min two-bar truss w/ stress contraints
    angle = np.pi/6
    load = 1.0
    nodes = np.array([
        [0 , 0.0,  0.0, -1, -1],
        [1 , 1.0,  0.0, 0, 0],
        [2 ,  1.0,  1.0,  -1,  -1]])
    elements = np.array([
        [0, 6, 0, 0, 1],
        [1, 6, 1, 1, 2]])
    mats = np.array([
        [1.0, 0.1],
        [1.0, 0.1]])
    loads = np.array([[1, load*np.cos(angle), -load*np.sin(angle)]])
    areas = mats[:, 1].copy()

    #%% Optimization
    nels = 2
    tot_w = 0.3
    bnds = [(1e-3, 0.1) for cont in range(nels)]
    weight_fun = lambda areas, nodes, elements, tot_w:\
            tot_w - weight(areas, nodes, elements)
    weight_cons = [{'type': 'ineq', 'fun':weight_fun,
        'args': (nodes, elements, tot_w)}]
    cons = weight_cons
    res = minimize(compliance, x0=areas, args=(nodes, elements, loads, mats),
                   bounds=bnds, constraints=cons, method="SLSQP",
                   tol=1e-6, options={"maxiter": 500, "disp":True})

    #%% Results
    mats2 = mats.copy()
    mats2[:, 1] = areas[:]
    disp = fem_sol(nodes, elements, mats, loads)
    print("Original design: {}".format(areas))
    print("Weigth: {}".format(weight(areas, nodes, elements)))
    print("Stresses: {}".format(pos.stress_truss(nodes, elements, mats, disp)))
    disp = fem_sol(nodes, elements, np.column_stack((mats[:,0], res.x)), loads)
    print("Optimized design: ", res.x)
    print("Weigth: {}".format(weight(res.x, nodes, elements)))
    print("Stresses: {}".format(pos.stress_truss(nodes, elements, mats2, disp)))

