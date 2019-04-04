import numpy as np
import numpy.matlib
import cmath
from scipy.linalg import block_diag, orth
from scipy.optimize import minimize
import scipy as sp
import networkx as nx
np.random.seed(160398798)

def one_step_controller(x, schedule, gamma, alpha, q, r, resource_quantities, system, state_size, value_model):
    u = 0
    action_size = value_model.layers[-1].output_shape[1]
    resource_classes = len(resource_quantities)
    initial_representation = np.zeros((resource_classes + 1, action_size))
    initial_representation[resource_classes, :] += 1


    input_list = []
    cost_list = []
    for i in range(action_size):
        def f(control_input):
            b_active = system.B @ np.diag(schedule[0, :]*gamma[0]) + system.B @ np.diag(schedule[1, :]*gamma[1])
            next_state = system.A @ x + b_active @ control_input
            next_agent_state = np.hstack((next_state, resource_quantities, initial_representation.flatten()))
            applied_state = np.reshape(next_agent_state, [1, state_size])

            applied_control = np.diag(schedule[0, :]*gamma[0]) @ control_input \
                              + np.diag(schedule[1, :]*gamma[1]) @ control_input
            system_loss = x.transpose() @ q @ x + applied_control.transpose() @ r @ applied_control

            return system_loss - alpha*value_model.predict(applied_state)[0][i]

        controllers = []
        for a, b in zip(system.subA, system.subB):
            q_sub = np.eye(np.shape(a)[0])
            r_sub = 1
            dare = sp.linalg.solve_discrete_are(a, b, q_sub, r_sub)
            controllers.append(-np.linalg.inv(b.transpose() @ dare @ b + r_sub) @ b.transpose() @ dare @ a)
        control_matrix = block_diag(*controllers)
        u0 = (np.diag(schedule[0, :]) + np.diag(schedule[1, :]))@control_matrix @ x
        u = minimize(f, u0, method='powell', options = {'xtol': 1e-5, 'disp': False})

        input_list.append(u.x)
        cost_list.append(f(u.x))
    return input_list, cost_list


def solve_mare(a, b, q, r, delta):
    p = sp.linalg.solve_discrete_are(a, b, q, r)
    p_new = a.transpose()@p@a + q - delta*a.transpose()@p@b@np.linalg.inv(b.transpose()@p@b+r)@b.transpose()@p@a
    while np.abs(np.linalg.norm(p, 'fro') - np.linalg.norm(p_new, 'fro')) > 0.001:
        p = p_new
        p_new = a.transpose()@p@a + q - delta*a.transpose()@p@b@np.linalg.inv(b.transpose()@p@b+r)@b.transpose()@p@a
    return p_new

def generate_random_system(n, m, marginally_stable=True):
    """
    Generate a random discrete time system.

    Parameters
    ----------
    n : int
        Order of the system model.
    m : int
        Number of inputs.
    marginally_stable : boolean
        Boolean variable that determines whether the system should be marginally stable.
    Returns
    -------
    a : nxn system dynamic matrix
    b : nxm input dynamic matrix
    """

    if marginally_stable:
        n_integrator = np.int((np.random.random_sample() < 0.1) + np.sum(np.random.random_sample((n-1,)) < 0.01))
        n_double = np.int(np.floor(np.sum(np.random.random_sample((n-n_integrator,)) < 0.05)/2))
        n_complex = np.int(np.floor(np.sum(np.random.random_sample((n - n_integrator - 2*n_double,)) < 0.5) / 2))
        n_real = n - n_integrator - 2*n_double - 2*n_complex

        rep = 2*np.random.random_sample((n_double,)) - 1
        real = 2*np.random.random_sample((n_real,)) - 1
        poles = []
        if n_complex != 0:
            for i in range(n_complex):
                mag = np.random.random_sample()
                comp = mag * cmath.exp(complex(0, np.pi * np.random.random_sample()))
                re = comp.real
                im = comp.imag
                poles.append(np.array([[re, im], [-im, re]]))
        if n_integrator != 0:
            poles.append(np.eye(n_integrator))
        if n_double != 0:
            for pole in rep:
                poles.append(np.eye(2)*pole)
        if n_real != 0:
            poles.append(np.diag(real))

        t = orth(np.random.random_sample((n, n)))
        a = np.linalg.lstsq(t, block_diag(*poles), rcond=None)[0] @ t

        b = np.random.random_sample((n, m))
        mask = np.random.random_sample((n, m)) < 0.75
        zero_col = np.all(np.logical_not(mask), axis=0, keepdims=True)
        b = b*(mask+np.matlib.repmat(zero_col, n, 1))
    else:
        n_unstable = np.int(1 + np.sum(np.random.random_sample((n - 1,)) < 0.1))
        n_double = np.int(np.floor(np.sum(np.random.random_sample((n - n_unstable,)) < 0.05) / 2))
        n_complex = np.int(np.floor(np.sum(np.random.random_sample((n - n_unstable - 2 * n_double,)) < 0.5) / 2))
        n_real = n - n_unstable - 2 * n_double - 2 * n_complex

        unstable = np.random.uniform(1.01, 1.5, n_unstable)
        for k in range(n_unstable):
            if np.random.random_sample() < 0.5:
                unstable[k] *= -1

        rep = 2*np.random.random_sample((n_double,)) - 1
        real = 2*np.random.random_sample((n_real,)) - 1
        poles = []
        if n_complex != 0:
            for i in range(n_complex):
                mag = np.random.random_sample()
                comp = mag * cmath.exp(complex(0, np.pi * np.random.random_sample()))
                re = comp.real
                im = comp.imag
                poles.append(np.array([[re, im], [-im, re]]))
        if n_unstable != 0:
            poles.append(np.diag(unstable))
        if n_double != 0:
            for pole in rep:
                poles.append(np.eye(2) * pole)
        if n_real != 0:
            poles.append(np.diag(real))

        t = orth(np.random.random_sample((n, n)))
        a = np.linalg.lstsq(t, block_diag(*poles), rcond=None)[0] @ t

        b = np.random.random_sample((n, m))
        mask = np.random.random_sample((n, m)) < 0.75
        zero_col = np.all(np.logical_not(mask), axis=0, keepdims=True)
        b = b * (mask + np.matlib.repmat(zero_col, n, 1))

    return a, b


def random_graph(n, l):
    r = 0.5
    g = nx.random_geometric_graph(n, r)
    l2 = nx.algebraic_connectivity(g)
    while np.abs(l2-l) > 0.2:
        if l2 > l:
            r = r*0.9
        else:
            r = r*1.1
        g = nx.random_geometric_graph(n, r)
        l2 = nx.algebraic_connectivity(g)
    return nx.adjacency_matrix(g).todense()


def partition(n, k, sing):
    # Credits to 'Snakes and Coffee' from stackoverflow.com
    # n is the integer to partition, k is the length of partitions, l is the min partition element size
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= sing:
            yield (n,)
        raise StopIteration
    for i in range(sing, n+1):
        for result in partition(n-i, k-1, i):
            yield (i,)+result
