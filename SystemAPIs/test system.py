import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import system_models
from scipy.linalg import block_diag


def main():

    # Set up 3 single input systems with dimension 2x2 without dependence.
    # ~1/4 of the systems are unstable.
    sys_dim = 6
    sys_sub = 3
    x0_mean = np.zeros((sys_dim,))
    x0_cov = np.eye(sys_dim)*6
    system = system_models.LinearSystem(dimension=sys_dim, subsystems=sys_sub, dependent=False, stability=0.75)

    # # Set dynamics manually
    # dyn_mat = [np.array([[0.8, 0.5], [0, 0.3]]), np.array([[1.0, 0.7], [0, 0.4]]),
    #            np.array([[1.3, 0.8], [0, 0.5]])]
    # inp_mat = [np.array([[0], [1]]), np.array([[0], [1]]),
    #            np.array([[0], [1]])]
    # system.A = block_diag(*dyn_mat)
    # system.subA = dyn_mat
    # system.B = block_diag(*inp_mat)
    # system.subB = inp_mat

    # Set up LQR controllers
    controllers = []
    for a, b in zip(system.subA, system.subB):
        q_subsystem = np.eye(np.shape(a)[0])
        r_subsystem = 0.1
        dare = sp.linalg.solve_discrete_are(a, b, q_subsystem, r_subsystem)
        controllers.append(-np.linalg.inv(b.transpose()@dare@b +
                                          r_subsystem)@b.transpose()@dare@a)
    q_system = np.eye(np.shape(system.A)[0])
    r_system = np.eye(np.shape(system.B)[1])
    k_system = block_diag(*controllers)

    horizon = 100
    cost_sequence = []
    system_state = np.random.multivariate_normal(x0_mean, x0_cov)
    system.set_state(system_state)

    for k in range(horizon):
        control = k_system @ system_state
        new_system_state = system.state_update(control)
        cost_sequence.append(system_state.transpose()@q_system@system_state
                             + control.transpose()@r_system@control)
        system_state = new_system_state
    plt.xlabel(r'$ k $')
    plt.ylabel(r'$ J_k $')
    plt.plot(cost_sequence)
    plt.show()

    for a in system.subA:
        print(np.linalg.eigvals(a))


if __name__ == "__main__":
    main()
