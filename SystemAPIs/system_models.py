import numpy as np
import scipy as sp
import tools
from scipy.linalg import block_diag


class LinearSystem(object):
    def __init__(self, dimension, subsystems, init_mean, init_cov, n_mean, n_cov, dependent=True, stability=0.75):
        """
        This is a class for discrete time linear dynamical systems.

        Parameters
        ----------
        dimension : int
            Dimension of the dynamical system (state space, control space).
        subsystems : int
            Number of subsystems.
        dependent : boolean
            Activate week dependency between subsystems.
        stability: float
            Degree of stability for the average independent subsystem.
        """

        self.dim = dimension
        self.subsystems = subsystems
        self.dependent = dependent
        self.stability = stability
        self.state = None
        self.n_mean = n_mean
        self.n_cov = n_cov
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.q_system = None
        self.r_system = None

        if self.dim < 4:
            raise ValueError('Dimension must be greater than or equal to 4.')
        if self.dim < self.subsystems:
            raise ValueError('Dimension less than number of subsystems.')
        if 0 > self.stability or self.stability > 1:
            raise ValueError('Stability not in [0,1].')

        partitions = list(tools.partition(n=self.dim, k=self.subsystems, sing=2))
        partition = partitions[np.random.choice(len(partitions), 1)[0]]
        n_stable = np.int(np.around(self.subsystems*self.stability, decimals=0))
        stability = np.array([1]*n_stable + [0]*np.int(self.subsystems-n_stable))
        np.random.shuffle(stability)

        dyn_mat = []
        inp_mat = []
        for i in range(self.subsystems):
            a, b = tools.generate_random_system(partition[i], 1, marginally_stable=stability[i])
            dyn_mat.append(a)
            inp_mat.append(b)

        self.A = block_diag(*dyn_mat)
        self.subA = dyn_mat
        self.B = block_diag(*inp_mat)
        self.subB = inp_mat
        self.adjacency = np.zeros(self.dim)
        if dependent:
            self.adjacency = tools.random_graph(n=self.subsystems, l=-0.2)
            for i, j in np.argwhere(self.adjacency):
                pos1 = int(np.sum(partition[:-i]))
                pos2 = int(np.sum(partition[:-j]))
                self.A[pos1:pos1+partition[i], pos2:pos2+partition[j]] \
                    += 0.25*np.random.rand(partition[i], partition[j])

    def state_update(self, control):
        """
        Parameters
        ----------
        control : float
            system control input
        Returns
        -------
        Noisy state update.
        """
        if np.shape(control)[0] != self.subsystems:
            raise ValueError('Control dimension does not fit input dimension of the system.')
        self.state = np.einsum('ij,j->i', self.A, self.state) + np.einsum('ij,j->i', self.B, control) \
                    + np.random.multivariate_normal(self.n_mean, self.n_cov)
        return self.state

    def reset_state(self):
        self.state = np.random.multivariate_normal(self.init_mean, self.init_cov)

    def set_state(self, new_state):
        if np.shape(new_state)[0] != self.dim:
            raise ValueError('New state does not fit defined system dimension.')
        self.state = new_state

    def set_system(self, a, b):
        self.A = a
        self.B = b


class BaseNetwork_DQN(object):
    def __init__(self, classes, links, quantities, quality):
        self.classes = classes
        self.quantities = quantities
        self.quality = quality
        self.links = links
        self.cases = None

        for k in range(self.classes):
            if self.cases is None:
                self.cases = sp.special.binom(self.links, self.quantities[k])
            else:
                self.cases *= sp.special.binom(self.links-np.sum(self.quantities[:k]), self.quantities[k])
        self.cases = np.int(self.cases)

    def net_output(self, action):
        output = np.zeros((self.links,))

        temp = self.links-self.quantities[0]

        if np.random.sample() <= self.quality[0]:
            output[np.int(np.ceil((action+1)/temp))-1] = 1

        if np.mod(action, temp) >= np.int(np.ceil((action+1)/temp))-1:
            if np.random.sample() <= self.quality[1]:
                output[np.mod(action, temp) + 1] = 1
        else:
            if np.random.sample() <= self.quality[1]:
                output[np.mod(action, temp)] = 1
        return output


class BaseNetwork(object):
    def __init__(self, quality, links):
        self.classes = len(quality)
        self.quality = quality
        self.links = links

    def output(self, schedule):
        # Evaluate schedule
        result = np.zeros((self.links,))
        action = np.zeros((self.links * self.classes,))
        for i in range(self.links):
            resource = np.where(schedule[:, i])[0][0]
            if resource != self.classes:
                action[2 * i + resource] += 1
                if np.random.sample() <= self.quality[resource]:
                    result[i] = 1
        return result, action
