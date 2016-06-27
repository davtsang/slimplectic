import slimplectic_GGL as ggl, numpy as np


###########################################
# Interface class for the nonconservative #
#         variational integrator          #
###########################################

class GalerkinGaussLobatto(object):

  def __init__(self, t, q_list, v_list, mod_list = False):
    # Validate inputs
    assert type(t) is str, "String input required."
    assert len(q_list) == len(v_list), "Unequal number of coordinates and velocities."
    self._num_dof = len(q_list)
    assert type(q_list) is list, "List input required."
    assert type(v_list) is list, "List input required."

    for ii in range(len(q_list)):
      assert type(q_list[ii]) is str, "String input required."
      assert type(v_list[ii]) is str, "String input required."

    # Make sympy variables
    self.t = ggl.Symbol(t)
    self.q = [ggl.Symbol(qq) for qq in q_list]
    self.v = [ggl.Symbol(vv) for vv in v_list]

    # Double the sympy variables
    self.qp, self.qm = ggl.q_Generate_pm(self.q)
    self.vp, self.vm = ggl.q_Generate_pm(self.v)
    
    #keep track of which variables are periodic and need to be modded
    if mod_list:
        self.modlist = mod_list
    if not mod_list:
        self.modlist = []
        for q in q_list:
            self.modlist.append(False)
    

  def keys(self):
    return self.__dict__.keys()

  def discretize(self, L, K, order, method='explicit', verbose=False):
    """Generate the nonconservative variational integrator maps"""
    self._qi_soln_map, self._q_np1_map, self._pi_np1_map, self._qdot_n_map = ggl.Gen_GGL_NC_VI_Map(self.t, \
                              self.q, self.qp, self.qm, \
                              self.v, self.vp, self.vm, \
                              L, K, order, method=method, verbose=verbose)

  def integrate(self, q0_list, pi0_list, t):
    """Numerical integration from given initial data"""

    # Check if total Lagrangian is discretized already
    if not hasattr(self, '_qi_soln_map'):
        raise AttributeError("Run `discretize` to discretize the total Lagrangian.")

    # Validate input
    assert type(q0_list) in [list, np.ndarray], "List or numpy array input required."
    assert type(pi0_list) in [list, np.ndarray], "List or numpy array input required."

    # Allocate memory for solutions
    t_len = t.size
    q_list_soln = np.zeros((t_len, self._num_dof))
    pi_list_soln = np.zeros((t_len, self._num_dof))

    # Set initial data
    q_list_soln[0,:] = q0_list
    # mod the value of any periodic variables that have mod value specified
    # This prevents NC evolution error due to increasing roundoff error as 
    # any cyclic periodic variables become large. 
    for jj,mod in enumerate(self.modlist):
        if mod:
            q_list_soln[0,jj] = q_list_soln[0,jj]%mod
    pi_list_soln[0,:] = pi0_list


    # Perform the integration at fixed time steps
    dt = t[1]-t[0]
    for ii in range(1, t_len):
        args = [q_list_soln[ii-1], pi_list_soln[ii-1], t[ii-1], dt]
        qi_sol = self._qi_soln_map(*args)
        q_list_soln[ii] = self._q_np1_map(qi_sol, *args)
        # mod the value of any periodic variables that have mod value specified
        # This prevents NC evolution error due to increasing roundoff error as 
        # any cyclic periodic variables become large. 
        for jj,mod in enumerate(self.modlist):
            if mod:
                q_list_soln[ii,jj] = q_list_soln[ii,jj]%mod
        pi_list_soln[ii] = self._pi_np1_map(qi_sol, *args)

    # Return the numerical solutions
    return q_list_soln.T, pi_list_soln.T

  def __call__(self, qi_list, vi_list, t):
    return self.integrate(qi_list, vi_list, t)


##########################################
# Class for 4th order Runge-Kutta method #
##########################################

class RungeKutta4(object):

  def __init__(self):
    self._b = np.array([1./6., 1./3., 1./3., 1./6.])
    self._k = [[],[],[],[]]

  def _iter(self, tn, yn_list, f, h):
    self._k[0] = f(tn, yn_list)
    self._k[1] = f(tn+h/2., yn_list+h/2.*self._k[0])
    self._k[2] = f(tn+h/2., yn_list+h/2.*self._k[1])
    self._k[3] = f(tn+h, yn_list+h*self._k[2])
    return yn_list + h*np.sum(self._b[ii]*self._k[ii] for ii in range(4))

  def integrate(self, q0_list, v0_list, t, f):
    y0_list = np.hstack([q0_list, v0_list])
    h = t[1]-t[0]
    ans = np.zeros((t.size, len(y0_list)))
    ans[0,:] = y0_list
    for ii, tt in enumerate(t[:-1]):
      ans[ii+1,:] = self._iter(tt, ans[ii], f, h)
    out = ans.T
    return out[:len(q0_list)], out[len(q0_list):]

  def __call__(self, q0, v0, t, f):
    return self.integrate(q0, v0, t, f)


##########################################
# Class for 2nd order Runge-Kutta method #
##########################################

class RungeKutta2(object):

  def __init__(self):
    self._b = np.array([0., 1.])
    self._k = [[],[]]

  def _iter(self, tn, yn_list, f, h):
    self._k[0] = f(tn, yn_list)
    self._k[1] = f(tn+h/2., yn_list+h/2.*self._k[0])
    return yn_list + h*np.sum(self._b[ii]*self._k[ii] for ii in range(2))

  def integrate(self, q0_list, v0_list, t, f):
    y0_list = np.hstack([q0_list, v0_list])
    h = t[1]-t[0]
    ans = np.zeros((t.size, len(y0_list)))
    ans[0,:] = y0_list
    for ii, tt in enumerate(t[:-1]):
      ans[ii+1,:] = self._iter(tt, ans[ii], f, h)
    #return ans.T
    out = ans.T
    return out[:len(q0_list)], out[len(q0_list):]

  def __call__(self, q0, v0, t, f):
    return self.integrate(q0, v0, t, f)
