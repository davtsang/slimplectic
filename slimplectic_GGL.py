from __future__ import division, print_function
from sympy import *
import numpy
import scipy.optimize


def GGLdefs(r, precision=20):
    """ Gives the Collocation points, weights and derivative matrix
    for the Galerkin-Gauss-Lobatto Variational Integrator.
    GGLdefs returns as a tuple:
    GGLxs[r+2] - the numerical x - collocation points array,
    GGLws[r+2] - the numerical weights
    GGLDM[r+2, r+2] - the derivative matrix (as a function of x)
    all evaluated for a system with r intermediate points,
    i.e. (r+2) total collocation points, up to arbitrary precision,
    which is the number of sig. figs. in decimal representation
    """

    # Convenience lambda to evaluate to given precision
    nprec = lambda x: N(x,precision)

    # Set polynomial order n for a given r intermediate points
    # or r+2 total collocation points
    n = r+1

    # find collocation points for the Gauss-Lobatto quadrature
    x = symbols('x')
    GGLxs = list(map(nprec,
                polys.polytools.real_roots(
                    (x**2 -1)*diff(legendre(n, x), x),
                    n+1)))

    # Determine the weight functions
    GGLws = list(map(nprec, [2/(n*(n+1)*(legendre(n, xj))**2) for xj in GGLxs]))

    # Determine the derivative matrix using the grid points evaluated
    # to the right position
    GGLDM = [[0 for xx in range(n+1)] for xx in range(n+1)]
    for i in range(n+1):
        for j in range(n+1):
            if (i == j and j == 0):
                GGLDM[i][j] = nprec(-(1/4)*n*(n+1))
            if (i == j and j == n):
                GGLDM[i][j] = nprec((1/4)*n*(n+1))
            if (i == j and (0 < j) and (j < n)):
                GGLDM[i][j] = S.Zero
            if (i != j):
                GGLDM[i][j] = nprec(legendre(n, GGLxs[i])/(legendre(n, GGLxs[j])*(GGLxs[i]-GGLxs[j])))
                #print GGLDM[i][j]

    return GGLxs, GGLws, GGLDM





def q_Generate_pm(qlist):
    """q_Generate_pm generates the plus and minus doubled variables of qlist
    Output:
    (qplist, qmlist)
    qplist[dof] - the list of symbols of the form q_+
    qmlist[dof] - the list of symbols of the form q_-

    Input:
    qlist[dof] - the 1-d list of symbols that you want to double
    """
    qplist = []
    qmlist = []
    for i in range(len(qlist)):
        qplist.append(Symbol(repr(qlist[i]) + '_+', real=True))
        qmlist.append(Symbol(repr(qlist[i]) + '_-', real=True))
    return qplist, qmlist


def Gen_pi_list(qlist):
    """Generate_pi generates the symbol list for the nonconservative
    discrete momenta pi
    Output: (pi_n_list, pi_np1_list)
    pi_n_list[dof] - list of symbols for the current pi_n
    pi_np1_list[dof] - list of symbols for the next pi_n+1
    Input:
    qlist[dof] - the 1-d list of symbols that you want make
                 momenta for
    """
    pi_n_list = []
    pi_np1_list = []
    for i in range(len(qlist)):
        pi_n_list.append(Symbol("\pi_" + repr(qlist[i]) + "^{[n]}", real=True))
        pi_np1_list.append(Symbol("\pi_" + repr(qlist[i]) + "^{[n+1]}", real=True))
    return pi_n_list, pi_np1_list




def Physical_Limit(q_list, q_p_list, q_m_list, expression):
    """ Physical_Limit takes the physical limit of a function of
    the doubled +/- variables that is taking q_- -> 0 and q_+ -> q
    The q_lists are expected to be 1-d lists.
    If you are passing in q_tables please flatten them using
    something like:
    q_list = [qval for qvallist in qtable for qval in qvallist]

    Physical_Limit outputs PL_Expr an sympy object equivalent to
    expression with the physical limit taken.

    Inputs:
    q_list[dof] - list of sympy objects that correspond to the dof
    q_p_list[dof] - list of sympy objects that correspond to the + dof
    q_m_list[dof] - list of sympy objects that corerspond to the - dof

    Physical_Limit assumes that the q_lists share the same ordering.
    """

    dof_count = len(q_list)

    sub_list = []
    for dof in range(dof_count):
        sub_list.append((q_p_list[dof], q_list[dof]))
        sub_list.append((q_m_list[dof], 0))

    PL_Expr = expression.subs(sub_list)
    return PL_Expr





def GGL_q_Collocation_Table(qlist, cp_count):
    """GGL_q_Collocation_Table creates the symbol tables necessary
    to evaluate a degree of freedom at each of cp_count collocation
    points, calling them qname^[n], qname^(i), qname^[n+1] for each qname in qlist.

    Output:
    __qtable[dof][cp_count]

    Input:
    qlist[dof] - the list of degree of freedom symbols you
                 want to generate collocation symbols for
    cp_count - the collocation point count

    """
    qtable = []
    for q in qlist:
        qvec = [Symbol("{" +repr(q) #+"}"
                              + "^{[n]}}")]
        for i in range(1, cp_count-1):
            qvec.append(Symbol("{" +repr(q) #+"}"
                               + "^{(" + repr(i)+")}}"))
        qvec.append(Symbol("{" +repr(q) #+"}"
                              + "^{[n+1]}}"))
        qtable.append(qvec)

    return qtable

def DM_Sum(DMvec, qlist):
    """Helper function to matrix dot product the DM matrix with a qvector
    Assumes that DMVec is the same length as qlist
    """
    sum = 0
    for j in range(len(DMvec)):
        sum += DMvec[j]*qlist[j]
    return sum





def GGL_Gen_Ld(tsymbol, q_list, qprime_list, L, ddt, r, paramlist = [], precision=20):
    """GGL_Gen_Ld generates the discrete Lagrangian for use in determining
    the GGL variational integrator.

    Outputs:
    (Ld, q_Table)
    Ld -  the algebraic expression for the discrete Lagrangian,
    q_Table[len(q_list)][r+2] -  the array of sympy symbols for
                                 qs at each quadrature point.

    Inputs are:
    tsymbol - the symbol used for the explicit time dependence of the Lagrangian
    q_list[dof] - list of sympy variables (not including time derivatives),
    qprime_list[dof] - list of sympy variables for the time derivatives
                       in the same order
    L(t, q, dotq) - algebraic expression for the full Lagrangian,
                    in terms of q_list and qprime_list variables
    t - sympy variable for time, if used in the Lagrangian
    r - number of intermediate quadrature steps,
    paramlist - the constant parameter substitution list for evaluations
    precision - precision for which evaluations occur
                (should be higher than machine precision
                 for for best results)
    """
    #Initialize GGLdefs for collocation points, weights and derivative matrix
    xs, ws, DM = GGLdefs(r, precision)

    #Create q_Table for all the algebraic variables based on q_list
    #q_Table[len(q_list)][r+2]
    #[[..],..,[qx_i0,...,qx_i(r+2)],..,[..]]

    q_Table = GGL_q_Collocation_Table(q_list, r+2)

    #Create list of times for evaluating L
    t_list = [tsymbol + 0.5*(1 + xi)*ddt for xi in xs]

    #Create dphidt_Table for the algebraic form of dPhi/dt
    #where Phi is the polynomial
    #interpolation of q over the quadrature points
    #dphidt_Table[len(q_list)][r+2]
    #Make sure to multiply by dx/dt = 2/ddt


    dphidt_Table = []
    for qs in q_Table:
        dphidt_Table.append([DM_Sum(DMvec, qs)*2/ddt for DMvec in DM])


    #Create list of substitution pairs used at each quadrature location
    #sublist[r+2][2*len(q_list)]

    sublist = []
    for i in range(r+2):
        pairs = []
        for j in range(len(q_list)):
            pairs.append((q_list[j], q_Table[j][i]))
            pairs.append((qprime_list[j], dphidt_Table[j][i]))
            pairs.append((tsymbol, t_list[i]))
        sublist.append(pairs)

    #Evaluate Ld which is the weighted sum over each point
    Ld = 0
    for i in range(r+2):
        Ld += 0.5*ddt*ws[i]*(L.subs(sublist[i])).subs(paramlist)
    return Ld , q_Table







def GGL_Gen_Kd(tsymbol, q_p_list, q_m_list, qprime_p_list, qprime_m_list, K, ddt, r, paramlist = [], precision=20):
    """GGL_Gen_Kd generates the discrete nonconservative potential
    for use in determining the GGL variational integrator.

    Outputs:
    (Kd, q_p_Table, q_m_Table)
    Kd - the algebraic expression for the discrete nonconservative
         potential
    q_{p/m}_Table[len(q_list)][r+2][2] -
        the array of sympy symbols representing the
        q's at each quadrature point for +/-.

    Inputs are:
    tsymbol - the symbol used for the explicit time dependence
              of the Lagrangian
    q_p_list[dof] - list of sympy variables for the q_+
                    doubled dof,
    q_m_list[dof] - list of sympy variables for the
                    q_- doubled dof,
    qprime_p_list[dof] - list of sympy variables qdot_+
                         doubled dof derivatives,
    qprime_m_list[dof] - list of sympy variables qdot_-
                         doubled dof derivatives,
    K - algebraic expression for the continuous Lagrangian,
        in terms of q_p_list, q_m_list qprime_p_list
        and qprime_m_list variables
    ddt - symbol for the time step
    r - number of intermediate quadrature steps,
    paramlist - the constant parameter substitution
                list for evaluations
    precision - precision for which evaluations occur
                (should be higher than machine precision
                for for best results)
    """
    #Initialize GGLdefs for collocation points, weights and derivative matrix
    xs, ws, DM = GGLdefs(r, precision)

    #Create q_{p/m}_Table for all the algebraic variables based on q_list
    #q_{p/m}_Table[len(qlist)][r+2]
    #[[..],..,[qx_i0,...,qx_i(r+2)],..,[..]]

    q_p_Table = GGL_q_Collocation_Table(q_p_list, r+2)

    q_m_Table = GGL_q_Collocation_Table(q_m_list, r+2)

    #Create dphidt_{p/m}_Table for the algebraic form of dPhi/dt where
    #Phi is the polynomial interpolation of q over the quadrature points
    #dphidt_{p/m}_Table[len(qlist)][r+2]
    #Make sure to multiply by dx/dt = 2/ddt

    dphidt_p_Table = []
    for qs in q_p_Table:
        dphidt_p_Table.append([DM_Sum(DMvec, qs)*2/ddt for DMvec in DM])

        dphidt_m_Table = []
    for qs in q_m_Table:
        dphidt_m_Table.append([DM_Sum(DMvec, qs)*2/ddt for DMvec in DM])

    #Create list of substitution pairs used at each quadrature location
    #sublist[r+2][4*len(qlist)]

    sublist = []
    for i in range(r+2):
        pairs = []
        for j in range(len(q_p_list)):
            pairs.append((q_p_list[j], q_p_Table[j][i]))
            pairs.append((qprime_p_list[j], dphidt_p_Table[j][i]))
        for j in range(len(q_m_list)):
            pairs.append((q_m_list[j], q_m_Table[j][i]))
            pairs.append((qprime_m_list[j], dphidt_m_Table[j][i]))
        sublist.append(pairs)

    #Evaluate Ld which is the weighted sum over each point
    Kd = 0
    for i in range(r+2):
        Kd += 0.5*ddt*ws[i]*(K.subs(sublist[i])).subs(paramlist)
        #print ws[i]
    return Kd , q_p_Table, q_m_Table



def Gen_iter_EOM_List(q_Table, q_p_Table, q_m_Table, pi_n_list, pi_np1_list, Ld, Kd, ddt):
    """
    Gen_iter_EOM_Tables generate the symbolic Equation of Motion Tables
    to be used for iteration

    Output:
    EOM_List[dof*(r+1)] - List of sympy equations of motion
    Here the equations of motion are assumed to be:
    (for i in [1..r])
          [ddt*Ld_Table[0][i] + Kd_Table[0][i],
           pi_1^[n] + ddt*(Ld_Table[0][0] + Kd_Table[0][0]),
           ...
           ddt*Ld_Table[dof][i] + Kd_Table[dof][i],
           pi_1^[n] + ddt*(Ld_Table[dof][0] + Kd_Table[dof][0])]

    The equation for pi_n+1 will be computed separately as it does
    not need to be iterated.

    Inputs:
    q_table[dof][r+2] - Table of sympy variables for dofs
    q_p_table[dof][r+2] - Table of sympy variables for + dofs
    q_m_table[dof][r+2] - Table of sympy variables for - dofs
    pi_n_list[dof] - List of sympy variables for the n.c.
                     discrete momemta at current time
    pi_np1_list[dof] - List of sympy variables for the n.c.
                       discrete momemta at the next step
    Ld - Sympy expression for Ld
    Kd - Sympy expression for Kd
    ddt - Sympy symbol for the time step
    """

    #Define symbol long lists for use with the Physical Limit function
    q_longlist = [q for qvec in q_Table for q in qvec]
    q_p_longlist = [q for qvec in q_p_Table for q in qvec]
    q_m_longlist = [q for qvec in q_m_Table for q in qvec]


    # Create the Ld and Kd parts of the EOM
    # By taking the derivative wrt the appropriate dof and
    # taking the physical limit for Kd

    Ld_EOM_Table = [[diff(Ld, q)
                       for q in qvec]
                      for qvec in q_Table]
    Kd_EOM_Table = [[Physical_Limit(q_longlist,
                                      q_p_longlist,
                                      q_m_longlist,
                                      diff(Kd, q_m))
                       for q_m in qvec]
                      for qvec in q_m_Table]

    # Create Symbolic Equation of Motion Tables
    # We don't have the EOM for pi_n+1 since it will not need to be solved
    # implicitly later

    EOM_List = []
    for i in range(len(Ld_EOM_Table)):
        for j in range(1, len(Ld_EOM_Table[0])-1):
            EOM_List.append(Ld_EOM_Table[i][j]
                                     + Kd_EOM_Table[i][j])
        EOM_List.append(pi_n_list[i]
                                 + Ld_EOM_Table[i][0]
                                 + Kd_EOM_Table[i][0])

    #print EOM_List
    return EOM_List


def Gen_J_Expr_Table(expr_vec, var_vec):
    """Gen_J_Expr_Table generate a table of Jacobian
    output:
    J - Jacobian table of sympy expressions
    where  J[i][j] = d(expr_vec[i])/d(var_vec[j])

    input:
    expr_vec - vector of sympy expressions
    var_vec - vector of sympy variables
    """
    J = []
    for i in range(len(expr_vec)):
        J_vec = []
        for j in range(len(var_vec)):
            J_vec.append(diff(expr_vec[i], var_vec[j]))
        J.append(J_vec)

    return J


def pi_ic_from_qdot(qdot_vec, q_vec,
                     tval, ddt,
                     pi_guess_vec,
                     qi_sol_func, qdot_n_func):
    """This finds the initial condition for the pi vector
    for a given qdot_vec and q_vec initial condition,
    since pi depends on the choice of discretization.

    Outputs:
    pi_init_sol - ndarray for the solution of that matches
                  the initial condition in terms of q and
                  qdot given

    Inputs:
    qdot_vec[dof] - ndarray of initial qdot to be matched
    q_vec[dof] - ndarray of initial q
    tval - float for the initial time
    ddt - float for the time step size
    pi_guess_vec[dof] - initial guess for pi
    qi_sol_func - 1st function returned by Gen_GGL_NC_VI_Map
                  that generates an ndarray for the qi_sol
    qdot_n_func - 4th function returned by Gen_GGL_NC_VI_Map
                  that calculates the value of qdot for a
                  given pi_n
    """
    def fun(pi_vec):
        qi_sol = qi_sol_func(q_vec, pi_vec, tval, ddt)
        qdot_guess = qdot_n_func(qi_sol, q_vec, pi_vec, tval, ddt)
        return qdot_vec - qdot_guess

    return scipy.optimize.root(fun, pi_guess_vec)



def pi_ic_from_qnext(q_next_vec, q_vec,
                     tval, ddt,
                     pi_guess_vec,
                     qi_sol_func, q_np1_func):
    """This finds the initial condition for the pi vector
    for a given qdot_vec and q_vec initial condition,
    since pi depends on the choice of discretization.

    Outputs:
    pi_init_sol - ndarray for the solution of that matches
                  the initial condition in terms of q and
                  qdot given

    Inputs:
    qdot_vec[dof] - ndarray of initial qdot to be matched
    q_vec[dof] - ndarray of initial q
    tval - float for the initial time
    ddt - float for the time step size
    pi_guess_vec[dof] - initial guess for pi
    qi_sol_func - 1st function returned by Gen_GGL_NC_VI_Map
                  that generates an ndarray for the qi_sol
    qdot_n_func - 4th function returned by Gen_GGL_NC_VI_Map
                  that calculates the value of qdot for a
                  given pi_n
    """
    def fun(pi_vec):
        qi_sol = qi_sol_func(q_vec, pi_vec, tval, ddt)
        q_next_guess = q_np1_func(qi_sol, q_vec, pi_vec, tval, ddt)
        return q_next_vec - q_next_guess

    return scipy.optimize.root(fun, pi_guess_vec)


def Gen_GGL_NC_VI_Map(t_symbol,
                      q_list, q_p_list, q_m_list,
                      v_list, v_p_list, v_m_list,
                      Lexpr,
                      Kexpr,
                      r,
                      sym_paramlist = [],
                      sym_precision = 20,
                      eval_modules = "numpy",
                      method = 'implicit',
                      verbose = True,
                      verbose_rational = True
                     ):
    """Gen_GGL_NC_VI_Map generates the mapping functions for the
    Galerkin-Gauss-Lobatto Nonconservative Variational Integrator
    described in Tsang, Galley, Stein & Turner (2015), for a generic
    System specified by the user using sympy symbols.

    Output:
    (qi_table_func, q_np1_func, pi_np1_func, v_n_func)
    A tuple of functions:
    *********************
     qi_sol_func(q_n_vec, pi_n_vec, tval, ddt)
       description: Main function of the mapping, iterates
                    to find the r intermediate values for each
                    degree of freedom q.
       outputs: qi_sol
                   - scipy.OptimizeResult containing values for
                     the intermediate and next values of q used
                     for this (r+2)th order method. This is used
                     by the other functions to calculate the
                     mappings and other values
       inputs: q_n_vec[dof]
                   - list of q_n degree of freedom values at the
                     current step
               pi_n_vec[dof]
                   - list of pi_n nonconservative discrete
                     momentum at the current step
               tval - current step initial time value
               ddt - step size in time.

     q_np1_func(qi_table, ddt)
       description: Computes the next position in the mapping
       outputs: q_np1_vec[dof]
                   - list of next value for each dof
       inputs: qi_table[dof][r+2]
                   - qi_table of iterated starting, intermediate
                     and final values for the dof, created by
                     qi_table_func
               ddt - step size in time

     pi_np1_func(qi_table, ddt)
       description: Computes the next n.c. momenta in the mapping
       outputs: pi_np1_vec[dof]
                   - list of next n.c. momenta value for each dof
       inputs: qi_table[dof][r+2]
                   - qi_table of iterated starting, intermediate
                     and final values for the dof, created by
                     qi_table_func
               ddt - step size in time

    qdot_n_func(qi_table, ddt)
       description: Computes the velocity at the current time
       outputs: v_n_vec[dof]
                   - list of velocity values for each dof
       inputs: qi_table[dof][r+2]
                   - qi_table of iterated starting, intermediate
                     and final values for the dof, created by
                     qi_table_func
               ddt - step size in time

    **********************


    Inputs:
    t_symbol - Symbol used for the time variable in expressions
    q_list[dof] - list of symbols representing the degrees
                  of freedom of the problem
    q_p_list[dof] - list of symbols representing the + doubled
                    degrees of freedom
    q_m_list[dof] - list of symbols representing the - doubled
                    degrees of freedom
    v_list[dof] - list of symbols representing the time
                  derivatives of the dof
    v_p_list[dof] - list of symbols representing the
                    time derivatives of the + doubled dof
    v_m_list[dof] - list of symbols representing the
                    time derivatives of the - doubled dof
    Lexpr - sympy expression for the Lagrangian, L, in terms of
            q_list and v_list variables.
    Kexpr - sympy expression for the nonconservative potential,
            K, in terms of q_p/m_list and v_p/m_list variables.
    r - number of intermediate points to be evaluated for (r+2)
        total collocation points. The order of the GGL NC VI
        method is given by (2r + 2).
    sym_paramlist - list of symbolic parameters to be substituted
                    for evaluations (ie physical constants). All
                    such non-dof symbols must be substituted for
                    numerical values here. If this is not the case
                    an error during the lambdification will occur.
    sym_precision - precision which the GGL constants are
                    evaluated to, prior to lambidification to
                    machine precision. This is to prevent
                    roundoff error from building up
    eval_modules - modules that need to be used to numerically
                   evaluate Lexpr and Kexpr
    verbose - boolean value that turns on and off verbose output
              defaults to True
    verbose_rational - boolean value that sets the verbose output
                       if True numerical values will be rationalized
                       if False values will be left as floats.
    """

    #Define the symbol for h, that we will use in the algebraic
    #expressions
    ddt_symbol = Symbol('h_{GGL}')

    #Determine the Ld and Kd symbolic expressions for this system
    #As well as the q symbols for each dof and collocation point

    Ld, q_Table\
           = GGL_Gen_Ld(t_symbol,
                        q_list, v_list,
                        Lexpr,
                        ddt_symbol,
                        r,
                        paramlist = sym_paramlist,
                        precision = sym_precision)
    Kd, q_p_Table, q_m_Table \
           = GGL_Gen_Kd(t_symbol,
                        q_p_list,
                        q_m_list,
                        v_p_list,
                        v_m_list,
                        Kexpr,
                        ddt_symbol,
                        r,
                        paramlist = sym_paramlist,
                        precision = sym_precision)

    #Generate momenta symbol lists
    pi_n_list, pi_np1_list = Gen_pi_list(q_list)

    #Generate the Equation of Motion Table for
    # q^[n] and q^(i)'s, but not q^[n+1]
    # (since pi_n+1 will be evaluated directly later)
    EOM_List = Gen_iter_EOM_List(q_Table, q_p_Table, q_m_Table,
                                 pi_n_list, pi_np1_list,
                                 Ld, Kd,
                                 ddt_symbol)




    #return EOM_List
    #qi symbol list: q^i and q^n+1 for each dof
    #the variables to be solved for by the implicit
    #method
    qi_symbol_list = []
    for dof in range(len(q_Table)):
        for i in range(1, r+2):
            qi_symbol_list.append(q_Table[dof][i])



    #Generate flat list of the symbols for lambdification
    full_variable_list = []
    for i in range(len(q_list)):
        for j in range(r+2):
            full_variable_list.append(q_Table[i][j])
        full_variable_list.append(pi_n_list[i])
    full_variable_list.append(t_symbol)
    full_variable_list.append(ddt_symbol)
    #print full_variable_list


    def Convert_EOM_Args(qi_vec, qn_vec, pi_nvec, tval, ddt):
        #Convert_EOM_Args returns an argument list for
        #for the lambdified EOM functions.
        #this should be [q_1^[n], q_1^(i), q_1^[n+1], pi_1^n,...]
        #these should all be numerical values
        EOM_arg_list = []
        dof_count = len(qn_vec)
        for dof in range(dof_count):
            EOM_arg_list.append(qn_vec[dof])
            for i in range(r+1):
                EOM_arg_list.append(qi_vec[dof*(r+1) + i])
            EOM_arg_list.append(pi_nvec[dof])
        EOM_arg_list.append(tval)
        EOM_arg_list.append(ddt)
        #print EOM_arg_list
        return EOM_arg_list

    #Generate the list of functions for evaulating the EOM
    EOM_Func_List = [lambdify(tuple(full_variable_list),
                              EOM,
                              modules=eval_modules)
                     for EOM in EOM_List]



    # Generate the J_qi_vec sympy variables to take
    # the Jacobian with respect to
    J_qi_vec = []
    for i in range(len(q_Table)):
        for j in range(1,r+2):
            J_qi_vec.append(q_Table[i][j])

    # Generate the Jacobian function table
    J_Expr_Table = Gen_J_Expr_Table(EOM_List, J_qi_vec)
    J_Func_Table = [[lambdify(tuple(full_variable_list),
                              J_Expr, modules=eval_modules)
                     for J_Expr in J_Expr_Vec]
                    for J_Expr_Vec in J_Expr_Table]


    # EOM_Val_Vec is the function to be passed to
    # scipy.optimize.root() that returns the Equation
    # of Motion evaulations that should be zero for
    # the correct values extra arguments qn_vec, pi_nvec,
    # t, and ddt should be passed as well

    def EOM_Val_Vec(qi_vec, qn_vec, pi_nvec, tval, ddt):
        # First convert the argument list for
        # the lambdified functions
        EOM_arg_list = Convert_EOM_Args(qi_vec,
                                        qn_vec,
                                        pi_nvec,
                                        tval,
                                        ddt)
        #print EOM_arg_list
        #Next we evaulate the EOM functions
        #in EOM_List
        out = numpy.array([EOM_Func(*tuple(EOM_arg_list))
                            for EOM_Func in EOM_Func_List])
        #print out
        return out



    # EOM_J_Matrix is the function to be passed
    # to scipy.optimize.root() that returns the
    # Jacobian matrix for

    def EOM_J_Matrix(qi_vec, q_n_vec, pi_n_vec, tval, ddt):
        # First convert the argument list for
        # the lambdified functions
        EOM_arg_list = Convert_EOM_Args(qi_vec,
                                        q_n_vec,
                                        pi_n_vec,
                                        tval,
                                        ddt)
        #Next Evaluate the J_Matrix
        J_Matrix = [[J_Func(*tuple(EOM_arg_list))
                    for J_Func in J_Func_Vec]
                    for J_Func_Vec in J_Func_Table]
        #print "J_matrix:"
        #print J_Matrix

        return numpy.array(J_Matrix)







    if method == 'explicit':
        #print 'EXPLICIT METHOD'
        qi_func_args = []
        for dof in range(len(q_Table)):
            qi_func_args.append(q_Table[dof][0])
        for dof in range(len(pi_n_list)):
            qi_func_args.append(pi_n_list[dof])
        qi_func_args.append(t_symbol)
        qi_func_args.append(ddt_symbol)


        qi_sol_dict = solve(EOM_List, qi_symbol_list, dict=True)
#        print qi_symbol_list[0]
#        print qi_sol_dict
#        print qi_sol_dict[0]
        if not qi_sol_dict:
            print("ERROR: explicit solve failed, try implicit solution")
            return
        qi_sol_list = [qi_sol_dict[0][qi_symbol]
                        for qi_symbol in qi_symbol_list]
#        print qi_sol_list
        qi_func_list = [lambdify(qi_func_args,
                                 qi_sol,
                                 modules=eval_modules)
                        for qi_sol in qi_sol_list]

        #print qi_sol_list
        #print qi_sol_list





    #These are the output functions:
    #############################


    if method == 'explicit':
        def qi_sol_func_explicit(q_n_vec, pi_n_vec, tval, ddt, root_args={}):
            """This function evaluates the explicit equations
            for the intemediate points [{q_1^(i)_0}, q_1^[n+1]_0, ...]
            to generate the iterated intermediate results
            for the explicit GGL-NC-VI method.

            Output:
            qi_sol - the ndarray that contains the value of qi
            Input:
            q_n_vec[dof] - ndarray of current q_n values
            pi_n_vec[dof] - ndarray of current pi_n values
            tval - float for the current value of time
            ddt - float for the size of the time step
            """
            #Populate qi_0 array for nsolve()

            qi_arg_vals = []
            for dof in range(len(q_n_vec)):
                qi_arg_vals.append(q_n_vec[dof])
            for dof in range(len(pi_n_vec)):
                qi_arg_vals.append(pi_n_vec[dof])
            qi_arg_vals.append(tval)
            qi_arg_vals.append(ddt)

            qi_sol = [qi_func(*tuple(qi_arg_vals) )
                                  for qi_func in qi_func_list]
            #print qi_arg_vals
            #print qi_sol

            return numpy.array(qi_sol)





    def qi_sol_func_implicit(q_n_vec, pi_n_vec, tval, ddt,
                             root_args = {'tol':1e-10}):
        """This function uses q_n_vec as a guess for each of the
        intermediate points [{q_1^(i)_0}, q_1^[n+1]_0, ...]
        to generate the iterated intermediate results
        for the implicit GGL-NC-VI method.

        Output:
        qi_sol - nd.array qi_sol for the
                 results of that give the roots of the appropriate
                 discrete equations of motion.
        Input:
        q_n_vec[dof] - ndarray of current q_n values
        pi_n_vec[dof] - ndarray of current pi_n values
        tval - float for the current value of time
        ddt - float for the size of the time step
        root_args - dictionary of arguments to be included in the
                    scipy.optimize.root() method e.g.
                    {'method':'hybr', 'tol': 1e-8}

        """
        #Populate qi_0 array for nsolve()
        qi_0 = []
        for i in range(len(q_n_vec)):
            for j in range(1, r+2):
                qi_0.append(q_n_vec[i])
        qi_0 = numpy.array(qi_0)

        qi_sol = scipy.optimize.root(**dict(list({'fun': EOM_Val_Vec,
                                           'x0': qi_0,
                                           'args': (q_n_vec,
                                                    pi_n_vec,
                                                    tval,
                                                    ddt),
                                           'jac': EOM_J_Matrix
                                          }.items())
                                            + list(root_args.items())
                                         )
                                    )
        return qi_sol.x

    def q_np1_func(qi_sol, q_n_vec, pi_n_vec, tval, ddt):
        """This function uses the qi_sol from the first
        Gen_GGL_NC_VI_Map returned function to calculate
        the q's for the next step. In this case it will be just a
        simple lookup.

        Outputs:
        q_np1_vec[dof] - ndarray of next q values

        Inputs:
        qi_sol[dof*(r+1)] - ndarray of qi_values from
                            qi_sol_func
        q_n_vec[dof] - ndarray of current q_n values
        pi_n_vec[dof] - ndarray of current pi_n values
        tval - float for the current value of time
        ddt - float for the size of the time step

        """
        q_np1_vec = [qi_sol[dof*(r+1)+r] for dof in range(len(q_n_vec))]
        return numpy.array(q_np1_vec)


    # Here are some variables and functions to help
    # with creating the output functions

    q_longlist = [q for qvec in q_Table for q in qvec]
    q_p_longlist = [q for qvec in q_p_Table for q in qvec]
    q_m_longlist = [q for qvec in q_m_Table for q in qvec]
    pi_n_expr =  [diff(Ld, q_Table[dof][-1])
                  + Physical_Limit(q_longlist,
                                   q_p_longlist,
                                   q_m_longlist,
                                   diff(Kd, q_m_Table[dof][-1]))
                   for dof in range(len(q_Table))]
    #return pi_n_expr



    pi_Func_Vec = [lambdify(full_variable_list,
                            expr, modules = eval_modules)
                            for expr in pi_n_expr]



    def pi_np1_func(qi_sol, q_n_vec, pi_n_vec, tval, ddt):
        """This function uses the qi_sol from the first
        Gen_GGL_NC_VI_Map returned function to calculate
        the pi's for the next step. This involves evaluating the last
        equation of motion dL_d/d(q^[n+1]) + ...

        Outputs:
        pi_np1_vec[dof] - ndarray of next pi values

        Inputs:
        qi_sol[dof*(r+1)] - ndarray of qi_values from
                            qi_sol_func
        q_n_vec[dof] - ndarray of current q_n values
        pi_n_vec[dof] - ndarray of current pi_n values
        tval - float for the current value of time
        ddt - float for the size of the time step
        """
        EOM_Arg_list = Convert_EOM_Args(qi_sol,
                                        q_n_vec,
                                        pi_n_vec,
                                        tval,
                                        ddt)
        pi_np1_vec = [pi_func(*tuple(EOM_Arg_list))
                      for pi_func in pi_Func_Vec]

        #print qi_sol, q_n_vec, pi_n_vec, ddt
        #print pi_np1_vec

        return numpy.array(pi_np1_vec)


    #We need DM for the dotq function
    xs, ws, DM = GGLdefs(r)

    def qdot_n_func(qi_sol, q_n_vec, pi_n_vec, tval, ddt):
        """This function uses the qi_sol from the first
        Gen_GGL_NC_VI_Map returned function to calculate
        the qdot velocities for current step. This involves
        evaluating qdot using the derivative matrix defined by
        the GGL_defs function, this will be evaluated when this
        function is generated, rather than each time it is called.

        Outputs:
        pi_np1_vec[dof] - ndarray of next pi values

        Inputs:
        qi_sol[dof*(r+1)] - ndarray of qi_values from
                            qi_sol_func
        q_n_vec[dof] - ndarray of current q_n values
        pi_n_vec[dof] - ndarray of current pi_n values
        tval - float for the current value of time
        ddt - float for the size of the time step
        """
        qi_table = []
        for dof in range(len(q_n_vec)):
            qi_vec = [q_n_vec[dof]]
            for i in range(r+1):
                qi_vec.append(qi_sol[dof*(r+1)+i])
            qi_table.append(qi_vec)

        qdot_vec =[numpy.dot(numpy.array(DM), qi_vec)[0]*2/ddt
                     for qi_vec in qi_table]
        return numpy.array(qdot_vec, dtype=float)

    #Verbose output:

    if verbose:
        print('===================================')
        print('For Lagrangian:')
        print('\t L = ' + latex(Lexpr))
        print('and K-potential:')
        print('\t K = ' + latex(Kexpr))
        print('********************')
        print('The Order '+ repr(2*r+2) +' discretized Lagrangian is:')

        if verbose_rational:
            print('\t L_d^n = '
                  + latex(nsimplify(Ld,
                                    tolerance=1e-15,
                                    rational = True)))
        else:
            print('\t L_d^n = ' + latex(simplify(Ld)))

        print('The Order '+ repr(2*r+2) +' discretized K-potential is:')
        if verbose_rational:
            print('\t K_d^n = '
                  + latex(nsimplify(Kd,
                                    tolerance=1e-12,
                                    rational=verbose_rational)))
        else:
            print('\t K_d^n = ' + latex(simplify(Kd)))

        print('********************')
        print('The Order '+ repr(2*r+2) +' Discretized Equations of motion:')

        if verbose_rational:
            for dof in range(len(q_Table)):
                for i in range(r+1):
                    print( '\t0 = ' + latex(nsimplify(expand(EOM_List[dof*(r+1) + i]),
                                            tolerance=1e-15,
                                            rational=verbose_rational)))
                print('\t0 = '+ latex(nsimplify(-pi_np1_list[dof] + expand(pi_n_expr[dof]),
                                                 tolerance=1e-15,
                                                 rational=verbose_rational)))

        else:
            for dof in range(len(q_Table)):
                for i in range(r+1):
                    print( '\t0 = ' + latex(simplify(expand(EOM_List[dof*(r+1) + i]))))
                print('\t0 = ' + latex(-pi_np1_list[dof] + simplify(expand(pi_n_expr[dof]))))
        print('===================================')




    #########################
    if method == 'implicit':
        return qi_sol_func_implicit, q_np1_func, pi_np1_func, qdot_n_func
    elif method == 'explicit':
        return qi_sol_func_explicit, q_np1_func, pi_np1_func, qdot_n_func
    else:
        print("GGL_NC_VI ERROR: method = " + method + " unknown.")
        return
