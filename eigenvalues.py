''' Module that contains functions for processes dealing with eigenvalues.
    Contains:
        1) Iterating through a Markov Chain.
        2) Finding a steady-state vector for a stochastic matrix if possible.
        3) Solves the characteristic equation for eigenvalues.
        4) Finds corresponding eigenvectors for eigenvalues.
        5) Construct the matrices for diagonalizations.
        6) Performs the spectral decomposition of a matrix.
        7) Solves the quadratic form Q(x) = (x^T) * A * x
        8) Categorizes matrices used in quadratic forms.
        9) Performs the SVD on a matrix.

    NOTE: Vector inputs should be lists of numbers.
    NOTE: Matrix inputs should be of the form of nested lists. Each inner list
    should correspond with a row in the matrix.
        Ex) A = [[1, 2], [2, 1]] is a matrix with the first row being [1, 2] and
            the second row is [2, 1]
    NOTE: I am only concerning myself with matrices that have real eigenvalues.
        Complex eigenvalues would come in a subsequent update.

    Author: Brendan Boyer
    Version: 1.0'''

import operations as op
import solving_systems as ss
import sympy as sp
import copy

''' Iterates through a Markov Chain n times.

    Precondition: [matrix] is stochastic, or when the entries are nonzero and the
    columns add to one. The size of [matrix] is n x n.
    Precondition: [vector] is n x 1.
    Precondition: [n] is nonnegative.'''
def markov_iteration(matrix, vector, n):
    if n < 1:
        return vector
    else:
        next = op.matrix_vector_multi(matrix, vector)
        return markov_iteration(matrix, next, n-1)

''' Finds a steady-state vector for a stochastic matrix.

    Precondition: [matrix] is stochastic, or when the entries are nonzero and the
    columns add to one. The size of [matrix] is n x n.
    Postcondition: If the steady-state vector exists, return a vector of length
    n. If it depends on the initial state (i.e. it has more than 1 free variable), return [inf]. Otherwise, return [0].'''
def steady_state(matrix):
    identity = op.identity_matrix(len(matrix))
    neg_iden = op.matrix_scalar_multi(-1, identity)
    p_minus_i = op.matrix_addition(matrix, neg_iden)
    if op.is_invertible(p_minus_i):
        return [0]
    else:
        for row in p_minus_i:
            row.append(0)
        reduced = ss.echelon_form(p_minus_i, 0, 0)
        for row in range(0, len(reduced)):
            if ss.all_zeroes(reduced[row]):
                reduced[row] = [1 for x in reduced[row]]
        reduced = ss.rref(reduced)
        if len(ss.free_indexes(reduced)) > 0:
            return [float('inf')]
        else:
            return ss.const_vector(reduced)

''' Calculates the determinant of a characteristic equation.

    Precondition: [matrix] is n x n.'''
def determinant_charact(sym, matrix):
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    determinant = 0
    for row in range(0, len(matrix)):
        new_matrix = []
        coeff = matrix[row][0]
        sign = pow(-1, row)
        for ind in range(0, len(matrix)):
            if row != ind:
                new_matrix.append(matrix[ind][1:])
        determinant += coeff * sign * determinant_charact(sym, new_matrix)
    return sp.factor(determinant)

''' Solves the characteristic equation for some matrix.

    Precondition: [matrix] is n x n.
    Postcondition: Eigenvalues are sorted least to greatest.'''
def solve_charact(matrix):
    x = sp.symbols('x')
    for row in range(0, len(matrix)):
        expr = matrix[row][row] - x
        matrix[row][row] = expr
    det = determinant_charact(x, matrix)
    result = [sp.simplify(x).evalf() for x in sp.solve(det, x)]
    result = [sp.re(x) for x in result]
    return sorted(result)

''' Finds a corresponding eigenvector for each eigenvalue.

    Precondition: [matrix] is n x n.
    Postcondition: Each vector makes all free variables equal one.'''
def find_eigenvectors(matrix):
    orig_matrix = copy.deepcopy(matrix)
    values = solve_charact(matrix)
    values = list(dict.fromkeys(values))
    vectors = []
    for eig in values:
        new_matrix = copy.deepcopy(orig_matrix)
        for row in range(0, len(new_matrix)):
            new_matrix[row][row] = new_matrix[row][row] - eig
            new_matrix[row].append(0)
        reduced = ss.rref(new_matrix)
        vect = []
        for row in range(0, len(reduced)):
            if ss.all_zeroes(reduced[row]):
                reduced[row][len(reduced[0]) - 1] = 1
            else:
                for col in range(row + 1, len(reduced[0]) - 1):
                    reduced[row][len(reduced[0]) - 1] -= reduced[row][col]
            vect.append(reduced[row][len(reduced[0]) - 1])
        vectors.append(vect)
    return vectors

''' Returns the eigenvector matrix for a diagonalizable matrix.

    Postcondition: returns a matrix as defined above if [matrix] is diagonalizable.
                   returns [[]] if not diagonalizable.
    Postcondition: Eigenvectors are ordered based on eigenvalues least to greatest.'''
def diagonalizable_p(matrix):
    vectors = find_eigenvectors(matrix)
    if op.is_basis_reals(vectors, len(matrix)):
        p = []
        for col in range(0, len(vectors)):
            p_row = []
            for row in range(0, len(vectors[0])):
                p_row.append(vectors[row][col])
            p.append(p_row)
        return p
    else:
        return [[]]

''' Returns the eigenvalue matrix for a diagonalizable matrix.

    Postcondition: returns a matrix as defined above if [matrix] is diagonalizable.
                   returns [[]] if not diagonalizable.
    Postcondition: Eigenvalues are ordered least to greatest.'''
def diagonalizable_d(matrix):
    orig = copy.deepcopy(matrix)
    values = solve_charact(matrix)
    placements = []
    num_vects = 0
    for val in values:
        new_matrix = copy.deepcopy(orig)
        for row in range(0, len(new_matrix)):
            new_matrix[row][row] = new_matrix[row][row] - val
        ech_form = ss.echelon_form(new_matrix, 0, 0)
        for row in ech_form:
            if ss.all_zeroes(row):
                placements.append(val)
                num_vects += 1
    result = []
    for ind in range(0, num_vects):
        row = []
        for col in range(0, num_vects):
            if ind == col:
                row.append(placements[ind])
            else:
                row.append(0)
        result.append(row)
    return result

''' Computes the spectral decomposition of some matrix.

    Precondition: [matrix] is symmetric.
    Postcondition: Each entry of the result corresponds to one column of the matrix.
        Adding each entry of the result will yield [matrix].'''
def spectral(matrix):
    orig_matrix = copy.deepcopy(matrix)
    values = solve_charact(matrix)
    vectors = find_eigenvectors(orig_matrix)
    vectors = op.orthonormal(vectors)
    vectors = op.transpose(vectors)
    comp = []
    for val, vect in zip(values, vectors):
        vect = [vect]
        rank_mat = op.matrix_multiplication(op.transpose(vect), vect)
        res = op.matrix_scalar_multi(val, rank_mat)
        comp.append(res)
    return comp

''' Solves an equation of the quadratic form Q(x) = (x^T)*A*x.

    Precondition: [matrix] is symmetric and size n.
    Precondition: [x] is length n.'''
def quad_solve(matrix, x):
    a_x = op.matrix_vector_multi(matrix, x)
    combo = [a*b for a,b in zip(x, a_x)]
    total = 0
    for num in combo:
        total += num
    return total

''' Determines whether a list of numbers includes postive, negative, or a mix of signed numbers.

    Postcondition: In the result, the first element denotes existence of positive numbers,
        and the second element denotes existence of negative numbers.
    Postcondition: A zero list will return [False, False].'''
def signs(lst):
    pos = False
    neg = False
    i = 0
    while not (pos * neg) and i < len(lst):
        if lst[i] > 0:
            pos = True
        elif lst[i] < 0:
            neg = True
        i += 1
    return [pos, neg]

''' Determines what type of quadratic form a matrix would produce. Possibilities include:
        1) Positive Definite-> All positive eigenvalues.
        2) Positive Semidefinite-> Eigenvalues are nonnegative.
        3) Indefinite-> Mix of positive and negative eigenvalues.
        4) Negative Semidefinite-> Eigenvalues are nonpositive.
        5) Negative Definite-> All negative eigenvalues.

    Precondition: [matrix] is symmetric.
    Postcondition: Returns a string containing one of the categories above. The string
        is written as it is written above (Ex- Indefinite).
    Postcondition: A zero matrix will result in "Zero".'''
def quad_category(matrix):
    values = solve_charact(matrix)
    if values.count(0) == len(values):
        return "Zero"
    elif values.count(0) > 0:
        val_signs = signs(values)
        pos = val_signs[0]
        neg = val_signs[1]
        if pos and neg:
            return "Indefinite"
        elif pos:
            return "Positive Semidefinite"
        else:
            return "Negative Semidefinite"
    else:
        val_signs = signs(values)
        pos = val_signs[0]
        neg = val_signs[1]
        if pos and neg:
            return "Indefinite"
        elif pos:
            return "Positive Definite"
        else:
            return "Negative Definite"

''' Performs the Singular Value Decomposition for some matrix.

    Postcondition: Written such that each element of the result corresponds to the
        following:
            i = s_i * u_i * v^T_i where s_i is a singular value, u_i is the left
            singular vector that corresponds to s_i, and v^T_i is the right
            singular vector that corresponds to s_i.'''
def svd(matrix):
    a_t_a = op.matrix_multiplication(op.transpose(copy.deepcopy(matrix)), matrix)
    values = solve_charact(copy.deepcopy(a_t_a)) #eigenvalues
    values = [pow(x, 1/2) for x in values] #singular values
    vects = find_eigenvectors(a_t_a)
    vects = op.orthonormal_vectors(vects) #right singular vectors
    left_vects = []
    for v, s in zip(vects, values):
        if s != 0:
            add = op.scalar_multi((1/s), op.matrix_vector_multi(matrix, v))
            left_vects.append(add)
    mat_elems = []
    for u in range(0, len(left_vects)):
        mat = []
        for e in left_vects[len(left_vects) - 1 - u]:
            row = [e * x for x in vects[len(vects) - 1 - u]]
            mat.append(op.scalar_multi(values[len(left_vects) - u], row))
        mat_elems.append(mat)
    return mat_elems
