''' Module that contains functions for various vector and matrix operations.
    Contains:
        1) Vector addition and scalar multiplication
        2) Matrix-vector multiplication
        3) Matrix addition and scalar multiplication
        4) Transpose Matrix creation
        5) Matrix-matrix multiplication
        6) Form identity matrices
        7) Check invertibility and form inverses
        8) Get coordinates of a vector with respect to a basis.
        9) Change of basis for some vector.
        10) Matrix determinants.
        11) Matrix determinants if two matrices are multiplied together.
        12) Cramer's Method
        13) Determining if a basis spans some space R^n
        14) Calculates inner product, length, and distance of vectors.
        15) Constructs an orthonormal matrix from a set of vectors.
        16) Computes the weights and components for orthogonal decomposition.
        17) Performs the Gram-Schmidt process to form an orthogonal basis.
        18) Constructs the QR decomposition for some matrix.
        19) Computes least-squares problems based on QR decomposition.

    NOTE: Vector inputs should be lists of numbers.
    NOTE: Matrix inputs should be of the form of nested lists. Each inner list
    should correspond with a row in the matrix.
        Ex) A = [[1, 2], [2, 1]] is a matrix with the first row being [1, 2] and
            the second row is [2, 1]

    Author: Brendan Boyer
    Version: 1.0'''

import solving_systems as ss
import copy
import math

''' Adds two vectors.

    Precondition: [v1] and [v2] must be the same length.'''
def add_vectors(v1, v2):
    return [a+b for a, b in zip(v1, v2)]

''' Multiplies a vector by a scalar.'''
def scalar_multi(c, v1):
    return [c*a for a in v1]

''' Multiplies a matrix by a vector.

    Precondition: [matrix] has m rows and n columns.
    Precondition: [vect] has length n.
    Postcondition: [result] is of length m.'''
def matrix_vector_multi(matrix, vect):
    result = []
    total = 0
    for row in matrix:
        combo = [a*b for a, b in zip(row, vect)]
        for num in combo:
            total += num
        result.append(total)
        total = 0
    return result

''' Adds two matrices together.

    Precondition: [a] and [b] must be the same size.'''
def matrix_addition(a, b):
    result = []
    for row in range(0, len(a)):
        new_row = add_vectors(a[row], b[row])
        result.append(new_row)
    return result

''' Multiplies a matrix by a scalar.'''
def matrix_scalar_multi(c, a):
    for row in range(0, len(a)):
        a[row] = scalar_multi(c, a[row])
    return a

''' Forms a vector from rows of a matrix.'''
def form_vector_from_rows(matrix, col):
    result = []
    for row in matrix:
        result.append(row[col])
    return result

''' Forms the transpose of a matrix. '''
def transpose(matrix):
    result = []
    for row in range(0, len(matrix[0])):
        matrix_row = form_vector_from_rows(matrix, row)
        result.append(matrix_row)
    return result

''' Multiplies a matrix by another matrix.

    Precondition: [a] is size m x n.
    Precondition: [b] is size n x p.
    Postcondition: [result] is size m x p.'''
def matrix_multiplication(a, b):
    result = []
    for col in range(0, len(b[0])):
        vect = form_vector_from_rows(b, col)
        multi = matrix_vector_multi(a, vect)
        result.append(multi)
    return transpose(result)

''' Creates the identity matrix for the given space.

    Precondition: [n] is greater than 0 (>= 1).'''
def identity_matrix(n):
    identity = []
    for row in range(0, n):
        row_vect = []
        for col in range(0, n):
            if row == col:
                row_vect.append(1)
            else:
                row_vect.append(0)
        identity.append(row_vect)
    return identity

''' Checks if a matrix is invertible. '''
def is_invertible(matrix):
    if len(matrix) != len(matrix[0]):
        return False
    elif ss.lin_indep_homogeneous(transpose(matrix)):
        return True
    else:
        return False

    '''identity = identity_matrix(len(matrix))
    reduced = ss.rref(matrix)
    if identity == reduced:
        return True
    else:
        return False'''

''' Returns the inverse of a matrix.
    If matrix is not invertible, an empty list will be returned.'''
def inverse(matrix):
    matrix_copy = copy.deepcopy(matrix)
    if is_invertible(matrix):
        identity = identity_matrix(len(matrix))
        matrix_to_add = copy.deepcopy( matrix_copy)
        for row in range(0, len(matrix_to_add)):
            matrix_to_add[row] = matrix_to_add[row] + identity[row]
        reduced = ss.rref(matrix_to_add)
        inverse = []
        for row in range(0, len(reduced)):
            row_to_add = []
            for col in range(0, len(matrix)):
                row_to_add.append(reduced[row][col + len(matrix)])
            inverse.append(row_to_add)
        return inverse
    else:
        return []

''' Returns the coordinates of a vector relative to some basis.

    Precondition: [basis] is a linearly independent set of vectors that span
    some space of interest, where each vector should be represented as described above.'''
def coords_from_basis(vector, basis):
    matrix = []
    for row in range(0, len(basis[0])):
        matrix_row = []
        for col in range(0, len(basis)):
            matrix_row.append(basis[col][row])
        matrix.append(matrix_row)
        matrix[row].append(vector[row])
    reduced = ss.rref(matrix)
    coords = []
    for row in reduced:
        coords.append(row[(len(reduced[0]) - 1)])
    return coords

''' Change of basis. Converts a vector whose coordinates correspond to some basis
    [b] to a vector whose coordinates correspond to some basis [c].

    Precondition: [b] and [c] are linearly independent sets of vectors that span
    some space of interest, where each vector should be represented as described above.'''
def change_of_basis(vector, b, c):
    x_b = coords_from_basis(vector, b)
    b_c = []
    for row in b:
        add = coords_from_basis(row, c)
        b_c.append(add)
    matrix = []
    for row in range(0, len(b_c[0])):
        matrix_row = []
        for col in range(0, len(b_c)):
            matrix_row.append(b_c[col][row])
        matrix.append(matrix_row)
    b_c_inverse = inverse(matrix)
    for row in range(0, len(b_c_inverse)):
        b_c_inverse[row].append(x_b[row])
    reduced = ss.rref(b_c_inverse)
    coords = []
    for row in reduced:
        coords.append(row[(len(reduced[0]) - 1)])
    return coords

''' Calculates the determinant of a matrix.

    Precondition: [matrix] is square.'''
def determinant(matrix):
    ech_form = ss.echelon_form(matrix, 0, 0)
    det = 1
    for index in range(0, len(matrix)):
        det = det * matrix[index][index]
    return det

''' Calculates the determinant of two matrices multiplied together.

    Precondition: [a] and [b] are sqaure and the same size.'''
def determinant_multiplied(a, b):
    return determinant(a) * determinant(b)

''' Implementation of Cramer's Rule. This rule is as follows:
    The unique solution of Ax = b has entries given by
    x_i = (det(A_i(b)) / det(A)) for i = 1, .., n where
    A_i(b) = [a1, .., b (in column i), .., an]

    Precondition: [matrix] is square and invertible.'''
def cramers(matrix, b):
    det_calc = copy.deepcopy(matrix)
    det_a = determinant(det_calc)
    x = []
    for n in range(0, len(matrix)):
        matrix_i = copy.deepcopy(matrix)
        for row in range(0, len(matrix_i)):
            matrix_i[row][n] = b[row]
        det_a_i = determinant(matrix_i)
        x.append(det_a_i / det_a)
    return x

''' Determines whether a set of vectors is a basis for some space of dimension n.
    Ex) [[1, 0, 0], [0, 1, 0], [0, 0, 1]] forms a basis for R^3.
        [[1, 0, 0], [0, 1, 0], [1, 1, 0]] does not.

    Precondition: Each vector in [vectors] is of length n.'''
def is_basis_reals(vectors, n):
    if ss.lin_indep_homogeneous(vectors) and len(vectors) == n:
        return True
    else:
        return False

''' Calculates the inner product of two vectors.

    Precondition: [v1] and [v2] must be the same length.'''
def inner_prod(v1, v2):
    result = 0
    for ind in range(0, len(v1)):
        result += v1[ind] * v2[ind]
    return result

''' Calculates the length of a vector.'''
def length(v1):
    result = 0
    for ind in range(0, len(v1)):
        result += pow(v1[ind], 2)
    return pow(result, 1/2)

''' Calculates the distance between two vectors.

    Precondition: [v1] and [v2] must be the same length.'''
def distance(v1, v2):
    return length(add_vectors(v1, scalar_multi(-1, v2)))

'''Constructs orthonormal versions of a set of vectors.

    Precondition: Each vector in [vectors] is the same length.
    Precondition: No vector in [vectors] can be the zero vector.'''
def orthonormal_vectors(vectors):
    normal = []
    for vect in vectors:
        vect_len = length(vect)
        new_vect = scalar_multi(1/vect_len, vect)
        normal.append(new_vect)
    return normal

''' Constructs an orthonormal matrix from a set of vectors.

    Precondition: Each vector in [vectors] is the same length.
    Postcondition: Returns a matrix with the columns being orthonormal corresponding
        to each vector in [vectors].'''
def orthonormal(vectors):
    normal = orthonormal_vectors(vectors)
    matrix = []
    for col in range(0, len(normal[0])):
        new_row = []
        for row in range(0, len(normal)):
            new_row.append(normal[row][col])
        matrix.append(new_row)
    return matrix

''' Computes the orthogonal decomposition weights of some vector for some basis.

    Precondition: [basis] is a valid basis for the space of interest.
    Precondition: Each vector in [basis] as well as [y] is the same length.
    Postcondition: Weights are ordered in the same order as the vectors in [basis].'''
def ortho_decomp_coords(y, basis):
    coords = []
    for vect in basis:
        weight = inner_prod(y, vect) / inner_prod(vect, vect)
        coords.append(weight)
    return coords

''' Computes the individual components of a vector that is an element of the basis.

    Precondition: [basis] is a valid basis for the space of interest.
    Precondition: Each vector in [basis] as well as [y] is the same length.'''
def ortho_decomp_basis_vect_comps(y, basis):
    coords = ortho_decomp_coords(y, basis)
    vect = []
    for num, vector in zip(coords, basis):
        new_vect = scalar_multi(num, vector)
        vect.append(new_vect)
    return vect

''' Computes the total component of a vector that is an element of the basis.

    Precondition: [basis] is a valid basis for the space of interest.
    Precondition: Each vector in [basis] as well as [y] is the same length.'''
def ortho_decomp_basis_vect(y, basis):
    coords = ortho_decomp_coords(y, basis)
    vects = ortho_decomp_basis_vect_comps(y, basis)
    vect = 0
    for ind in vects:
        if vect == 0:
            vect = ind
        else:
            vect = add_vectors(vect, ind)
    return vect

''' Computes the component of a vector that is an element of the space orthogonal
    to the basis.

    Precondition: [basis] is a valid basis for the space of interest.
    Precondition: Each vector in [basis] as well as [y] is the same length.'''
def ortho_decomp_non_basis_vect(y, basis):
    ortho = ortho_decomp_basis_vect(y, basis)
    return add_vectors(y, scalar_multi(-1, ortho))

''' Creates one of the vectors needed in the Gram-Schmidt process.

    Postcondition: In an attempt to minimize fractions, result is multiplied
        by the inner product of the last vector in [vectors] with itself.
    Postcondition: If applicable, vector elements are rounded to six decimal places.
    Postcondition: Vector elements are divided by the GCD of all elements.'''
def gram_vector(x, vectors):
    vector = x
    dnom = 0
    if len(vectors) > 0:
        for v in vectors:
            v_dot_v = inner_prod(v, v)
            weight = inner_prod(x, v) / v_dot_v
            dnom = v_dot_v
            vector = add_vectors(vector, scalar_multi(-weight, v))
        vector = scalar_multi(dnom, vector)
        vector = [round(a, 6) for a in vector]
        simp = abs(int(vector[0]))
        for v in vector:
            simp = math.gcd(simp, abs(int(v)))
        if simp != 0:
            vector = [a / simp for a in vector]
    return vector

''' Performs the Gram-Schmidt process for orthogonal bases.

    Precondition: [basis] is nonzero
    Precondition: Each vector in [basis] is the same length.'''
def gram_schmidt(basis):
    vectors = []
    for x in basis:
        vectors.append(gram_vector(x, vectors))
    return vectors

''' Constructs the Q matrix in the QR decomposition via the Gram-Schmidt process.

    Precondition: [matrix] is m x n with linearly independent columns.'''
def qr_fact_q(matrix):
    columns = []
    for col in range(0, len(matrix[0])):
        a = form_vector_from_rows(matrix, col)
        columns.append(a)
    columns = gram_schmidt(columns)
    q = []
    for col in columns:
        len_col = length(col)
        vect = scalar_multi(1/len_col, col)
        q.append(vect)
    return transpose(q)

''' Constructs the R matrix in the QR decomposition via the Gram-Schmidt process.

    Precondition: [matrix] is m x n with linearly independent columns.'''
def qr_fact_r(matrix):
    e_i = transpose(qr_fact_q(matrix))
    a_i = transpose(matrix)
    r = []
    for i in range(0, len(a_i)):
        r_row = []
        for j in range(0, i):
            r_row.append(0)
        for j in range(i, len(a_i)):
            r_row.append(inner_prod(e_i[i], a_i[j]))
        r.append(r_row)
    return r

''' Gives the least-squares solution of Ax = b using the QR decomposition method.

    Precondition: [matrix] is m x n with linearly independent columns.
    Precondition: [b] is of length m.
    Postcondition: Result is of length n.'''
def least_squares_qr(matrix, b):
    q = qr_fact_q(matrix)
    r = qr_fact_r(matrix)
    mult = matrix_multiplication(inverse(r), transpose(q))
    return matrix_vector_multi(mult, b)
