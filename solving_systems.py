'''Module that contains procedures for solving systems of linear equations.
    Contains:
        1) Reducing a matrix to echelon form
        2) Reducing a matrix to reduced echelon form
        3) Check for a consistent system
        4) Solve Systems of Linear Equations
            -Find the constant vector, free variables if applicable.
            -Convert a list to a vector string representation.
            -Return solution sets with vector representations as string lists.
        5) Checks for linear independence based on a homogeneous system.

    Author: Brendan Boyer
    Version: 1.0'''

'''Rearranges rows so that zeroes are moved to the bottom. '''
def rearrange_rows(matrix, m, n):
    if matrix [m][n] == 0:
        found = False
        row = m+1
        while not found and row < len(matrix):
            if matrix[row][n] != 0:
                temp = matrix[m]
                matrix[m] = matrix[row]
                matrix[row] = temp
                found = True
            row = row + 1
        if not found and n < len(matrix[0])-1:
            matrix = rearrange_rows(matrix, m, n+1)
    return matrix

''' Finds the index of the next pivot column.'''
def next_pivot(matrix_row, prev_col):
    found = False
    col = 0
    while not found and col < prev_col:
        if matrix_row[col] != 0:
            found = True
        else:
            col = col + 1
    return col

''' Determines if a row contains all zeroes.'''
def all_zeroes(row):
    zeroes = True
    col = 0
    while zeroes and col < len(row):
        if row[col] != 0:
            zeroes = False
        else:
            col += 1
    return zeroes

'''Reduces a matrix to echelon form, which has the following conditions:
    1) All nonzero rows are above zero rows.
    2) Each leading entry of a row is in a column to the right of the leading entry of the above row.
    3) All entries in a column below a leading entry are zeroes.

    Precondition: [m] = [n] = 0 when initially called. These are the pivot coordinates.
    Precondition: [coeff_matrix] must have the same number of elements in each row.
    Postcondition: entries that have a magnitude of less than 10^-10 are rounded to zero.'''
def echelon_form(coeff_matrix, m, n):
    if m >= len(coeff_matrix) or n >= len(coeff_matrix[0]) or len(coeff_matrix) == 1:
        return coeff_matrix
    else:
        rearrange_rows(coeff_matrix, m, n)
        for row in range(m, len(coeff_matrix)-1):
            add_row = coeff_matrix[m]
            if not all_zeroes(add_row):
                next_piv = next_pivot(add_row, len(add_row))
                add_row_pivot = add_row[next_piv]
                if add_row_pivot != 0:
                    change_elem = coeff_matrix[row+1][next_piv]
                    row_multiple = [(-change_elem / add_row_pivot) * x for x in add_row]
                    coeff_matrix[row+1] = [a+b for a,b in zip(coeff_matrix[row+1], row_multiple)]
                for col in range(0, len(coeff_matrix[0])):
                    if abs(round(coeff_matrix[row+1][col], 6)) < pow(10, -10):
                        coeff_matrix[row+1][col] = 0
        return echelon_form(coeff_matrix, m+1, n+1)

''' Iterates through the pivot positions of the echelon matrix. For use with [rref].'''
def pivot_clear(matrix, p_r, p_c):
    if p_r < 0 or p_c < 0:
        return matrix
    else:
        new_col = next_pivot(matrix[p_r], p_c)
        pivot_div = matrix[p_r][new_col]
        if pivot_div == 0:
            return pivot_clear(matrix, p_r-1, p_c)
        else:
            reduce_row = matrix[p_r]
            matrix[p_r] = [x / pivot_div for x in reduce_row]
            for row in range(0, p_r):
                add_row = matrix[row]
                change_elem = matrix[row][new_col]
                row_multiple = [-change_elem * x for x in matrix[p_r]]
                matrix[row] = [a+b for a,b in zip(add_row, row_multiple)]
            return pivot_clear(matrix, p_r - 1, new_col - 1)

'''Checks if a matrix is consistent. That is, it checks for whether the rightmost
    column of the matrix is not a pivot column. (i.e. a row like: [0, ..., 0, b]
    for some b in the real numbers.)

    Precondition: [matrix] is an augmented matrix.
    Note: Some matrices, when in echelon form, will appear to be inconsistent when
        they are in fact consistent due to rounding by the computer.'''
def consistent(matrix):
    ech_form = echelon_form(matrix, 0, 0)
    consistent = True
    row = 0
    while consistent and row < len(ech_form):
        check = ech_form[row]
        pivot = next_pivot(check, len(check))
        if pivot == len(check) - 1:
            consistent = False
        else:
            row += 1
    return consistent

''' Converts an inconsistent row to all zeroes. For use with [rref].

    Precondition: Each row in [matrix] has been row-reduced using echelon form.'''
def convert_incon(matrix):
    if len(matrix[0]) > 1:
        for row in range(0, len(matrix)):
            if not consistent([matrix[row]]):
                matrix[row][len(matrix[0]) - 1] = 0
    return matrix

''' Reduces a matrix to row-reduced echelon form (rref), which has the following conditions:
    1) All conditions present in the echelon form. (See [echelon_form] documentation.)
    2) The leading entry in each nonzero row is one (1).
    3) Each leading one is the only nonzero entry in its column.

    Precondition: [matrix] has m rows with n elements in each row.
    Precondition: [matrix] is consistent. If numerical rounding occurs in the
        formation of the echelon form, inconsistent rows will be converted to zeroes.'''
def rref(matrix):
    ech_form = echelon_form(matrix, 0, 0)
    ech_form = convert_incon(ech_form)
    pivot_row = len(ech_form) - 1
    pivot_col = len(ech_form[0]) - 1
    return pivot_clear(ech_form, pivot_row, pivot_col)

''' Returns the vector associated with some free variable.

    Precondition: [matrix] is in reduced echelon form.'''
def free_vector(matrix, col):
    set = []
    for row in range(0, len(matrix[0]) -  1):
        if row == col:
            set.append(1)
        elif row >= len(matrix):
            set.append(0)
        else:
            set.append(matrix[row][col])
    return set

''' Returns the constant vector associated with a matrix.

    Precondition: [matrix] is in reduced echelon form.'''
def const_vector(matrix):
    set = []
    row = 0
    col = next_pivot(matrix[0], len(matrix[0]) - 1)
    for i in range(0, col):
        set.append(0)
    while col < len(matrix[0]) - 1:
        if col >= len(matrix):
            set.append(0)
        elif matrix[row][col] != 1:
            next_col = next_pivot(matrix[col], len(matrix[0]) - 1)
            for i in range(col, next_col):
                set.append(0)
            if next_col < len(matrix[0]) - 1:
                set.append(matrix[col][len(matrix[0]) - 1])
            col = next_col
        else:
            set.append(matrix[row][len(matrix[0]) - 1])
        col += 1
        row += 1
    return set

''' Returns the indexes of all free columns in the matrix.

    Precondition: [reduced] is a matrix in reduced echelon form.'''
def free_indexes(reduced):
    next_piv = 0
    row = 0
    free_col = []
    while next_piv < len(reduced[0]) - 1 and row < len(reduced):
        next_col = next_pivot(reduced[row], len(reduced[0]))
        if next_col != next_piv:
            free_col.append(next_piv)
        else:
            row += 1
        next_piv += 1
    if row < len(reduced[0]) - 1:
        for i in range(next_piv, len(reduced[0]) - 1):
            free_col.append(i)
    return free_col

''' Converts a list of numbers to a vector expressed as "(a, b, ..., n).

    Precondition: [lst] must contain at least one element.'''
def convert_to_vector(lst):
    vect = '('
    for coord in lst:
        vect = vect + '{num:.3f}, '
        if coord == -0:
            vect = vect.format(num = abs(coord))
        else:
            vect = vect.format(num = coord)
    vect = vect[0:len(vect) - 2] + ')'
    return vect

''' Returns the solution set to the system of linear equations. There are three scenarios:
        1) Unique Solution: Returned as a string list: ["(a, b, ..., n)"] where
            (a, b, ..., n) is the coordinates of the solution. Values will contain
            three decimal places.
        2) Infinite Solutions: Returned as (1), but will represent the free variables
            as "(a, b, ..., j) xn" where n is the index of the variable plus one
            and (a, b, ..., j) is the direction of the free variable.
            Ex) ["(a, b, ..., xi)", "(c, d, ..., xj) x3"]
        3) No Solution: Single value string list: ["incon"]

    Precondition: [matrix] has the same number of elements in each row.'''
def solutions(matrix):
    is_cons = consistent(matrix)
    if is_cons:
        reduced = rref(matrix)
        free_col = free_indexes(reduced)
        solution_set = []
        constant = const_vector(reduced)
        constant = convert_to_vector(constant)
        solution_set.append(constant)
        if len(free_col) != 0:
            for index in free_col:
                vect = free_vector(reduced, index)
                vect_str = convert_to_vector(vect)
                vect_str = vect_str + ' x' + str(index + 1)
                solution_set.append(vect_str)
            return solution_set
        else:
            return solution_set
    else:
        return ['incon']

''' Checks whether a set of vectors is linearly independent. This method checks
    independence based on the homogeneous solution (Ax = 0). A return value of
    [True] denotes linear independence.

    Precondition: Each vector comes from the same vector space.'''
def lin_indep_homogeneous(vectors):
    matrix = []
    for row in range(0, len(vectors[0])):
        matrix_row = []
        for col in range(0, len(vectors)):
            matrix_row.append(vectors[col][row])
        matrix.append(matrix_row)
        matrix[row].append(0)
    reduced = rref(matrix)
    free_col = free_indexes(reduced)
    if free_col == []:
        return True
    else:
         return False
