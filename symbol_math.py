# Samuel Eriksson Lidbrink Grudat20
"""A symbolic math API.

This module is a simple library of tools to make symbolic computations and basic operations on algebraic functions.



Example:
    Example usage:

        >>> import symbol_math
        >>> f = symbol_math.Function("x^2+x+3x", "x")
        >>> print(f.simplify())
        x^2 + 4*x
        >>> print(f.derivative())
        4 + 2*x
        >>> g = Function("exp(y)", "y")
        >>> print(f.change_variable("y"))
        y^2 + 4*y
        >>> print(f+g)
        y^2 + 4*y + exp(y)
        >>> f.evaluate(3)
        21.0
        >>> g.finite_integration(0, 1, tol=1e-5)
        1.7182818298909466
"""
import math


class Function(object):
    """A class for storing functions.


    """

    def __init__(self, expression, variable):
        """
        Args:
            expression (string): The functions expression. Must be an algebraic expression
            variable (string): The function variable
        Raises:
            TypeError: expression not recognisable as algebraic expression or variable name equal to protected function
        """
        _check_expression(expression, variable)
        self._expression = _interp_expr(replace_var(expression, variable, "X"), "X")
        self.variable = variable

    def simplify(self):
        """Simplifies expression to simplest form.
        Returns:
            string: new expression for function
            """
        self._expression = _simp_helper(self._expression)
        return self

    def evaluate(self, value):
        """Evaluates the function at specified value
        Args:
            value (float, int): Value where function should be evaluated
        Returns:
            float: function evaluation
        """
        return evaluate(str(self), self.variable, value)

    def change_variable(self, variable):
        """Changes the current function variable to the given variable.
        Args:
            variable (string): New variable for function
        Returns:
            string: New function expression
        """
        self.variable = variable
        return self

    def __add__(self, other):
        """Addition operator for two functions of the same variables.
        Overloads + operator as addition defined for functions.
        Args:
            other (Function): Other function to be added
        Returns:
            object: the function that is the sum of self and other
        Raises:
             TypeError: If other.variable != self.variable
         """
        if other.variable != self.variable:
            raise TypeError("Functions must have the same variable")
        return Function(str(self) + "+" + str(other), self.variable)

    def __str__(self):
        """String representation.
        Overloads str operator with function expression"""
        return _fix_out(self._expression, self.variable)

    def derivative(self):
        """Calculate derivative of function
        Returns:
            object: the function derivative
        Raises:
             SystemError: Cannot compute derivative internally
         """
        der = Function("1", self.variable)
        der._expression = _der_helper(self._expression)
        return der

    def finite_integration(self, lower_bound, upper_bound, tol=0.01):
        """Calculate derivative of function
        Args:
            lower_bound (float): lower integration bound
            upper_bound (float): upper integration bound
            tol (float): error tolerance for calculation. Must be non-negative
        Returns:
            float: integral of self over lower_bound to upper_bound
        Raises:
            TypeError: If either lower_bound, upper_bound or tol is not of type float
            ValueError: If tol < 0
         """
        for arg in [lower_bound, upper_bound, tol]:
            if not isinstance(arg, (float, int)):
                raise TypeError("Arguments must be floats or ints")
        if tol < 0:
            raise ValueError("tol must be positive, not " + str(tol))
        val, prev_val = 0, tol + 1
        exp = 1
        while abs(prev_val-val) > tol:
            prev_val = val
            exp += 1
            n = 10 ** exp
            part_sum = (self.evaluate(lower_bound) + self.evaluate(upper_bound))/2/n
            for k in range(1, n):
                point = lower_bound + k/n* (upper_bound-lower_bound)
                part_sum += self.evaluate(point) * 1/n
            val = part_sum
        return val


COMMON_OPERATORS = ["asinh", "acosh", "atanh", "sinh", "cosh", "tanh", "asin", "acos", "atan", "sin", "tan", "cos",
                    "log", "exp"]
_OPERATOR_INVERSE = {"sinh": "asinh", "cosh": "acosh", "tanh": "atanh", "asinh": "sinh", "acosh": "cosh",
                     "atanh": "tanh", "sin": "asin", "cos": "acos", "tan": "atan", "asin": "sin", "atan": "tan",
                     "acos": "cos", "log": "exp", "exp": "log"}
_OPERATOR_DERIVATIVE = {"sinh": "cosh", "cosh": "sinh", "tanh": [1, "/", ["cosh", ["X"]], "^", 2],
                        "asinh": [1, "/", ["X", "^", 2, "+", 1], "^", 0.5],
                        "acosh": [1, "/", ["X", "^", 2, "-", 1], "^", 0.5],
                     "atanh": [1, "/", [1, "-", "X", "^", 2]], "sin": "cos", "cos": "sin",
                        "tan": [1, "/", ["cos", ["X"]], "^", 2], "asin": [1, "/", [1, "-", "X", "^", 2], "^", 0.5],
                        "atan": [1, "/", ["X", "^", 2, "+", 1]], "acos": ["-", 1, "/", [1, "-", "X", "^", 2], "^", 0.5],
                        "log": [1, "/", "X"], "exp": "exp"}


def simplify(expression, variable):
    """Simplifies expression to simplest form.
    Args:
        expression (string): The expression that should be simplified.
        variable (string): Used variable in expression
    Returns:
        string: simplified expression.
    Raises:
        TypeError: expression not recognisable as algebraic expression or variable name equal to protected function
    """
    _check_expression(expression, variable)
    expr_list = _simp_helper(_interp_expr(replace_var(expression, variable, "X"), "X"))
    return _fix_out(expr_list, variable)


def derivative(expression, variable):
    """Takes the derivative of the expression.
    Args:
        expression (string): Original expression.
        variable (string): Variable used in expression
    Returns:
        string: Derivative of expression.
    Raises:
        TypeError: expression not recognisable as algebraic expression or variable name equal to protected function
    """
    _check_expression(expression, variable)
    expr_list = _der_helper(_interp_expr(replace_var(expression, variable, "X"), "X"))
    return _fix_out(expr_list, variable)


def _fix_out(expr_list, variable):
    out = ""
    for thing in expr_list:
        if thing == "+" or thing == "-":
            out += " " + thing + " "
        else:
            out += str(thing)
    return replace_var(out, "X", variable)


def _der_helper(expression_list):
    """A help function for derivation."""

    def der_grouped(group):
        if not var_in(group):
            return [0]
        if "+" not in group and "-" not in group:
            return der_mult(group)
        for index in range(0, len(group)):
            item = group[index]
            if item == "+" or item == "-":
                if index > 0:
                    return der_grouped(group[:index]) + [item] + der_grouped(group[index+1:])
                else:
                    return [item] + der_grouped(group[index+1:])

    def der_mult(group):
        if not var_in(group):
            return [0]
        if "*" not in group:
            return der_div(group)
        for index in range(0, len(group)):
            if group[index] == "*":
                return group[:index] + ["*"] + der_mult(group[index+1:]) + ["+"] + group[index+1:] +\
                       ["*"] + der_div(group[:index])

    def der_div(group):
        if not var_in(group):
            return [0]
        if "/" not in group:
            return der_exp(group)
        for index in range(len(group)-1, 0, -1):
            if group[index] == "/":
                f, g = group[:index], group[index+1:]
                return [der_div(f) + ["*"] + g + ["-"] + f + ["*"] + der_exp(g), "/"] + g + ["^", 2]

    def der_exp(group):
        if not var_in(group):
            return [0]
        if "^" not in group:
            return der_op(group)
        for index in range(0, len(group)):
            if group[index] == "^":
                base, exp = group[:index], group[index+1:]
                if not var_in(exp):
                    return exp + ["*"] + base + ["^", exp + ["-", 1]]
                elif not var_in(base):
                    return [["log", base], "*", der_par(exp)] + ["*"] + base + ["^"] + exp
                else:
                    return der_mult([["log", base], "*", exp]) + ["*"] + base + ["^"] + exp

    def der_op(group):
        if not var_in(group):
            return [0]
        if len(group) == 2 and isinstance(group[0], str) and group[0] in COMMON_OPERATORS:
            op_der = _OPERATOR_DERIVATIVE[group[0]]
            if isinstance(op_der, str):
                return [[op_der, group[1]], "*", der_par([group[1]])]
            else:
                return [[x_replacer(op_der, group[1])], "*", der_par([group[1]])]
        else:
            return der_par(group)

    def der_par(group):
        if not var_in(group):
            return [0]
        if group == "X" or group == ["X"]:
            return [1]
        elif len(group) == 1 and isinstance(group[0], list):
            return der_grouped(group[0])
        else:
            raise SystemError("Derivation went wrong :(.")

    def var_in(group):
        for item in group:
            if isinstance(item, list) and var_in(item):
                return True
            elif item == "X":
                return True
        return False

    def x_replacer(group, instead):
        if not isinstance(group, list) or not var_in(group):
            return group
        for index in range(0, len(group)):
            item = group[index]
            if item == "X":
                group[index] = instead
            else:
                group[index] = x_replacer(group[index], instead)
        return group

    grouped_list = _group(_basic_fix(expression_list))
    der_group = der_grouped(grouped_list)
    new_list, _ = _sort_and_ungroup(der_group)
    return _simp_helper(new_list)


def _simp_helper(expression_list, mem=None):
    """A help function for simplify."""
    # memorizes previous result to avoid loops
    if mem is None:
        mem = []
    mem.append(expression_list.copy())

    new_list, _ = _sort_and_ungroup(_group(_basic_fix(expression_list)))
    if len(new_list) == 0:
        return [0]
    fixed_list = _basic_fix(new_list)
    if fixed_list in mem:
        return fixed_list
    else:
        return _simp_helper(fixed_list, mem)


def _basic_fix(list_to_fix):
    """does basic simplifications"""
    prev_expr = []
    while list_to_fix != prev_expr:
        prev_expr = list_to_fix.copy()
        prev_thing = "{"
        par = {0: [""]}
        par_ok = {0: False}
        par_mult = {}
        par_count = 0
        for index in range(0, len(list_to_fix)):
            thing = list_to_fix[index]
            if index + 1 < len(list_to_fix):
                next_thing = list_to_fix[index + 1]
            else:
                next_thing = "}"

            if _is_int(thing):
                list_to_fix[index] = int(float(thing))
            # Remove redundant expressions
            if (isinstance(prev_thing, str))\
                    and ((thing == 0 and prev_thing in "+-{(" and isinstance(next_thing, str) and next_thing in "+-")
                         or (thing == 1 and prev_thing == "^") or (thing == 1 and prev_thing in "*/")):
                if index - 1 > 0:
                    list_to_fix = list_to_fix[:index - 1] + list_to_fix[index + 1:]
                else:
                    list_to_fix = list_to_fix[index + 1:]
                break
            elif thing == "+" and (isinstance(prev_thing, str) and prev_thing in "*/^"):
                list_to_fix = list_to_fix[:index] + list_to_fix[index + 1:]
                break
            elif thing == "-" and (isinstance(prev_thing, str) and prev_thing in "*/^"):
                list_to_fix = list_to_fix[:index] + [-float(next_thing)] + list_to_fix[index + 2:]
                break
            elif prev_thing == "(" and next_thing == ")":
                if index-2 < 0 or not isinstance(list_to_fix[index-2], str) or (list_to_fix[index-2] not in
                                                                                COMMON_OPERATORS):
                    if index - 1 > 0:
                        list_to_fix = list_to_fix[:index-1] + [thing] + list_to_fix[index+1:]
                    else:
                        list_to_fix = [thing] + list_to_fix[index+1:]
                    break
            elif thing == "+" and (prev_thing == "{" or prev_thing == "("):
                if index > 0:
                    list_to_fix = list_to_fix[:index] + list_to_fix[index+1:]
                else:
                    list_to_fix = list_to_fix[index+1:]
                break

            # does basic calculation
            if not isinstance(thing, list):
                if thing == "/" and _is_int(prev_thing) and _is_int(next_thing):
                    divisor = math.gcd(int(prev_thing), int(next_thing))
                    if divisor not in [0, 1]:
                        list_to_fix[index-1] = int(prev_thing)/divisor
                        list_to_fix[index+1] = int(next_thing)/divisor
                        break

            # Remove redundant parenthesis
            for outside in range(par_count - 1, 0, -1):
                if par_ok[outside]:
                    par[outside].append(thing)
            if thing == "(":
                par_count += 1
                if prev_thing == "+" or prev_thing == "{":
                    par[par_count] = ["+"]
                    par_ok[par_count] = True
                elif prev_thing == "-":
                    par[par_count] = ["-"]
                    par_ok[par_count] = False
                elif prev_thing == "*" and _is_float(list_to_fix[index-2])\
                        and (index - 3 <= 0 or list_to_fix[index - 3] == "+" or list_to_fix[index-3] == "-"):
                    par_mult[par_count] = list_to_fix[index - 2]
                    if index-3 <= 0 or list_to_fix[index-3] == "+":
                        par[par_count] = ["+"]
                    else:
                        par[par_count] = ["-"]
                    par_ok[par_count] = True
                else:
                    par_ok[par_count] = False
            elif thing == ")":
                if par_ok[par_count] and (isinstance(next_thing, str) and next_thing in "+-}"):
                    content = par[par_count]
                    par_len = len(content)
                    if content[1] == "+" or content[1] == "-":
                        content = content[1:]
                    to_mult = float(par_mult.setdefault(par_count, 1))
                    if to_mult != 1:
                        multiplied = []
                        for item in content:
                            if item == "+" or item == "-":
                                multiplied.extend([item, str(to_mult), "*"])
                            else:
                                multiplied.append(item)
                        content = multiplied
                        par_len += 2
                    if index - par_len -1 > 0:
                        list_to_fix = list_to_fix[:index - par_len-1] + content.copy() + list_to_fix[index + 1:]
                    else:
                        list_to_fix = content.copy() + list_to_fix[index + 1:]
                    break
                par_count -= 1
            elif par_ok[par_count]:
                if par[par_count][0] == "-":
                    if thing == "+":
                        to_add = "-"
                    elif thing == "-":
                        to_add = "+"
                    else:
                        to_add = thing
                else:
                    to_add = thing
                par[par_count].append(to_add)

            prev_thing = thing
    return list_to_fix


def _group(list_to_group):
    """groups similar object together in preferred order"""
    first_level = []
    is_first = True
    is_op = False
    op_inv = False
    op = ""
    second_level = []
    par_count = 0
    for item_index in range(0, len(list_to_group)):
        item = list_to_group[item_index]
        if isinstance(item, str) and item in COMMON_OPERATORS and not is_op and is_first:
            is_op = True
            op = item
            if item in _OPERATOR_INVERSE.keys():
                if list_to_group[item_index + 1] == "(" and list_to_group[item_index + 2] == _OPERATOR_INVERSE[item]:
                    op_inv = True
        elif item == "(":
            if not is_first:
                second_level.append(item)
            par_count += 1
            is_first = False
        elif item == ")":
            par_count -= 1
            if op_inv and par_count == 1:
                if list_to_group[item_index + 1] != ")":
                    op_inv = False
            if par_count == 0:
                if is_op:
                    is_op = False
                    if op_inv:
                        first_level.append(second_level[2:-1])
                    elif second_level[0] == "(" and second_level[-1] == ")":
                        first_level.append([op, _group(second_level[1:-1])])
                    else:
                        first_level.append([op, _group(second_level)])
                else:
                    first_level.append(_group(second_level))
                is_first = True
                second_level = []
            else:
                second_level.append(item)
        elif is_first:
            first_level.append(item)
        else:
            second_level.append(item)
    return first_level


def _sort_and_ungroup(list_to_fix):
    while len(list_to_fix) == 1 and isinstance(list_to_fix[0], list):
        list_to_fix = list_to_fix[0]
    res = []
    list_to_fix, req_par = _sort_group(list_to_fix.copy())
    for item_index in range(0, len(list_to_fix)):
        item = list_to_fix[item_index]
        if isinstance(item, list):
            nested_list, nest_par = _sort_and_ungroup(item)
            if (item_index + 1 < len(list_to_fix) and list_to_fix[item_index + 1] == "^") or\
                    (item_index > 0 and isinstance(list_to_fix[item_index - 1], str)
                     and list_to_fix[item_index - 1] in COMMON_OPERATORS):
                nest_par = True
            if res[-1] == "^":
                nest_par = True
            if nest_par:
                res.append("(")
            for nested_item in nested_list:
                res.append(nested_item)
            if nest_par:
                res.append(")")
        else:
            res.append(item)
    return res, req_par


def _sort_group(list_to_sort):
    req_par = False
    if len(list_to_sort) == 0 or isinstance(list_to_sort[0], str) and list_to_sort[0] in COMMON_OPERATORS:
        return list_to_sort, req_par
    types = dict()
    cur = []
    prev_sign = 1
    if list_to_sort[0] == "-":
        prev_sign = -1
    for item_index in range(0, len(list_to_sort)+1):
        if item_index < len(list_to_sort):
            item = list_to_sort[item_index]
        else:
            item = "}"
        if (item == "+" or item == "-" or item == "}") and\
                (len(cur) > 0 and (item_index - 1 >= 0 and list_to_sort[item_index - 1] != "(")):
            fixed_part = _fix_mult(cur)
            if len(fixed_part) == 1:
                types["1"] = [[1], types.setdefault("1", [[1], 0])[1] + prev_sign * fixed_part[0]]
            else:
                types[str(fixed_part[2:])] = [fixed_part[2:], types.setdefault(str(fixed_part[2:]), ["", 0])[1] +
                                              prev_sign * float(fixed_part[0])]
            if item == "+":
                prev_sign = 1
            elif item == "-":
                prev_sign = -1
            cur = []
        else:
            cur.append(item)
    res = []
    keys = sorted(types.keys())
    if len(keys) > 1:
        req_par = True
    for dic_key in keys:
        if types[dic_key][1] == 0:
            continue
        elif types[dic_key][1] == 1:
            res.extend(["+"] + types[dic_key][0])
        elif types[dic_key][1] == -1:
            res.extend(["-"] + types[dic_key][0])
        elif types[dic_key][1] > 0:
            res.extend(["+", types[dic_key][1], "*"] + types[dic_key][0])
        elif types[dic_key][1] < 0:
            res.extend(["-", -types[dic_key][1], "*"] + types[dic_key][0])
    return res, req_par


def _fix_mult(list_to_fix):
    """Returns an expression without +- simplified with a leading coefficient (and perhaps *[...])"""
    prev = "{"
    const = 1
    var_pow = 0
    exp = []
    par = []
    exp_to_check = []
    par_to_check = {}
    for item_index in range(0, len(list_to_fix)):
        item = list_to_fix[item_index]
        if item_index + 1 == len(list_to_fix):
            next_i = "}"
        else:
            next_i = list_to_fix[item_index + 1]
        if (isinstance(item, str) and item in "*/^") or (isinstance(prev, str) and prev == "^"):
            prev = item
            continue
        if prev == "*" or prev == "{":
            sign = 1
        elif prev == "/":
            sign = -1
        if item == "X":
            if next_i == "^":
                x_pow = list_to_fix[item_index + 2]
                if x_pow == "X" or isinstance(x_pow, list):
                    if sign == 1:
                        if len(exp) > 0:
                            exp.append("*")
                    elif sign == -1:
                        if len(exp) == 0:
                            exp.append(1)
                        exp.append("/")
                    exp.extend(["X", "^", x_pow])
                    prev = item
                    continue
            else:
                x_pow = 1
            if isinstance(x_pow, int) or x_pow.isdigit():
                var_pow += sign * int(float(x_pow))
            else:
                var_pow += sign * float(x_pow)
        elif _is_float(item):
            if next_i == "^":
                num_pow = list_to_fix[item_index + 2]
                if num_pow == "X":
                    exp_to_check.append(item)
                    prev = item
                    continue
                elif isinstance(num_pow, list):
                    if sign == 1:
                        if len(exp) > 0:
                            exp.append("*")
                    elif sign == -1:
                        if len(exp) == 0:
                            exp.append(1)
                        exp.append("/")
                    exp.extend([item, "^", num_pow])
                    prev = item
                    continue
            else:
                num_pow = 1
            if sign == 1:
                const *= float(item) ** float(num_pow)
            elif sign == -1:
                const /= float(item) ** float(num_pow)
        elif isinstance(item, list) and prev != "^":

            if next_i == "^" and not _is_float(list_to_fix[item_index + 2]):
                if sign == 1:
                    if len(par) > 0:
                        par.append("*")
                elif sign == -1:
                    if len(par) == 0:
                        par.append(1)
                    par.append("/")
                if next_i == "^":
                    par.extend([item, "^", list_to_fix[item_index + 2]])
            else:
                if next_i == "^":
                    extra = float(list_to_fix[item_index + 2])
                else:
                    extra = 1
                par_to_check[str(item)] = [item, par_to_check.setdefault(str(item), [[], 0])[1] + sign * extra]
        prev = item
    par_keys = sorted(par_to_check.keys())
    for dic_key in par_keys:
        if par_to_check[dic_key][1] == 0:
            continue
        elif par_to_check[dic_key][1] == 1:
            if len(par) > 0:
                par.append("*")
            par.append(par_to_check[dic_key][0])
        elif par_to_check[dic_key][1] == -1:
            if len(par) == 0:
                par.append(1)
            par.extend(["/"] + [par_to_check[dic_key][0]])
        elif par_to_check[dic_key][1] > 0:
            if len(par) > 0:
                par.append("*")
            par.extend([par_to_check[dic_key][0]] + ["^"] + [str(par_to_check[dic_key][1])])
        else:
            if len(par) == 0:
                par.append(1)
            par.extend(["/"] + [par_to_check[dic_key][0]] + ["^"] + [str(-par_to_check[dic_key][1])])

    # simplifies exponents
    prev_check = []
    while prev_check != exp_to_check:
        prev_check = exp_to_check.copy()
        exp_to_check = sorted(exp_to_check)
        for dig_index in range(0, len(exp_to_check) - 1):
            cur_dig = exp_to_check[dig_index]
            next_dig = exp_to_check[dig_index + 1]
            if cur_dig == next_dig:
                if dig_index > 0:
                    exp_to_check = exp_to_check[:dig_index] + exp_to_check[dig_index + 2:] + [str(float(cur_dig) *
                                                                                                  float(next_dig))]
                    break
                else:
                    exp_to_check = exp_to_check[dig_index + 2:] + [str(float(cur_dig) * float(next_dig))]
                    break
    if len(exp_to_check) == 1:
        if len(exp) == 0:
            exp = [exp_to_check[0], "^", "X"]
        else:
            exp.extend(["*", exp_to_check[0], "^", "X"])
    elif len(exp_to_check) > 1:
        if len(exp) > 0:
            exp.append("*")
        for dig_index in range(0, len(exp_to_check)):
            exp.extend([exp_to_check[dig_index], "^", "X"])
            if dig_index < len(exp_to_check) - 1:
                exp.append("*")

    if const == 0:
        return [0]
    if len(exp) == 0:
        end = par
    elif len(par) == 0:
        end = exp
    else:
        end = exp + ["*"] + par
    if var_pow != 0:
        if len(end) > 0:
            end = ["*"] + end
        if not var_pow == 1:
            end = ["^", var_pow] + end
        end = ["X"] + end
    if len(end) > 0:
        return [const, "*"] + end
    else:
        return [const]


def _is_int(string):
    if not isinstance(string, (str, int, float)):
        return False
    dot = False
    for char in str(string):
        if char not in "1234567890.":
            return False
        if char == ".":
            dot = True
        elif dot:
            if char != "0":
                return False
    return True


def _is_float(string):
    if not isinstance(string, (str, int, float)):
        return False
    try:
        float(string)
        return True
    except ValueError:
        return False


def replace_var(expression, old_var, new_var):
    """Replaces old variable in expression with a new one, without changing built in functions.
    Args:
        expression (string): The expression that should be simplified.
        old_var (string): Old variable in expression
        new_var (string): New variable to be used
    Returns:
        string: updated expression
    """
    return _replace_helper(expression, old_var, new_var, 0)


def _replace_helper(expression, old_var, new_var, index):
    """Help function for replace_var."""
    if index == len(COMMON_OPERATORS):
        return expression.replace(old_var, new_var)
    else:
        sub_list = expression.split(COMMON_OPERATORS[index])
        for place in range(0, len(sub_list)):
            sub_list[place] = _replace_helper(sub_list[place], old_var, new_var, index + 1)
        return COMMON_OPERATORS[index].join(sub_list)


def _interp_expr(expression, variable, value=None):
    interp_expr = []
    curr = ""  # used to store numbers and common functions in one place
    prev_char = "{"  # signifies "character" before the first one
    add_par = 0
    to_check = replace_var(expression, variable, "X").replace("[", "(").replace("]", ")")
    for index in range(0, len(to_check)):
        char = to_check[index]
        if char == "(" and prev_char in "0123456789X)" or char in "0123456789X" and prev_char == ")":
            while add_par > 0:
                interp_expr.append(")")
                add_par -= 1
            interp_expr.append("*")
        if char in "".join(COMMON_OPERATORS):
            if prev_char not in "".join(COMMON_OPERATORS).lower() + "{":
                if prev_char in "0123456789X)":
                    while add_par > 0:
                        interp_expr.append(")")
                        add_par -= 1
                    interp_expr.append("*")
            curr += char
            if curr in COMMON_OPERATORS and (index + 1 == len(to_check) or (curr + to_check[index + 1]) not in
                                             COMMON_OPERATORS):
                interp_expr.append(curr)
                if not index + 1 == len(to_check) and not to_check[index + 1] == "(":
                    interp_expr.append("(")
                    add_par += 1
                curr = ""
                prev_char = "}"  # signifies a common function
                continue
        elif char in "0123456789.":
            if prev_char == "X":
                interp_expr.append("^")
            curr += char
            if index + 1 == len(to_check) or not to_check[index + 1] in "0123456789.":
                if _is_int(curr):
                    interp_expr.append(int(float(curr)))
                else:
                    interp_expr.append(float(curr))
                curr = ""
        elif char == "X":
            if prev_char in "1234567890X":
                interp_expr.append("*")
            if value is not None:
                interp_expr.append(str(value))
            else:
                interp_expr.append(variable)
        elif char in "+-*/()^":
            if not char == "^":
                while add_par > 0:
                    interp_expr.append(")")
                    add_par -= 1
            interp_expr.append(char)
            curr = ""
        prev_char = char
    while add_par > 0:
        interp_expr.append(")")
        add_par -= 1
    return interp_expr


def evaluate(expression, variable, value):
    """Evaluates the expression with given variable at value.
    Args:
        expression (string): The expression that should be simplified.
        variable (string): Used variable in expression
        value (float, int): Value where function should be evaluated
    Returns:
        float: evaluation of function.
    Raises:
        TypeError: expression not recognisable as algebraic expression or variable name equal to protected function"""
    if not isinstance(value, (float, int)):
        raise TypeError("Input value must be float or int")
    _check_expression(expression, variable)
    interp = _interp_expr(expression, variable, value)
    for index in range(0, len(interp)):
        thing = str(interp[index])
        if thing == "^":
            interp[index] = "**"
        elif thing in COMMON_OPERATORS:
            interp[index] = "math." + thing
        else:
            interp[index] = thing
    try:
        return float(eval("".join(interp)))
    except ZeroDivisionError:
        raise ZeroDivisionError("float division by zero for expression: " + expression + " at " + variable + " = " +
                                str(value))


def _check_expression(expr, variable):
    """Checks whether the expr is a valid string expression of an arithmetic expression"""
    if not isinstance(expr, str):
        raise TypeError("Expression must be a string")
    if expr == "":
        raise TypeError("Expression must be non empty")
    if not isinstance(variable, str):
        raise TypeError("Variable must be a string")
    for char in variable:
        if char in ".0123456789()[]+-*^/{}\\\\":
            raise TypeError("Invalid variable " + variable)
    for thing in ["{", "}"]:
        if thing in expr:
            raise TypeError("Invalid character " + thing + " in expression " + expr)
    simple_expr = expr
    for op in COMMON_OPERATORS:
        if op == variable.lower():
            raise TypeError("Invalid variable name " + variable + ". Protected operator.")
        simple_expr = simple_expr.replace(op, "}")  # } signifies a common operator
    simple_expr = simple_expr.replace(variable, "x").replace(" ", "")
    parenthesis_list = []
    prev_char = "{"  # { signifies the "character" before the first character
    dot_allowed = True
    for char_index in range(0, len(simple_expr)):
        char = simple_expr[char_index]
        if char_index + 1 < len(simple_expr):
            next_char = simple_expr[char_index+1]
        else:
            next_char = "{"
        if char not in ".0123456789()[]+-*^/x{}":
            raise TypeError("Invalid character: " + char + " in expression " + expr)
        if char in "([":
            parenthesis_list.append(char)
        elif char in ")]":
            if len(parenthesis_list) == 0:
                raise TypeError("Non matching parenthesis: " + char + " in expression " + expr)
            if char == ")":
                if not parenthesis_list.pop() == "(":
                    raise TypeError("Non matching parenthesis: [ ... )" + " in expression " + expr)
            else:
                if not parenthesis_list.pop() == "[":
                    raise TypeError("Non matching parenthesis: ( ... ]" + " in expression " + expr)
            if prev_char in ".+-*/^}":
                raise TypeError("Invalid syntax: " + prev_char.replace("}", "...") + char + " in expression " +
                                expr)
            elif prev_char in "([":
                raise TypeError("Empty parenthesis: " + char + prev_char + " in expression " + expr)
        elif char in "^*/" and prev_char == "{":
            raise TypeError("Invalid starting operator: " + char + " in expression " + expr)
        elif char in "+-*/^" and prev_char in "+-*/^[}.":
            raise TypeError("Invalid operator usage: " + prev_char.replace("}", "...") + char + " in expression "
                            + expr)
        elif char in "*/" and prev_char == "(":
            raise TypeError("Invalid operator usage: " + prev_char + char + " in expression " + expr)
        elif char == ".":
            if not dot_allowed:
                raise TypeError("Two decimal points in one number in expression: " + expr)
            else:
                dot_allowed = False
        elif prev_char == "^" and (next_char == "^" or char == "x" and next_char in "1234567890."):
            raise TypeError("Unclear use of exponent: " + prev_char + char + next_char + " in: " + expr)
        if char in "+-*^/()[]x}":
            dot_allowed = True
        prev_char = char
    if simple_expr[-1] in "+-*/^}.":
        raise TypeError("Invalid ending operator in expression " + expr)
    if not len(parenthesis_list) == 0:
        raise TypeError("Missing " + str(len(parenthesis_list)) + " ending parenthesis" + " in expression " + expr)


