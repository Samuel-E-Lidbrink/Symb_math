# Samuel Eriksson Lidbrink Grudat20
"""A symbolic math API.

This module is a simple library of tools to make symbolic computations and basic operations on algebraic functions.



Example:
    Example usage:

        >>> import symbol_math
        >>> f = symbol_math.Function("x^2+x+3x", "x")
        >>> print(f.simplify())
        x^2 + 4x
        >>> print(f.derivative())
        2x + 4

Todo:
    * Fix function 'simplify'
    * Add more functionality.
    * Improve documentation.
    * Error handling
"""


class Function(object):
    """A class for storing functions.

    Todo:
        * Add functionality for multivariable functions
        * Implement more operators such as sub, div.
        * Fix methods 'evaluate', 'derivative'
        * Error handling

    """
    def __init__(self, expression, variable):
        """
        Args:
            expression (string): The functions expression. Must be an algebraic expression
            variable (string): The function variable
        Raises:
            TypeError: expression not recognisable as algebraic expression
        """
        self.expression = expression
        self.variable = variable

    def simplify(self):
        """Simplifies expression to simplest form."""
        self.expression = simplify(self.expression)

    def evaluate(self, value):
        """Evaluates the function at specified value
        Args:
            value (float): Value where function should be evaluated
        Returns:
            float: function evaluation
        """
        pass

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
        return Function(simplify(self.expression + other.expression), self.variable)

    def __str__(self):
        """String representation.
        Overloads str operator with function expression"""
        return self.expression

    def derivative(self):
        """Calculate derivative of function
        Returns:
            object: the function derivative
        Raises:
             SystemError: Cannot compute derivative internally
         """
        pass

    def finite_integration(self, lower_bound, upper_bound):
        """Calculate derivative of function
        Args:
            lower_bound (float): lower integration bound
            upper_bound (float): upper integration bound
        Returns:
            float: integral of self over lower_bound to upper_bound
         """
        pass


def simplify(expression):
    """Simplifies expression to simplest form.
    Args:
        expression (string): The expression that should be simplified.
    Returns:
        string: simplified expression.
    """
    pass
