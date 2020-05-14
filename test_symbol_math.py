import symbol_math
import unittest


class TestFunction(unittest.TestCase):
    def setUp(self):
        self.f = symbol_math.Function("x^2+x+3x", "x")
        self.f.simplify()
        self.g = symbol_math.Function("exp(y)", "y")
        for bad_func in [["", "x"], ["1+y", "x"], ["x+y", "x+y"], ["1*/3", "x"], ["x^3^3", "x"]]:
            with self.assertRaises(TypeError):
                self.bad_func = symbol_math.Function(bad_func[0], bad_func[1])
        self.h = symbol_math.Function("3sinx2", "x")

    def test_str(self):
        self.assertEqual(str(self.f), "x^2 + 4*x")
        self.assertEqual(str(self.g), "exp(y)")
        self.assertEqual(str(self.h), "3*sin(x^2)")

    def test_add(self):
        with self.assertRaises(TypeError):
            str(self.f + self.g)
        self.g.change_variable("x")
        self.assertEqual(str(self.f + self.g), "x^2 + 4*x + exp(x)")

    def test_finite_integration(self):
        self.assertEqual(self.g.finite_integration(0, 1, tol=1e-5), 1.7182818298909466)

    def test_evaluate(self):
        self.assertEqual(self.f.evaluate(3), 21.0)


class TestSimplify(unittest.TestCase):
    def test_binary(self):
        for [expr, result] in [["1+2", "3"], ["x+7", "7 + x"], ["5x-2+4x-7x2", "-2 - 7*x^2 + 9*x"]]:
            self.assertEqual(result, symbol_math.simplify(expr, "x"))

    def test_arithmetic(self):
        for [expr, result] in [["1*0+3*2-6", "0"], ["1/3x2", "0.3333333333333333*x^2"], ["7*4+23-2", "49"]]:
            self.assertEqual(result, symbol_math.simplify(expr, "x"))

    def test_parenthesis(self):
        for [expr, result] in [["2-x+3(x+1)", "5 + 2*x"], ["(x-1)*3*(x-1)/(3)", "(-1 + x)^2"],
                               ["((3-7x2)/(1+x)^3*(1+x)*2-7^(3x-4+2*2))^2", "(2/(1 + x)^2*(3 - 7*x^2) - 7^(3*x))^2"]]:
            self.assertEqual(result, symbol_math.simplify(expr, "x"))

    def test_operators(self):
        for [expr, result] in [["sinxcosx+cosxsinx", "2*cos(x)*sin(x)"],
                               ["log(exp((10+x)^2))/(sin(asin(11-1+x)))-10(sinx-sinxcosh(acosh(x))/x)-10", "x"],
                               ["(1-x(1-2))/(1+x)", "1"]]:
            self.assertEqual(result, symbol_math.simplify(expr, "x"))


class TestDerivative(unittest.TestCase):
    def test_simple(self):
        for [expr, result] in [["435 +3 -2", "0"], ["x", "1"], ["x^10", "10*x^9"]]:
            self.assertEqual(result, symbol_math.derivative(expr, "x"))

    def test_mult(self):
        for [expr, result] in [["x*x", "2*x"], ["(x^2-3)*(1-x)", "-1 + 2*x"], ["x^10/(x+3*x*(x-1))", "10*x^9"]]:
            self.assertEqual(result, symbol_math.derivative(expr, "x"))

    def test_parenthesis(self):
        for [expr, result] in [["2-x+3(x+1)", "5 + 2*x"], ["(x-1)*3*(x-1)/(3)", "(-1 + x)^2"],
                               ["((3-7x2)/(1+x)^3*(1+x)*2-7^(3x-4+2*2))^2", "(2/(1 + x)^2*(3 - 7*x^2) - 7^(3*x))^2"]]:
            self.assertEqual(result, symbol_math.simplify(expr, "x"))

    def test_operators(self):
        for [expr, result] in [["sinxcosx+cosxsinx", "2*cos(x)*sin(x)"],
                               ["log(exp((10+x)^2))/(sin(asin(11-1+x)))-10(sinx-sinxcosh(acosh(x))/x)-10", "x"],
                               ["(1-x(1-2))/(1+x)", "1"]]:
            self.assertEqual(result, symbol_math.simplify(expr, "x"))



class TestEvaluate(unittest.TestCase):
    pass


class TestReplaceVar(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
