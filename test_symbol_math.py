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
    pass


class TestDerivative(unittest.TestCase):
    pass


class TestEvaluate(unittest.TestCase):
    pass


class TestReplaceVar(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()