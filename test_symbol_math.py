import symbol_math
import unittest


class TestFunction(unittest.TestCase):
    def setUp(self):
        self.f = symbol_math.Function("x^2+x+3x", "x")
        self.g = symbol_math.Function("exp(y)", "y")

    def test_init(self):
        self.assertEqual()


class TestSimplify(unittest.TestCase):
    pass


class TestDerivative(unittest.TestCase):
    pass


class TestReplaceVar(unittest.TestCase):
    pass



if __name__ == '__main__':
    unittest.main()