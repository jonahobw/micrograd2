"""Unit tests for the micrograd Value class to ensure equivalence with PyTorch operations."""
import unittest
import torch
from micrograd2.value import Value


class TestMicrogradEquivalence(unittest.TestCase):
    """Unit tests for the micrograd Value class to ensure equivalence with PyTorch operations."""

    def assert_almost_equal(self, val1, val2, places=7, msg=None):
        """Assert that two values are equal within a certain tolerance."""
        self.assertTrue(abs(val1 - val2) < 10**-places, msg=msg)

    def assert_mg_equivalent_pt(self, mg_value, torch_tensor, msg=None, grad=False):
        """Assert that a micrograd Value and a PyTorch tensor are equal."""

        self.assertAlmostEqual(
            mg_value.val, torch_tensor.item(), places=7, msg=f"{msg} - Value mismatch"
        )

        if grad:
            self.assertAlmostEqual(
                mg_value.grad,
                torch_tensor.grad.item() if torch_tensor.grad is not None else 1.0,
                places=7,
                msg=f"{msg} - Gradient mismatch",
            )

    def run_test_case(self, mg_result, torch_result, operand_pairs, msg_prefix=""):
        """Helper function to run a forward and backward pass and compare results."""

        self.assert_mg_equivalent_pt(
            mg_result, torch_result, f"{msg_prefix} - Forward pass"
        )

        # Backward pass
        mg_result.backward()
        torch_result.backward()
        for i, (mg_val, torch_val) in enumerate(operand_pairs):
            self.assert_mg_equivalent_pt(
                mg_val, torch_val, f"{msg_prefix} pair {i+1} - Backward pass", grad=True
            )

    def test_addition(self):
        """Test addition operation."""
        a_mg = Value(2.0, name="a")
        b_mg = Value(-3.0, name="b")
        c_mg = a_mg + b_mg

        a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
        c_torch = a_torch + b_torch

        self.run_test_case(
            c_mg, c_torch, [[a_mg, a_torch], [b_mg, b_torch]], "Addition"
        )

    def test_multiplication(self):
        """Test multiplication operation."""
        a_mg = Value(2.0, name="a")
        b_mg = Value(-3.0, name="b")
        c_mg = a_mg * b_mg

        a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
        c_torch = a_torch * b_torch

        self.run_test_case(
            c_mg, c_torch, [[a_mg, a_torch], [b_mg, b_torch]], "Multiplication"
        )

    def test_division(self):
        """Test division operation."""
        a_mg = Value(5.0, name="a")
        b_mg = Value(2.0, name="b")
        c_mg = a_mg / b_mg

        a_torch = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        c_torch = a_torch / b_torch

        self.run_test_case(
            c_mg, c_torch, [[a_mg, a_torch], [b_mg, b_torch]], "Division"
        )

    def test_power(self):
        """Test power operation."""
        a_mg = Value(2.0, name="a")
        b_mg = Value(3.0, name="b")
        c_mg = a_mg**b_mg

        a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        c_torch = a_torch**b_torch

        self.run_test_case(c_mg, c_torch, [[a_mg, a_torch]], "Power")

    def test_relu(self):
        """Test ReLU operation."""
        x_mg = Value(-1.0, name="x")
        out_mg = x_mg.relu()

        x_torch = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
        out_torch = torch.relu(x_torch)

        self.run_test_case(
            out_mg, out_torch, [[x_mg, x_torch]], "ReLU (negative input)"
        )

        x_mg = Value(2.0, name="x")
        out_mg = x_mg.relu()

        x_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        out_torch = torch.relu(x_torch)

        self.run_test_case(
            out_mg, out_torch, [[x_mg, x_torch]], "ReLU (positive input)"
        )

    def test_sigmoid(self):
        """Test sigmoid operation."""
        x_mg = Value(1.0, name="x")
        out_mg = x_mg.sigmoid()

        x_torch = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        out_torch = torch.sigmoid(x_torch)

        self.run_test_case(out_mg, out_torch, [[x_mg, x_torch]], "Sigmoid")

    def test_abs(self):
        """Test absolute value operation."""
        x_mg = Value(-5.0, name="x")
        out_mg = x_mg.abs()

        x_torch = torch.tensor(-5.0, dtype=torch.float64, requires_grad=True)
        out_torch = torch.abs(x_torch)

        self.run_test_case(out_mg, out_torch, [[x_mg, x_torch]], "Abs (negative input)")

        x_mg = Value(3.0, name="x")
        out_mg = x_mg.abs()

        x_torch = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        out_torch = torch.abs(x_torch)

        self.run_test_case(out_mg, out_torch, [[x_mg, x_torch]], "Abs (positive input)")

    def test_complex_expression(self):
        """Test complex expression involving multiple operations."""
        a_mg = Value(2.0, name="a")
        b_mg = Value(-3.0, name="b")
        c_mg = Value(5.0, name="c")
        d_mg = (a_mg * b_mg + c_mg).relu()

        a_torch = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
        c_torch = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        d_torch = torch.relu(a_torch * b_torch + c_torch)

        self.run_test_case(
            d_mg,
            d_torch,
            [[a_mg, a_torch], [b_mg, b_torch], [c_mg, c_torch]],
            "Complex Expression",
        )

    def test_complex_expression_2(self):
        """Test another complex expression involving multiple operations."""
        x_mg = Value(1.5, name="x")
        y_mg = (x_mg**2).sigmoid() / (x_mg + 1)

        x_torch = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        y_torch = (x_torch**2).sigmoid() / (x_torch + 1)

        self.run_test_case(y_mg, y_torch, [[x_mg, x_torch]], "Complex Expression 2")

    def test_complex_expression_3(self):
        """Test another complex expression involving multiple operations."""
        a_mg = Value(-4.0)
        b_mg = Value(2.0)
        c_mg = a_mg + b_mg
        d_mg = a_mg * b_mg + b_mg**3
        c_mg += c_mg + 1
        c_mg += 1 + c_mg + (-a_mg)
        d_mg += d_mg * 2 + (b_mg + a_mg).relu()
        d_mg += 3 * d_mg + (b_mg - a_mg).relu()
        e_mg = c_mg - d_mg
        f_mg = e_mg**2
        g_mg = f_mg / 2.0
        g_mg += 10.0 / f_mg

        a_torch = torch.Tensor([-4.0]).double()
        b_torch = torch.Tensor([2.0]).double()
        a_torch.requires_grad = True
        b_torch.requires_grad = True
        c_torch = a_torch + b_torch
        d_torch = a_torch * b_torch + b_torch**3
        c_torch = c_torch + c_torch + 1
        c_torch = c_torch + 1 + c_torch + (-a_torch)
        d_torch = d_torch + d_torch * 2 + (b_torch + a_torch).relu()
        d_torch = d_torch + 3 * d_torch + (b_torch - a_torch).relu()
        e_torch = c_torch - d_torch
        f_torch = e_torch**2
        g_torch = f_torch / 2.0
        g_torch = g_torch + 10.0 / f_torch

        self.run_test_case(g_mg, g_torch, [[a_mg, a_torch], [b_mg, b_torch]], "Complex Expression 3")


if __name__ == "__main__":
    unittest.main()
