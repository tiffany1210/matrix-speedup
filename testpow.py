from utils import *
from unittest import TestCase

class TestPow(TestCase):
    def test_pow(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(33, 33, seed=5)
        is_correct, speed_up = compute([dp_mat, 10], [nc_mat, 10], "pow")
        self.assertTrue(is_correct)


