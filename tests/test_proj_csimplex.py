"""
Test projection onto the capped simplex
"""
import pytest
import numpy as np
from proxlib.operators import proj_csimplex


@pytest.mark.parametrize("y", [np.array([[-10.0, 1.0, 1.0],
                                         [1.0, -10.0, 1.0],
                                         [1.0, 1.0, -10.0]])])
@pytest.mark.parametrize("h", [2.0])
@pytest.mark.parametrize("l", [0.0])
@pytest.mark.parametrize("u", [1.0])
def test_proj_csimplex(y, h, l, u):
    proj_csimplex(y, h, l, u)
    assert np.allclose(y, np.array([[0., 1., 1.],
                                    [1., 0., 1.],
                                    [1., 1., 0.]]))
