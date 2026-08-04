from pathlib import Path

import meshio
import numpy as np

from loudspeaker_axisym_fem.axisym_magnetics import load_tagged_meshio


def test_quad_split_preserves_physical_tag_order(tmp_path: Path, monkeypatch) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    mesh = meshio.Mesh(
        points,
        [("quad", np.array([[0, 1, 2, 3], [1, 4, 5, 2]]))],
        cell_data={"gmsh:physical": [np.array([6, 23])]},
    )
    monkeypatch.setattr(meshio, "read", lambda _: mesh)

    tagged = load_tagged_meshio(tmp_path / "unused.msh")

    np.testing.assert_array_equal(tagged.tri_domains, [6, 23, 6, 23])
    np.testing.assert_array_equal(
        tagged.triangles,
        [[0, 1, 2], [1, 4, 5], [0, 2, 3], [1, 5, 2]],
    )
