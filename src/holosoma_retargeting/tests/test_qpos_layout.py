import numpy as np

from holosoma_retargeting.src.qpos_layout import convert_qpos_layout, detect_qpos_layout


def test_omniretarget_to_native_field_order() -> None:
    omni = np.zeros((2, 43))
    omni[:, :4] = [1.0, 0.0, 0.0, 0.0]
    omni[:, 4:7] = [0.1, 0.2, 0.8]
    omni[:, 7:-7] = np.arange(29)
    omni[:, -7:-3] = [0.0, 0.0, 0.0, 1.0]
    omni[:, -3:] = [0.4, 0.5, 0.6]

    native = convert_qpos_layout(omni, "omniretarget", "native")

    np.testing.assert_allclose(native[:, :3], omni[:, 4:7])
    np.testing.assert_allclose(native[:, 3:7], omni[:, :4])
    np.testing.assert_allclose(native[:, 7:-7], omni[:, 7:-7])
    np.testing.assert_allclose(native[:, -7:-4], omni[:, -3:])
    np.testing.assert_allclose(native[:, -4:], omni[:, -7:-3])
    assert detect_qpos_layout(omni) == "omniretarget"
    assert detect_qpos_layout(native) == "native"


def test_qpos_layout_round_trip() -> None:
    rng = np.random.default_rng(7)
    native = rng.normal(size=(3, 43))
    omni = convert_qpos_layout(native, "native", "omniretarget")
    restored = convert_qpos_layout(omni, "omniretarget", "native")
    np.testing.assert_array_equal(restored, native)
