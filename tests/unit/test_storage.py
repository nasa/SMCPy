import copy
import numpy as np
import pytest

from smcpy.utils.storage import HDF5Storage, InMemoryStorage, PickleStorage


class DummyParticles:
    def __init__(self, params, log_likes, log_weights):
        self.attrs = {}
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.param_dict = params
        self.params = np.vstack([val for val in params.values()]).T


@pytest.fixture
def mock_particles(mocker):
    mocker.patch("smcpy.utils.storage.Particles", new=DummyParticles)
    param_dict = {
        "1": np.ones(10) * 2,
        "0": np.zeros(10),
        "2": np.ones(10),
    }
    log_likes = np.ones((10, 1))
    log_weights = np.ones((10, 1))
    mock_particles = DummyParticles(param_dict, log_likes, log_weights)
    mock_particles.total_unnorm_log_weight = 99
    mock_particles.attrs = {"phi": 2, "mutation_ratio": 1}
    return mock_particles


parametrize_file_storage = pytest.mark.parametrize(
    "storage_type, extension_type", [(HDF5Storage, "h5"), (PickleStorage, "pkl")]
)


class TestInMemoryStorage:
    def test_save(self, mocker):
        mocks = [mocker.Mock() for _ in range(3)]
        for i, mock in enumerate(mocks):
            mock.attrs = {"phi": i / 10, "mutation_ratio": i / len(mocks)}

        storage = InMemoryStorage()
        storage.save_step(mocks[0])
        storage.save_step(mocks[1])
        storage.save_step(mocks[2])

        assert storage[0] == mocks[0]
        assert storage[1] == mocks[1]
        assert storage[2] == mocks[2]
        assert storage[-1] == mocks[2]
        assert len(storage) == 3
        assert storage.phi_sequence == [0, 0.1, 0.2]
        assert storage.mut_ratio_sequence == [0, 1 / 3, 2 / 3]

        with pytest.raises(IndexError):
            storage[3]
        with pytest.raises(IndexError):
            storage[-4]

    @pytest.mark.parametrize("steps", [2, 3, 4, 10])
    def test_marginal_log_likelihood_calculation(self, mocker, steps):
        unnorm_log_weights = np.zeros((5, 1))
        expected_mll = [1] + [5 ** (i + 1) for i in range(steps)]

        mocked_particles = mocker.Mock()
        mocked_particles.total_unnorm_log_weight = np.log(5)

        storage = InMemoryStorage()
        storage._step_list = [mocked_particles for _ in range(steps)]

        mll = storage.estimate_marginal_log_likelihoods()
        np.testing.assert_array_almost_equal(mll, np.log(expected_mll))

    def test_cannot_restart(self):
        storage = InMemoryStorage()
        assert not storage.is_restart


@parametrize_file_storage
class TestFileStorage:
    def test_no_restart_file_not_exist(self, tmpdir, storage_type, extension_type):
        f = tmpdir / "test.txt"
        storage = storage_type(str(f))
        assert not storage.is_restart

    def test_not_exist_write(self, tmpdir, storage_type, extension_type):
        f = tmpdir / "test.txt"
        storage = storage_type(str(f), mode="w")
        assert not storage.is_restart

    def test_no_restart_write_mode(self, tmpdir, storage_type, extension_type):
        f = tmpdir / "test.txt"
        f.write_text("test", encoding="ascii")
        storage = storage_type(str(f), mode="w")
        assert not storage.is_restart

    def test_restart(self, tmpdir, mocker, storage_type, extension_type):
        f = tmpdir / f"test.{extension_type}"
        f.write_text("test", encoding="ascii")
        mocker.patch(
            f"smcpy.utils.storage.{storage_type.__name__}._init_length_on_restart"
        )
        storage = storage_type(str(f))
        assert storage.is_restart

    def test_os_scandir_params(self, mocker, tmpdir, storage_type, extension_type):
        filename = tmpdir / f"test.{extension_type}"

        mock_scandir = mocker.patch("smcpy.utils.storage.os.scandir")
        storage = storage_type(filename=filename)
        storage._open_file("a")

        mock_scandir.assert_called_once_with(tmpdir)

    @pytest.mark.parametrize("mode", [1, 1000, "b", "abc", "A", "W"])
    def test_invalid_input_mode(self, tmpdir, mode, storage_type, extension_type):
        filename = tmpdir / f"test.{extension_type}"
        with pytest.raises(ValueError):
            storage_type(filename=filename, mode=mode)

    def test_save(self, tmpdir, mock_particles, storage_type, extension_type):
        storage = storage_type(filename=tmpdir / f"test.{extension_type}")
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)

        def check_particles(p):
            np.testing.assert_array_equal(p.params, mock_particles.params)
            np.testing.assert_array_equal(p.log_likes, mock_particles.log_likes)
            np.testing.assert_array_equal(p.log_weights, mock_particles.log_weights)
            for key, val in mock_particles.param_dict.items():
                np.testing.assert_array_equal(p.param_dict[key], val)
            assert p.attrs == {
                "phi": 2,
                "mutation_ratio": 1,
                "total_unnorm_log_weight": 99,
            }

        assert isinstance(storage[0], DummyParticles)
        [check_particles(p) for p in storage]
        assert storage.phi_sequence == [2, 2]
        assert storage.mut_ratio_sequence == [1, 1]

    def test_load_existing(self, tmpdir, mock_particles, storage_type, extension_type):
        storage = storage_type(filename=tmpdir / f"test.{extension_type}")
        mp2 = copy.deepcopy(mock_particles)
        mp2.attrs["phi"] = 1
        mp2.attrs["mutation_ratio"] = 0
        [storage.save_step(mock_particles) for _ in range(12)]
        storage.save_step(mp2)

        storage = storage_type(filename=tmpdir / f"test.{extension_type}")
        storage.save_step(mock_particles)

        assert storage.phi_sequence == [2] * 12 + [1, 2]
        assert storage.mut_ratio_sequence == [1] * 12 + [0, 1]

    def test_length(self, tmpdir, mock_particles, storage_type, extension_type):
        storage = storage_type(filename=tmpdir / f"test.{extension_type}")
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)

        assert len(storage) == 2

    def test_neg_indexing(self, tmpdir, mock_particles, storage_type, extension_type):
        storage = storage_type(filename=tmpdir / f"test.{extension_type}")
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)

        assert storage[-1]
        assert storage[-2]
        assert storage[0]
        assert storage[1]

        with pytest.raises(IndexError):
            storage[2]
        with pytest.raises(IndexError):
            storage[-3]


class TestHDF5Storage:
    def test_overwrite_mode(self, mocker, tmpdir):
        h5_mock = mocker.patch("smcpy.utils.storage.h5py.File")

        filename = tmpdir / "test.h5"
        storage = HDF5Storage(filename=filename)

        storage._open_file("w")

        h5_mock.assert_called_once_with(filename, "w", track_order=True)

    def test_mode_default(self, tmpdir):
        filename = tmpdir / "test.h5"
        storage = HDF5Storage(filename=filename)

        assert storage._mode == "a"

    @pytest.mark.parametrize("mode", ["w", "a"])
    def test_mode_write(self, tmpdir, mode):
        filename = tmpdir / "test.h5"
        storage = HDF5Storage(filename=filename, mode=mode)

        assert storage._mode == mode

    def test_first_save_step_changes_mode(self, tmpdir, mock_particles):
        filename = tmpdir / "test.h5"
        storage = HDF5Storage(filename=filename, mode="w")

        storage.save_step(mock_particles)
        assert storage._mode == "a"


class TestPickleStorage:
    def test_open_wb_resets_len(self, tmpdir, mock_particles):
        """Opening in write-binary mode must reset _len to 0."""
        filename = tmpdir / "test.pkl"
        storage = PickleStorage(filename=filename)
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)
        assert len(storage) == 2

        f = storage._open_file("wb")
        f.close()
        assert len(storage) == 0

    def test_overwrite_mode(self, mocker, tmpdir):
        pickle_mock = mocker.patch("smcpy.utils.storage.open")

        filename = tmpdir / "test.pkl"
        storage = PickleStorage(filename=filename)

        storage._open_file("wb")

        pickle_mock.assert_called_once_with(filename, "wb")

    def test_mode_default(self, tmpdir):
        filename = tmpdir / "test.pkl"
        storage = PickleStorage(filename=filename)

        assert storage._mode == "ab"

    @pytest.mark.parametrize("mode", ["w", "a"])
    def test_mode_write(self, tmpdir, mode):
        filename = tmpdir / "test.pkl"
        storage = PickleStorage(filename=filename, mode=mode)

        assert storage._mode == mode + "b"

    def test_first_save_step_changes_mode(self, tmpdir, mock_particles):
        filename = tmpdir / "test.h5"
        storage = PickleStorage(filename=filename, mode="w")

        storage.save_step(mock_particles)
        assert storage._mode == "ab"

    def test_array_objects(self, tmpdir, mock_particles):
        storage = PickleStorage(filename=tmpdir / f"test.pkl")
        mock_particles.params = [list() for _ in range(3)]
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)

        def test_changed_array(p):
            for obj in p.params:
                print(obj)
                assert isinstance(obj, list)

        [test_changed_array(p) for p in storage]

    def test_len_stable_after_repeated_reads(self, tmpdir, mock_particles):
        """Multiple reads without writes must not increment _len."""
        storage = PickleStorage(filename=tmpdir / "test.pkl")
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)
        assert len(storage) == 3

        _ = storage[0]
        _ = storage[1]
        _ = storage[-1]
        _ = storage[0]
        assert len(storage) == 3

    def test_len_stable_after_iteration(self, tmpdir, mock_particles):
        """Iterating all items must not inflate _len."""
        storage = PickleStorage(filename=tmpdir / "test.pkl")
        for _ in range(4):
            storage.save_step(mock_particles)
        assert len(storage) == 4

        list(storage)
        assert len(storage) == 4

        list(storage)
        assert len(storage) == 4

    def test_len_stable_after_property_access(self, tmpdir, mock_particles):
        """Accessing phi_sequence/mut_ratio_sequence must not inflate _len."""
        storage = PickleStorage(filename=tmpdir / "test.pkl")
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)
        assert len(storage) == 2

        _ = storage.phi_sequence
        _ = storage.mut_ratio_sequence
        _ = storage.phi_sequence
        assert len(storage) == 2

    def test_overwrite_resets_offset_and_len(self, tmpdir, mock_particles):
        """wb mode must reset both _len and _last_scan_offset so subsequent
        saves count correctly from scratch."""
        filename = tmpdir / "test.pkl"
        storage = PickleStorage(filename=filename)
        storage.save_step(mock_particles)
        storage.save_step(mock_particles)
        assert len(storage) == 2

        f = storage._open_file("wb")
        f.close()
        assert len(storage) == 0
        assert storage._last_scan_offset == 0

        storage.save_step(mock_particles)
        assert len(storage) == 1

    def test_interleaved_reads_and_writes(self, tmpdir, mock_particles):
        """Interleaving reads and writes must maintain correct count."""
        storage = PickleStorage(filename=tmpdir / "test.pkl")
        storage.save_step(mock_particles)
        assert len(storage) == 1

        _ = storage[0]
        assert len(storage) == 1

        storage.save_step(mock_particles)
        assert len(storage) == 2

        _ = storage[0]
        _ = storage[1]
        assert len(storage) == 2

        storage.save_step(mock_particles)
        assert len(storage) == 3

    def test_restart_reads_dont_double_count(self, tmpdir, mock_particles):
        """After restart from existing file, reads must not double-count."""
        filename = tmpdir / "test.pkl"
        storage = PickleStorage(filename=filename)
        for _ in range(3):
            storage.save_step(mock_particles)

        storage2 = PickleStorage(filename=filename)
        assert len(storage2) == 3

        _ = storage2[0]
        _ = storage2[-1]
        list(storage2)
        assert len(storage2) == 3

        storage2.save_step(mock_particles)
        assert len(storage2) == 4

    def test_getitem_seeks_to_correct_object(self, tmpdir, mock_particles):
        """__getitem__ must deserialize the correct object via direct seek."""
        storage = PickleStorage(filename=tmpdir / "test.pkl")

        steps = []
        for i in range(5):
            mp = copy.deepcopy(mock_particles)
            mp.attrs["phi"] = i
            steps.append(mp)
            storage.save_step(mp)

        for i in range(5):
            assert storage[i].attrs["phi"] == i

        assert storage[-1].attrs["phi"] == 4
        assert storage[-5].attrs["phi"] == 0

    def test_getitem_after_restart(self, tmpdir, mock_particles):
        """After restart, __getitem__ must return correct objects by index."""
        filename = tmpdir / "test.pkl"
        storage = PickleStorage(filename=filename)

        for i in range(3):
            mp = copy.deepcopy(mock_particles)
            mp.attrs["phi"] = i * 10
            storage.save_step(mp)

        storage2 = PickleStorage(filename=filename)
        assert storage2[0].attrs["phi"] == 0
        assert storage2[1].attrs["phi"] == 10
        assert storage2[2].attrs["phi"] == 20
        assert storage2[-1].attrs["phi"] == 20

    def test_getitem_after_overwrite(self, tmpdir, mock_particles):
        """After wb overwrite and new saves, __getitem__ reflects new data only."""
        filename = tmpdir / "test.pkl"
        storage = PickleStorage(filename=filename)

        for i in range(3):
            mp = copy.deepcopy(mock_particles)
            mp.attrs["phi"] = i
            storage.save_step(mp)

        f = storage._open_file("wb")
        f.close()

        mp_new = copy.deepcopy(mock_particles)
        mp_new.attrs["phi"] = 99
        storage.save_step(mp_new)

        assert len(storage) == 1
        assert storage[0].attrs["phi"] == 99

        with pytest.raises(IndexError):
            storage[1]

    def test_getitem_random_access_order(self, tmpdir, mock_particles):
        """Accessing items out of order must return correct objects each time."""
        storage = PickleStorage(filename=tmpdir / "test.pkl")

        for i in range(4):
            mp = copy.deepcopy(mock_particles)
            mp.attrs["phi"] = i
            storage.save_step(mp)

        assert storage[3].attrs["phi"] == 3
        assert storage[-1].attrs["phi"] == 3
        assert storage[0].attrs["phi"] == 0
        assert storage[2].attrs["phi"] == 2
        assert storage[1].attrs["phi"] == 1
        assert storage[3].attrs["phi"] == 3
        assert storage[0].attrs["phi"] == 0

    def test_byte_offsets_grow_with_saves(self, tmpdir, mock_particles):
        """_byte_offsets must have one entry per saved step."""
        storage = PickleStorage(filename=tmpdir / "test.pkl")

        for i in range(5):
            storage.save_step(mock_particles)
            assert len(storage._byte_offsets) == i + 1

        assert storage._byte_offsets[0] == 0
        assert all(
            storage._byte_offsets[i] < storage._byte_offsets[i + 1] for i in range(4)
        )
