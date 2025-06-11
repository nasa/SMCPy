from smcpy.utils.progress_bar import progress_bar


class MockSampler:
    def __init__(self, show_progress_bar=False):
        self._show_progress_bar = show_progress_bar
        self.ran_func = False

    @progress_bar
    def func(self):
        self.ran_func = True


def test_run_progress_func():
    sampler = MockSampler(show_progress_bar=True)
    sampler.func()
    assert sampler._show_progress_bar == True
    assert sampler.ran_func == True


def test_dont_run_progress_func():
    sampler = MockSampler()
    sampler.func()
    assert sampler._show_progress_bar == False
    assert sampler.ran_func == False
