def progress_bar(func):
    def wrapper(self, *args, **kwargs):
        if self._show_progress_bar:
            return func(self, *args, **kwargs)

    return wrapper
