import threading


class ContextManager:
    _contexts = threading.local()

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, _type, _value, _traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        """Returns context stack"""
        if not hasattr(cls._contexts, "stack"):
            cls._contexts.stack = []
        return cls._contexts.stack

    @classmethod
    def get_context(cls):
        """Return the deepest context on the stack."""
        return cls.get_contexts()[-1]
