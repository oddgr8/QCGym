class GenericFidelity(object):
    """Fidelity base class"""

    def __call__(self, unitary, target):
        raise NotImplementedError(
            "This is a Fidelity base class. Use other specialized fidelities.")

    def __str__(self):
        return "Generic fidelity"
