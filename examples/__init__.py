import importlib


def get_example_setup(name):
    try:
        module = importlib.import_module("examples." + name)
        setup = getattr(module, "setup")
        return setup
    except Exception as e:
        print(e)
        raise RuntimeError(f"Cannot find example setup. Example name: {name}.")
