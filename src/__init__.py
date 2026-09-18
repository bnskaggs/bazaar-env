# Not a package. The prime CLI's `env push` layout check looks for any
# root-level *.py or an immediate <subdir>/__init__.py; with a src/ layout
# neither exists, so this file exists to satisfy the check. It is not
# included in the built wheel (hatch packages only src/bazaar_env and
# src/bazaar.py).
