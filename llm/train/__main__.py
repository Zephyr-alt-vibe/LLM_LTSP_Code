"""Allow ``python -m llm_mmi.train`` to dispatch to :mod:`llm_mmi.train.cli`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
