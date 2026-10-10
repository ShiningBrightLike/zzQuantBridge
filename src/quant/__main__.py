"""Allow ``python -m quant`` to use the CLI entry point."""

from .cli import main

raise SystemExit(main())
