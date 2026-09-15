#!/usr/bin/env bash
set -euo pipefail

uv run --group teaching python -m ipykernel install \
  --sys-prefix \
  --name invoiceops-ml-py312 \
  --display-name "InvoiceOps ML Python 3.12"
