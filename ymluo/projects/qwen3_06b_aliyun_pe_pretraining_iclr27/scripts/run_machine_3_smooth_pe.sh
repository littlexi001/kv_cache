#!/usr/bin/env bash
exec bash "$(cd "$(dirname "$0")" && pwd)/run_four_machine_worker.sh" 3
