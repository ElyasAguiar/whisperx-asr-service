#!/bin/bash
python3 -m app.server &
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9000