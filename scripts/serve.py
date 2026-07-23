#!/usr/bin/env python
"""Launch the workshop serving API."""

import uvicorn


def main():
    uvicorn.run("mlops_workshop.serving:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
