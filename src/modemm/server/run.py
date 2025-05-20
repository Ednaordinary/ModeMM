import argparse

import uvicorn

from .app import build


def run():
    parser = argparse.ArgumentParser(description='Run time config for Modemm')
    parser.add_argument("-c", "--config-file", type=str, help="Server config file path", default="config.json")
    parser.add_argument("-d", "--dynamic-config", help="Use the dynamic config loader", action='store_true')
    parser.add_argument("-p", "--port", type=int, help="Port to run the server on. It is recommended to change this "
                                                       "if running on the internet", default=14145)
    parser.add_argument("-n", "--no-advertise", help="Don't advertise docs and paths, making large scale "
                                                     "scraping more difficult", action="store_true")

    parser.add_argument("-m", "--max-income", type=int, help="Limits the maximum incoming content size",
                        default=4194432)

    args = parser.parse_args()
    app = build(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    run()
