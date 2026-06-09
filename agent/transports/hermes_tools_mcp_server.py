"""Deprecated compatibility module for ``gpucloud_tools_mcp_server``."""

from agent.transports.gpucloud_tools_mcp_server import *  # noqa: F401,F403


if __name__ == "__main__":
    from agent.transports.gpucloud_tools_mcp_server import main

    main()
