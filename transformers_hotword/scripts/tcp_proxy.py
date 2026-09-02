#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging


LOGGER = logging.getLogger("qwen3_asr.gateway")


async def copy_stream(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while data := await reader.read(1024 * 1024):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except ConnectionError:
            pass


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_host: str,
    upstream_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(
            upstream_host, upstream_port
        )
    except OSError as exc:
        LOGGER.warning(
            "upstream unavailable peer=%s error=%s", peer, type(exc).__name__
        )
        client_writer.close()
        await client_writer.wait_closed()
        return
    left = asyncio.create_task(copy_stream(client_reader, upstream_writer))
    right = asyncio.create_task(copy_stream(upstream_reader, client_writer))
    _done, pending = await asyncio.wait(
        (left, right), return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


async def run(
    listen_host: str, listen_port: int, upstream_host: str, upstream_port: int
) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(
            reader, writer, upstream_host, upstream_port
        ),
        listen_host,
        listen_port,
    )
    LOGGER.info(
        "listening=%s:%s upstream=%s:%s",
        listen_host,
        listen_port,
        upstream_host,
        upstream_port,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8000)
    parser.add_argument("--upstream-host", default="asr")
    parser.add_argument("--upstream-port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(
        run(args.listen_host, args.listen_port, args.upstream_host, args.upstream_port)
    )


if __name__ == "__main__":
    main()
