#!/usr/bin/env python3
"""
Download the tiny Shakespeare dataset.

requires requests: pip install requests
"""

import requests


def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)

    if response.status_code == 200:
        with open("data/tiny_shakespeare.txt", "w", encoding="utf-8") as f:
            f.write(response.text)

        print("Downloaded tiny Shakespeare dataset.")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")


if __name__ == "__main__":
    download_shakespeare()
