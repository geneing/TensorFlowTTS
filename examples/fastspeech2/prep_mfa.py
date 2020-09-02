# -*- coding: utf-8 -*-
# Copyright 2020 Eugene Ingerman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging

import argparse

def main():
    """Run fastspeech2 decoding from folder."""
    parser = argparse.ArgumentParser(
        description="Copy text files to the directory with wav files "
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        required=True,
        help="directory including ids/durations files.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )

    args = parser.parse_args()
    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    with open(os.path.join(args.rootdir, "metadata.csv"), encoding="utf-8") as f:
        items = [line.strip().split('|') for line in f]

    for fname, text, _ in items:
        with open(os.path.join(args.rootdir, "wavs", fname+".txt"), encoding="utf-8", mode='w') as f:
            logging.info(f'{fname} {text}')
            f.write(text)

if __name__ == "__main__":
    main()
