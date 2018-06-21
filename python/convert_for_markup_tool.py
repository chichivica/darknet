import os
import sys
import json


def process_file(filename):
    def _convert_block(block_lines):
        number = int(block_lines[0][0])
        samples = []
        for b_l in block_lines:
            sample = {
                "class": int(b_l[1]),
                "flags": 0,
                "rect": [
                    float(b_l[2]),
                    float(b_l[3]),
                    float(b_l[4]),
                    float(b_l[5]),
                ]
            }
            samples.append(sample)

        return {
            "number": number,
            "samples": samples
        }

    lines = []
    with open(filename, "r") as f:
        for line in f:
            lines.append(line.replace("\n", "").split("\t"))

    current_block = []
    if len(lines) > 0:
        current_block.append(lines[0])
    result = {
        "frames": [],

    }
    for i in range(1, len(lines)):
        if lines[i - 1][0] == lines[i][0]:
            current_block.append(lines[i])
        else:
            block = _convert_block(current_block)
            result["frames"].append(block)
            current_block = [lines[i]]

    with open(filename.replace(".txt", ".json"), "w") as json_f:
        json.dump(result, json_f, indent=4)


if __name__ == '__main__':
    file_before = sys.argv[1]
    process_file(file_before)
