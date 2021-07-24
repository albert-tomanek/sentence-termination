import re
from dataclasses import dataclass

@dataclass
class Slice:
    start: float
    end: float
    text: str

def subtitle_reader(filepath):
    with open(filepath) as file:
        text = file.read()

    pattern = "([0-9]+)\n([0-9:,]+)\s*-->\s*([0-9:,]+)\n(.+?)\n(\n|$)"

    for match in re.finditer(pattern, text, re.DOTALL):
        start, end, text = match.group(2,3,4)
        yield Slice(decode_time(start), decode_time(end), text)

def write_subtitles(subs: [Slice], filepath):
    with open(filepath, 'w') as file:
        for i, slice in enumerate(subs):
            file.write(f"{i+1}\n{encode_time(slice.start)} --> {encode_time(slice.end)}\n{slice.text}\n\n")

def decode_time(time: str) -> float:
    match = re.search('(\d\d+):(\d\d):(\d\d),(\d\d\d)', time)
    hour, min, sec, msec = match.group(1,2,3,4)
    return 3600 * int(hour) + 60 * int(min) + int(sec) + int(msec) / 1000

def encode_time(time: float) -> str:
    hour = int(time // 3600)
    min  = int(time %  3600 // 60)
    sec  = int(time %  3600 %  60 // 1)
    msec = int(time %  3600 %  60 % 1 * 1000)
    return f'{hour:02}:{min:02}:{sec:02},{msec:03}'
