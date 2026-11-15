"""Handwriting and print text-line boxes of one page. python samples/predict.py IMAGE [MODEL] [CONF]"""
import sys

from hwdet import Detector

det = Detector(sys.argv[2] if len(sys.argv) > 2 else "hwdet-s", conf=float(sys.argv[3]) if len(sys.argv) > 3 else None)
page = det.predict(sys.argv[1])
for b in page.boxes:
    print(f"{b.label:12s} {b.conf:.2f}", *map(int, b.xyxy))
print("handwriting on the page:", page.has_handwriting())
