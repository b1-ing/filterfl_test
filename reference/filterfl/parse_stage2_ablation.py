import re
import sys

for path in sys.argv[1:]:
    ma, asr = [], []
    with open(path) as f:
        for line in f:
            if '___Global_Test_Poison:False' in line:
                m = re.search(r'\(([\d.]+)%\)', line)
                ma.append(float(m.group(1)))
            elif '___Global_Test_Poison:True' in line:
                m = re.search(r'\(([\d.]+)%\)', line)
                asr.append(float(m.group(1)))
    def avg(l, a, b):
        return sum(l[a:b]) / len(l[a:b])
    print(f"=== {path} ===")
    print(f"MA  1-10,11-20,21-30: {avg(ma,0,10):.2f} {avg(ma,10,20):.2f} {avg(ma,20,30):.2f}")
    print(f"ASR 1-10,11-20,21-30: {avg(asr,0,10):.2f} {avg(asr,10,20):.2f} {avg(asr,20,30):.2f}")
