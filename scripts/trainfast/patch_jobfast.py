import sys
p, blk = sys.argv[1], open(sys.argv[2]).read()
s = open(p).read()
a = s.index('PREBUILD_ARG=""\n'); b = s.index('CPIDS=""\n', a)
s = s[:a] + blk + s[b:]
open(p, "w").write(s); print("patched")
