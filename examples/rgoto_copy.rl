# Bennett-style copy using rgoto/rfrom (Moriyama 2009, Section 2.1).
# Copies x to y using the reverse jump mechanism:
# 1. Forward: compute y ^= x, set copied flag
# 2. rgoto: reverse through the computation (undoing side effects)
# 3. rfrom: return to forward direction and exit with clean temps
(x) (x y) (copied)
pre-init: entry;
goto init;
init: fi (= copied 0) from pre-init else pre-end;
goto copy-entry;
copy-entry: from init;
if (= copied 0) goto copy-main else copy-exit;
copy-main: from copy-entry;
y ^= x;
copied ^= 1;
rgoto copy-exit;
copy-exit: from copy-entry;
rgoto copy-main;
pre-end: rfrom end;
goto init;
end: rfrom pre-end;
copied ^= 1;
exit;
