(n) (n output) (v1 v2 output-copied)
pre-init: entry;
goto init;

init: fi (= output-copied 0) from pre-init else pre-end;
v1 ^= 0;
v2 ^= 1;
goto test;

test: fi (= v1 0) from init else loop;
if (= n 0) goto copy-output-entry else loop;

loop: from test;
v1 += v2;
v1 <=> v2;
n -= 1;
goto test;

copy-output-entry: from test;
if (= output-copied 0) goto copy-output-main else copy-output-exit;

copy-output-main: from copy-output-entry;
output ^= v1;
output-copied ^= 1;
rgoto copy-output-exit;

copy-output-exit: from copy-output-entry;
rgoto copy-output-main;

pre-end: rfrom end;
goto init;

end: rfrom pre-end;
output-copied ^= 1;
exit;
