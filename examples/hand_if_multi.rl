(x flag) (x y flag) ()
start: entry;
if (!= flag 0) goto tbranch else ebranch;
tbranch: from start;
y ^= x;
y += 1;
goto join;
ebranch: from start;
y -= x;
goto join;
join: fi (!= flag 0) from tbranch else ebranch;
exit;
