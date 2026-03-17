(n) (acc) ()
init: entry;
goto loop;
loop: fi (= acc 0) from init else back;
acc += 1;
if (= n 0) goto done else step;
step: from loop;
n -= 1;
acc += 1;
goto back;
back: from step;
goto loop;
done: from loop;
exit;
