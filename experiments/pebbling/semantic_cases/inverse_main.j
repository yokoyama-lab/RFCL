// the computation is an uncall of a generator; nothing undoes it
// syntactic: almost all work is "reverse"; semantic: all forward
procedure gen(int r, int x)
  x += r * 3
  x += 1
  x += r

procedure main()
  int r
  int x
  x += 17
  uncall gen(r, x)
