// uncall used to compute, call used to clean up: the call is the reverse work
procedure gen(int r, int x)
  x += r * 3
  x += 1
  x += r

procedure main()
  int r
  int x
  int c
  x += 17
  uncall gen(r, x)
  c += r
  call gen(r, x)
