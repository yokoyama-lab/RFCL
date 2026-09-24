// uncall on different data: not an uncomputation of the earlier call
procedure f(int x, int y)
  y += x * 2
  y += x

procedure main()
  int a
  int b
  int c
  int d
  a += 3
  c += 3
  call f(a, b)
  d += 9
  uncall f(c, d)
