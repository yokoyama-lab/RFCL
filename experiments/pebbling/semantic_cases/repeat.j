// the second call starts where the first ended but does not restore it
procedure f(int x, int y)
  y += x

procedure main()
  int a
  int b
  a += 3
  call f(a, b)
  call f(a, b)
