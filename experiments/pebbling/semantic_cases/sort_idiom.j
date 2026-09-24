// call f(a, g); uncall f(o, g): the uncall clears the garbage g (partial
// inverse on g) while computing a new value in o, which differs from a
procedure f(int x, int g)
  g += x
  x += 5

procedure main()
  int a
  int g
  int o
  a += 3
  o += 8
  call f(a, g)
  uncall f(o, g)
