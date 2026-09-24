// CCU inside CCU: the inner pair in the outer uncall must not be counted twice
procedure f(int x, int y)
  y += x
  y += 1

procedure outer(int x, int z, int t)
  call f(x, t)
  z += t
  uncall f(x, t)

procedure main()
  int a
  int z
  int t
  int c
  a += 3
  call outer(a, z, t)
  c += z
  uncall outer(a, z, t)
