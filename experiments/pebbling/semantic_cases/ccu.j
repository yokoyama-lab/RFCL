// compute-copy-uncompute: the uncall undoes the call -> ratio ~2 both ways
procedure f(int x, int y)
  y += x * 2
  y += 1
  y += x

procedure main()
  int a
  int b
  int c
  a += 3
  call f(a, b)
  c += b
  uncall f(a, b)
