// a literal passed to a constant parameter: the uncall still exactly undoes the call
procedure f(int y, constant int k)
  y += k
  y += 1

procedure main()
  int b
  int c
  call f(b, 3)
  c += b
  uncall f(b, 3)
