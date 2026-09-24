// self-inverse procedure called twice on the same data
procedure sw(int x, int y)
  x <=> y

procedure main()
  int a
  int b
  int c
  a += 1
  call sw(a, b)
  c += b
  call sw(a, b)
