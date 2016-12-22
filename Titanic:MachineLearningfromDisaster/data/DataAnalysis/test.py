def fab(n):
    s, a, b = 0, 1, 1
    while s < n:
        yield a
        s, a, b = s + 1, b, a + b

for i in fab(5):
    print(i)
