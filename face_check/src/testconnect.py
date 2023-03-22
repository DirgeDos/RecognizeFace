import sys


def sum(a, b, c):
    return a + b + c


if __name__ == "__main__":
    a = (int(sys.argv[1]))
    b = (int(sys.argv[2]))
    c = (int(sys.argv[3]))
    s = sum(a, b, c)
    print("finish!!!")
    print(s)
