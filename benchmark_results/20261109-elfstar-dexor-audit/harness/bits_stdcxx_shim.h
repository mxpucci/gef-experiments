// Compat shim for non-libstdc++ toolchains: Serf's Elf* uses only std headers below.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
