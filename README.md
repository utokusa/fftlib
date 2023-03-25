# fftlib
FFT library

## Development

You can find available development commands in `Makefile`.

## Serialize format for complex number arrays

`[0+1j, 2+3j, 4+5j]` would be:

```
0 1
2 3
4 5
```

Real number arrays like `[0, 1, 2]` can be serialized as:

```
0
1
2
```

