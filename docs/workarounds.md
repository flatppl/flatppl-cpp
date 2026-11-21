# IREE workarounds

`flatppl-build` keeps the FlatPPL output in `query.mlir` and passes a separate
`query.iree.mlir` to stock `iree-compile`. The two files live in the model build
cache. The build applies the workarounds in `iree_workarounds.py` automatically.

## Matrix–vector products with a dynamic row count

IREE 3.11 and 3.14 can introduce a `stablehlo.reshape` with a dynamic result
when lowering `stablehlo.dot_general` for `?×P` times `P`. The workaround
uses `stablehlo.dot` for this unbatched contraction over axes `[1]` and `[0]`.
It matches the `f32` and `f64` forms emitted by FlatPPL and preserves the
operands, result type, and precision. Fixed-size, batched, and
algorithm-qualified contractions pass through unchanged.

Use this StableHLO module to check whether the workaround is still needed.

```mlir
module {
  func.func @matvec(%x: tensor<?x2xf32>, %b: tensor<2xf32>) -> tensor<?xf32> {
    %y = stablehlo.dot_general %x, %b, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<?x2xf32>, tensor<2xf32>) -> tensor<?xf32>
    return %y : tensor<?xf32>
  }
}
```

Remove the workaround once the minimum supported IREE version compiles this
module and runs it at multiple row counts.

## Double-precision math on macOS arm64

The pinned IREE compiler's embedded ELF linker fails to resolve `exp`, `log`,
and `sin` in the `f64` Poisson regression example. Build the example in `f32`.
