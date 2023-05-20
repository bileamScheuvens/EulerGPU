using ProgressBars
using CUDA
using GLMakie

function S(a,b,c; reversed=false)
    F(n) = n > b ? n-c : F(a + F(a + F(a + F(a + n))))
    reversed ? F.(b:-1:0) : F.(1:b)
end

function solveNaive(a,b,c)
    S(a,b,c) |> sum
end

function solveCPU(a,b,c; modu=10^9)
    # setup
    start = BigInt(b+4(a-c)) % modu
    diminisher = sum(1:a-1) % modu
    stepup = 3a - 3c
    blocksum(start) = start * a - diminisher
    # first block
    acc = blocksum(start) % modu
    nblocks = Int(floor(b/a))
    # middle blocks
    for i in ProgressBar(1:nblocks-1)
        blockstart = start + i*stepup
        acc = (acc + blocksum(blockstart)) % modu
    end
    # last blocks
    (acc + (start + nblocks*stepup)*((b%a)+1) - sum(1:b%a)) % modu
end

function solveGPU(a,b,c; modu=10^9)
    # setup
    start = (b+4(a-c)) % modu
    stepup = 3a - 3c
    diminisher = sum(1:a-1) % modu
    nblocks = Int(floor(b/a))
    # index
    gpumem = CuArray(1:nblocks-1)
    # blockstarts
    gpumem = start .+ (gpumem * stepup)
    # blocksum
    gpumem = ((gpumem.%modu * a.%modu) .- diminisher) .% modu
    # first block
    acc = CuArray([start]) .* a .- diminisher
    # middle blocks
    modsum(a,b) = (a + b) % modu
    acc = acc .+ reduce(modsum, gpumem; init=0)
    # last blocks
    acc = acc .+ ((start + nblocks*stepup)%modu*((b%a)+1) .- (sum(1:b%a)%modu))
    # free memory
    gpumem = nothing
    CUDA.@allowscalar first(acc) % 10^9
end

a,b,c = 50, 2000, 40
GLMakie.scatter(S(50, 2000, 40))
@time solveNaive(a,b,c)
@time solveCPU(a,b,c)
@time solveGPU(a,b,c)

a,b,c = 21^7, 7^21, 12^7
@time solveNaive(a,b,c)
@time solveCPU(a,b,c)
@time solveGPU(a,b,c)